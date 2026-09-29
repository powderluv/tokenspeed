import argparse
import copy
import json
import shlex
from pathlib import Path

import pytest
import yaml
from pipeline import normalize_task
from run_agentx import build_command, validate_models, validate_result

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"


def test_requires_served_model_with_256k_context():
    validate_models({"data": [{"id": MODEL, "max_model_len": 262144}]}, MODEL)
    for models in (
        [],
        [{"id": "another-model", "max_model_len": 262144}],
        [{"id": MODEL, "max_model_len": 32768}],
        [{"id": MODEL}],
    ):
        with pytest.raises(ValueError):
            validate_models({"data": models}, MODEL)


def test_standard_command(tmp_path):
    args = argparse.Namespace(
        engine="vllm",
        engine_version="0.30.0",
        num_gpus=4,
        evalscope="/tmp/evalscope-agentx/bin/evalscope",
        model=MODEL,
        base_url="http://127.0.0.1:8000/",
        parallel=32,
        outputs_dir=tmp_path / "benchmark",
    )
    command = build_command(args, tmp_path / "tokenizer")
    options = dict(zip(command[2:-1:2], command[3:-1:2]))
    assert options["--parallel"] == "32"
    assert options["--duration"] == "1800"
    assert options["--seed"] == "20260707"
    assert options["--data-source"] == "modelscope"
    assert options["--tokenizer-path"] == str(tmp_path / "tokenizer")
    assert options["--url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert json.loads(options["--scenario"]) == {
        "name": "agentx",
        "mode": "benchmark",
        "variant": "256k",
        "engine": "vllm",
        "engine_version": "0.30.0",
        "num_gpus": 4,
    }


def test_rejects_invalid_or_incomparable_results(tmp_path):
    valid = {
        "status": "completed",
        "submission_valid": True,
        "concurrency": 8,
        "duration": 1800,
        "seed": 20260707,
        "scenario": {"mode": "benchmark", "variant": "256k"},
        "dataset": {"verified": True, "source": "modelscope"},
        "metrics": {"output_token_throughput": {"avg": 100}},
        "error_summary": [],
    }
    path = tmp_path / "agentx_summary.json"
    path.write_text(json.dumps(valid))
    assert validate_result(tmp_path, 8) == valid
    for overrides in (
        {"status": "failed"},
        {"submission_valid": False},
        {"concurrency": 1},
        {"duration": 60},
        {"scenario": {"mode": "smoke", "variant": "256k"}},
        {"dataset": {"verified": False, "source": "modelscope"}},
        {"dataset": {"verified": True, "source": "huggingface"}},
        {"error_summary": ["request cancelled"]},
    ):
        path.write_text(json.dumps({**copy.deepcopy(valid), **overrides}))
        with pytest.raises(ValueError):
            validate_result(tmp_path, 8)


@pytest.mark.parametrize("engine", ["sglang", "vllm"])
def test_sweep_configs_preserve_topology_and_speculation(engine):
    paths = sorted((REPO_ROOT / "test/ci/perf").glob(f"{engine}-*-agentx-p*.yaml"))
    assert len(paths) == 6
    parallels = set()
    server_commands = set()
    for path in paths:
        task = normalize_task(path, REPO_ROOT)
        assert task["triggers"] == ["slurm"]
        assert task["runner"]["labels"] == ["slurm-gb300-4gpu"]
        raw = yaml.safe_load(path.read_text())
        assert raw["slurm"] == {"nodes": 1, "gpus_per_node": 4}
        # The Slurm base image has pip but no importable uv module. Bootstrap
        # its executable before creating either isolated environment.
        bootstrap = shlex.split(raw["install"][0])
        assert bootstrap[:4] == ["python3", "-m", "pip", "install"]
        bootstrap_dir = bootstrap[bootstrap.index("--target") + 1]
        assert "uv==0.11.25" in bootstrap
        for install in (raw["install"][1], raw["perf"]["install"][0]):
            commands = install.split(" && ")
            assert len(commands) == 2
            assert all(
                shlex.split(command)[0] == f"{bootstrap_dir}/bin/uv"
                for command in commands
            )
        command = shlex.split(raw["perf"]["command"])
        parallels.add(int(command[command.index("--parallel") + 1]))
        assert command[command.index("--engine") + 1] == engine
        server_commands.add(raw["server"]["command"])
    assert parallels == {1, 2, 4, 8, 16, 32}
    assert len(server_commands) == 1
    server = shlex.split(server_commands.pop())
    if engine == "sglang":
        assert server[server.index("--tp-size") + 1] == "4"
        assert server[server.index("--ep-size") + 1] == "4"
        assert server[server.index("--speculative-num-steps") + 1] == "3"
        assert server[server.index("--speculative-num-draft-tokens") + 1] == "4"
        override = json.loads(server[server.index("--json-model-override-args") + 1])
        assert override["index_share_for_mtp_iteration"] is True
    else:
        assert server[server.index("--tensor-parallel-size") + 1] == "4"
        assert "--enable-expert-parallel" in server
        assert json.loads(server[server.index("--speculative-config") + 1]) == {
            "method": "mtp",
            "num_speculative_tokens": 3,
            "index_share_for_mtp_iteration": True,
        }
