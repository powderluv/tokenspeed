# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Run one standard AgentX concurrency point against SGLang or vLLM."""

import argparse
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen

CONTEXT_LENGTH = 262144
DURATION_SECONDS = 1800
SEED = 20260707


def validate_models(payload: dict, model: str) -> None:
    """Require the requested model and the standard workload's context window."""
    models = [item for item in payload["data"] if item["id"] == model]
    if len(models) != 1:
        raise ValueError(f"{model} must be advertised exactly once by /v1/models")
    context_length = models[0].get("max_model_len")
    if type(context_length) is not int or context_length < CONTEXT_LENGTH:
        raise ValueError(f"{model} must advertise at least {CONTEXT_LENGTH} tokens")


def validate_result(output_dir: Path, parallel: int) -> dict:
    """Reject incomplete or incomparable results, retaining diagnostic files."""
    paths = list(output_dir.rglob("agentx_summary.json"))
    if len(paths) != 1:
        raise ValueError(f"expected one AgentX summary, found {len(paths)}")
    result = json.loads(paths[0].read_text())
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "completed" or result["submission_valid"] is not True:
        raise ValueError("AgentX did not produce a completed, valid benchmark")
    if (
        result["concurrency"] != parallel
        or result["duration"] != DURATION_SECONDS
        or result["seed"] != SEED
        or result["scenario"]["mode"] != "benchmark"
        or result["scenario"]["variant"] != "256k"
    ):
        raise ValueError("AgentX result does not match the standard benchmark")
    if result["dataset"]["verified"] is not True:
        raise ValueError("AgentX dataset hash was not verified")
    if result["dataset"]["source"] != "modelscope":
        raise ValueError("AgentX must use the standard ModelScope mirror")
    if not result["metrics"] or result["error_summary"]:
        raise ValueError("AgentX reported missing metrics or request errors")
    return result


def build_command(args: argparse.Namespace, tokenizer: Path) -> list[str]:
    """Build the standard run, preserving the workload and timing defaults."""
    scenario = {
        "name": "agentx",
        "mode": "benchmark",
        "variant": "256k",
        "num_gpus": args.num_gpus,
        "engine": args.engine,
        "engine_version": args.engine_version,
    }
    return [
        args.evalscope,
        "perf",
        "--scenario",
        json.dumps(scenario, separators=(",", ":")),
        "--model",
        args.model,
        "--tokenizer-path",
        str(tokenizer),
        "--url",
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        "--api",
        "openai",
        "--data-source",
        "modelscope",
        "--parallel",
        str(args.parallel),
        "--duration",
        str(DURATION_SECONDS),
        "--seed",
        str(SEED),
        "--outputs-dir",
        str(args.outputs_dir.resolve()),
        "--no-timestamp",
    ]


def run(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download

    if args.num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    output_dir = args.outputs_dir.resolve()
    # A failed invocation must not reuse a previous run's successful summary.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"benchmark output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with urlopen(f"{args.base_url.rstrip('/')}/v1/models", timeout=30) as response:
        models = json.load(response)
    (output_dir / "server-models.json").write_text(f"{json.dumps(models, indent=2)}\n")
    validate_models(models, args.model)

    # Reuse the shared HF cache; avoid downloading model weights or remote code.
    tokenizer = Path(
        snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            allow_patterns=[
                "config.json",
                "tokenizer*",
                "special_tokens_map.json",
                "vocab.*",
                "merges.txt",
                "chat_template*",
            ],
        )
    )
    command = build_command(args, tokenizer)
    (output_dir / "command.json").write_text(f"{json.dumps(command, indent=2)}\n")
    completed = subprocess.run(command, check=False)
    validate_result(output_dir, args.parallel)
    completed.check_returncode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parallel", type=int, choices=(1, 2, 4, 8, 16, 32), required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--engine", choices=("sglang", "vllm"), required=True)
    parser.add_argument("--engine-version", required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--evalscope", required=True)
    parser.add_argument("--outputs-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
