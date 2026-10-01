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

import inspect_agentx


def test_snapshot_is_scoped_to_source_and_preserves_failures(monkeypatch, tmp_path):
    commit = "a" * 40
    root = tmp_path / "slurm"
    (root / "snapshots").mkdir(parents=True)
    (root / "snapshots" / f"{commit}.tar").write_bytes(b"snapshot")
    (root / "logs").mkdir()
    for parallel, job_id in ((1, "100"), (16, "101")):
        log = (
            root
            / "logs"
            / (
                f"perf-tokenspeed-qwen3.8-flash-next-nvfp4-agentx-p{parallel}"
                f"-{commit[:12]}-123-{job_id}.out"
            )
        )
        log.write_text("original failure log")
        published = root / "runs" / job_id / "published"
        published.mkdir(parents=True)
        (published / "agentx_summary.json").write_text('{"submission_valid":false}')
        (published / "tokenizer").mkdir()
        (published / "tokenizer" / "tokenizer.json").write_text("large tokenizer")
    (root / "logs" / f"unrelated-{commit[:12]}-123-102.out").write_text("unrelated")
    (
        root
        / "logs"
        / (
            "perf-tokenspeed-qwen3.8-flash-next-nvfp4-agentx-p2-bbbbbbbbbbbb-123-103.out"
        )
    ).write_text("another source")

    def states(job_ids):
        assert job_ids == ["100", "101"]
        return {
            j: {"state": "FAILED", "elapsed": "01:00:00", "exit_code": "1:0"}
            for j in job_ids
        }

    monkeypatch.setattr(inspect_agentx, "slurm_states", states)
    monkeypatch.setattr(inspect_agentx, "scontrol_states", lambda ids: {})
    output = tmp_path / "report"
    inspect_agentx.collect(root, commit, output)
    assert (output / "100.log").read_text() == "original failure log"
    assert (output / "101-published" / "agentx_summary.json").read_text() == (
        '{"submission_valid":false}'
    )
    assert not (output / "102.log").exists()
    assert not (output / "103.log").exists()
    assert not (output / "100-published" / "tokenizer").exists()
    assert (
        root / "runs" / "100" / "published" / "tokenizer" / "tokenizer.json"
    ).is_file()
