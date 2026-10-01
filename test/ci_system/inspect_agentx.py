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
"""Collect existing GB300 AgentX logs and results without changing Slurm jobs."""

import argparse
import re
import shutil
from pathlib import Path

from slurm_submit import (
    Submission,
    load_task,
    scontrol_states,
    slurm_states,
    write_report,
)


def collect(artifact_root: Path, commit: str, output_dir: Path) -> None:
    """Snapshot the six AgentX tasks associated with one full source commit."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("commit must be a full lowercase SHA")
    if not (artifact_root / "snapshots" / f"{commit}.tar").is_file():
        raise ValueError("source snapshot is missing")
    repo = Path(__file__).resolve().parents[2]
    pattern = re.compile(
        rf"perf-(tokenspeed-qwen3\.8-flash-next-nvfp4-agentx-p(?:1|2|4|8|16|32))"
        rf"-{commit[:12]}-[0-9]+-([0-9]+)\.out"
    )
    submissions = []
    for log in sorted((artifact_root / "logs").glob(f"*-{commit[:12]}-*.out")):
        match = pattern.fullmatch(log.name)
        if match is None:
            continue
        task = load_task(repo, f"test/ci/perf/{match[1]}.yaml", "slurm-gb300-4gpu")
        submissions.append(Submission(task, match[2], log))
    if not submissions:
        raise ValueError("no AgentX job logs found for this commit")
    job_ids = [item.job_id for item in submissions]
    states = slurm_states(job_ids)
    states.update(
        scontrol_states([job_id for job_id in job_ids if job_id not in states])
    )
    write_report(submissions, states, artifact_root / "runs", output_dir)
    for item in submissions:
        published = artifact_root / "runs" / item.job_id / "published"
        if published.is_dir():
            shutil.copytree(
                published,
                output_dir / f"{item.job_id}-published",
                ignore=shutil.ignore_patterns("tokenizer"),
                dirs_exist_ok=True,
            )
    (output_dir / "source-commit.txt").write_text(f"{commit}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    collect(args.artifact_root, args.commit, args.output_dir)


if __name__ == "__main__":
    main()
