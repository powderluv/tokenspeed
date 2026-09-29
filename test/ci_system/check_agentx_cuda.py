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
"""Check CUDA linking and Triton execution before loading AgentX model weights."""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(source, output, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    tl.store(output + offsets, tl.load(source + offsets))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-gpus", type=int, required=True)
    args = parser.parse_args()
    if args.num_gpus < 1 or torch.cuda.device_count() != args.num_gpus:
        raise RuntimeError(
            f"Expected {args.num_gpus} GPUs, found {torch.cuda.device_count()}"
        )

    # Exercise the same -lcuda linker lookup used by SGLang's JIT extensions.
    cuda_root = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    with tempfile.TemporaryDirectory(prefix="agentx-cuda-") as directory:
        root = Path(directory)
        source = root / "driver.cc"
        executable = root / "driver"
        source.write_text("#include <cuda.h>\nint main() { return cuInit(0); }\n")
        subprocess.run(
            [
                "c++",
                str(source),
                f"-I{cuda_root / 'include'}",
                "-lcuda",
                "-o",
                str(executable),
            ],
            check=True,
        )
        subprocess.run([str(executable)], check=True)
    print("CUDA driver linking and initialization passed", flush=True)

    for device in range(args.num_gpus):
        with torch.cuda.device(device):
            source = torch.arange(128, device=f"cuda:{device}", dtype=torch.float32)
            output = torch.empty_like(source)
            copy_kernel[(1,)](source, output, BLOCK_SIZE=128)
            torch.cuda.synchronize()
            torch.testing.assert_close(output, source, rtol=0, atol=0)
            print(
                f"GPU {device}: {torch.cuda.get_device_name(device)}; "
                f"Triton {triton.__version__} compile and execution passed",
                flush=True,
            )


if __name__ == "__main__":
    main()
