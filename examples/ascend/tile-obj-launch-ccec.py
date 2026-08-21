"""Compile a vadd NPU-IR kernel to ``.o`` and test Tile load/launch.

The MLIR-facing compiler driver is ``bishengir-compile``; it lowers NPU-IR
through the HIVM compilation pipeline and emits the object consumed by the Tile
runtime.  After sourcing the CANN environment, run the complete flow with:

    python examples/ascend/tile_obj_launch_demo.py --compile-and-run

Compile without launching, or launch an existing object, with:

    python examples/ascend/tile_obj_launch_demo.py --compile-only
    python examples/ascend/tile_obj_launch_demo.py build/tile_obj_demo/vadd_kernel.o

The Tile target currently emits ``tilelang.*`` MLIR and does not yet contain
the TileLangIR-to-NPU-IR lowering pass.  Until that pass lands, this demo
compiles the matching checked-in ``tilelangir/test/vadd_npuir.mlir`` fixture.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_NPUIR = _REPO_ROOT / "tilelangir/test/vadd_npuir.mlir"
_DEFAULT_OBJECT = _REPO_ROOT / "build/tile_obj_demo/vadd_kernel.o"


def vadd(M=1024, N=256, block_M=32):
    import tilelang.language as T

    num_blocks = M // block_M
    dtype = "float32"

    @T.prim_func
    def vadd_kernel(
        A: T.Buffer((M, N), dtype),
        B: T.Buffer((M, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(num_blocks) as bx:
            a_shared = T.alloc_shared((block_M, N), dtype)
            b_shared = T.alloc_shared((block_M, N), dtype)
            c_shared = T.alloc_shared((block_M, N), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, 0:N], a_shared)
            T.copy(B[bx * block_M : (bx + 1) * block_M, 0:N], b_shared)

            VL = 64

            with T.SimdVF():
                for r in range(0, block_M):
                    for i in range(0, N // VL):
                        a_frag = T.alloc_frag((VL,), dtype)
                        b_frag = T.alloc_frag((VL,), dtype)
                        c_frag = T.alloc_frag((VL,), dtype)

                        T.copy(a_shared[r, i * VL : (i + 1) * VL], a_frag)
                        T.copy(b_shared[r, i * VL : (i + 1) * VL], b_frag)

                        T.vadd(a_frag, b_frag, c_frag)

                        T.copy(c_frag, c_shared[r, i * VL : (i + 1) * VL])

            T.copy(c_shared, C[bx * block_M : (bx + 1) * block_M, 0:N])

    return vadd_kernel


def emit_tile_mlir() -> None:
    import tilelang

    artifact = tilelang.lower(vadd(), target="tile")
    print(artifact.kernel_source)


def _executable(candidate: str | Path) -> str | None:
    candidate = os.path.expandvars(os.path.expanduser(str(candidate)))
    resolved = shutil.which(candidate)
    if resolved is not None:
        return resolved
    path = Path(candidate)
    if path.is_file() and os.access(path, os.X_OK):
        return str(path.resolve())
    return None


def find_bishengir_compile(explicit: str | Path | None = None) -> str:
    """Find the Ascend MLIR-to-object compiler driver."""
    if explicit is not None:
        resolved = _executable(explicit)
        if resolved is None:
            raise FileNotFoundError(f"bishengir-compile is not executable: {explicit}")
        return resolved

    candidates: list[str | Path] = []
    if os.environ.get("BISHENGIR_COMPILE"):
        candidates.append(os.environ["BISHENGIR_COMPILE"])
    candidates.append("bishengir-compile")

    for env_name in ("BISHENGIR_ROOT_PATH", "BISHENGIR_HOME"):
        if os.environ.get(env_name):
            candidates.append(Path(os.environ[env_name]) / "bin/bishengir-compile")

    if os.environ.get("ASCEND_HOME_PATH"):
        ascend_home = Path(os.environ["ASCEND_HOME_PATH"])
        candidates.append(ascend_home / "bin/bishengir-compile")
        candidates.append(ascend_home / "bisheng_toolkit/bishengir/bin/bishengir-compile")

    candidates.extend(
        [
            _REPO_ROOT / "3rdparty/AscendNPU-IR/build/install/bin/bishengir-compile",
            _REPO_ROOT / "3rdparty/AscendNPU-IR-Dev/build/install/bin/bishengir-compile",
        ]
    )

    for candidate in candidates:
        resolved = _executable(candidate)
        if resolved is not None:
            return resolved

    raise FileNotFoundError(
        "could not find bishengir-compile; source the Ascend environment, set "
        "BISHENGIR_COMPILE, or pass --compiler /path/to/bishengir-compile"
    )


def compile_npuir(
    npuir_path: str | Path,
    obj_path: str | Path,
    compiler: str | Path | None = None,
) -> Path:
    """Compile NPU-IR MLIR into the device object loaded by the Tile runtime."""
    npuir_path = Path(npuir_path).resolve()
    obj_path = Path(obj_path).resolve()
    if not npuir_path.is_file():
        raise FileNotFoundError(f"NPU-IR input does not exist: {npuir_path}")

    compiler_path = find_bishengir_compile(compiler)
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        compiler_path,
        str(npuir_path),
        "-enable-hivm-compile=true",
        "-o",
        str(obj_path),
    ]
    print(f"compile: {shlex.join(command)}")
    subprocess.run(command, check=True, cwd=obj_path.parent)

    if not obj_path.is_file() or obj_path.stat().st_size == 0:
        raise RuntimeError(f"compiler did not produce a non-empty object: {obj_path}")
    print(f"object: {obj_path} ({obj_path.stat().st_size} bytes)")
    return obj_path


def launch(obj_path: str | Path) -> None:
    import torch
    import torch_npu  # noqa: F401

    import tilelang
    from tilelang.opentile import compile_tile_obj

    M, N = 1024, 256
    artifact = tilelang.lower(vadd(M=M, N=N), target="tile")
    kernel = compile_tile_obj(artifact, obj_path)

    a = torch.randn(M, N, device="npu", dtype=torch.float32)
    b = torch.randn(M, N, device="npu", dtype=torch.float32)
    c = torch.empty(M, N, device="npu", dtype=torch.float32)
    kernel(a, b, c)
    torch.npu.synchronize()

    torch.testing.assert_close(c, a + b, rtol=1e-5, atol=1e-5)
    print("vadd tile .o load/launch: PASS")


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("obj", nargs="?", type=Path, help="launch an existing object")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--emit-mlir", action="store_true", help="print TileLangIR MLIR")
    mode.add_argument("--compile-only", action="store_true", help="compile NPU-IR to .o")
    mode.add_argument(
        "--compile-and-run",
        action="store_true",
        help="compile NPU-IR, then load, launch, and check the result",
    )
    parser.add_argument("--npuir", type=Path, default=_DEFAULT_NPUIR, help="NPU-IR MLIR input")
    parser.add_argument("-o", "--output", type=Path, default=_DEFAULT_OBJECT, help="output object")
    parser.add_argument("--compiler", type=Path, help="path to bishengir-compile")
    return parser, parser.parse_args()


def main() -> None:
    parser, args = parse_args()
    if args.obj is not None and (args.emit_mlir or args.compile_only or args.compile_and_run):
        parser.error("the positional object cannot be combined with a mode option")

    if args.emit_mlir:
        emit_tile_mlir()
    elif args.compile_only:
        compile_npuir(args.npuir, args.output, args.compiler)
    elif args.compile_and_run:
        launch(compile_npuir(args.npuir, args.output, args.compiler))
    elif args.obj is not None:
        launch(args.obj)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
