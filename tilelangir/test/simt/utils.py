"""Helpers for offline SIMT compilation tests.

The production TileLang/OpenTileAS pipeline is still being defined, so the
tests intentionally exercise the explicit ``tile-opt`` -> ``tile-translate``
-> ``ccec`` piercing flow that is used by Onboard.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from os import PathLike
from pathlib import Path
from typing import Any


def _find_executable(
    explicit: str | PathLike[str] | None,
    env_name: str,
    default: str,
) -> str:
    requested = os.fspath(explicit) if explicit is not None else os.environ.get(env_name, default)
    resolved = shutil.which(requested)
    if resolved is not None:
        return resolved

    candidate = Path(requested).expanduser()
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate.resolve())

    raise FileNotFoundError(
        f"unable to find {default!r}; install it on PATH, set {env_name}, "
        f"or pass {default}=... explicitly"
    )


def _tilelang_opt_options(target: str) -> tuple[str, ...]:
    """Return the explicit TileLang-to-LLVM piercing pass sequence."""

    return (
        "--convert-tilelang-to-npu",
        "--cse",
        "--canonicalize",
        "--npu-plan-memory",
        "--tilelang-materialize-simt-parallel",
        "--tilelang-convert-simt-memref",
        f"--convert-npu-to-cce=target={target} disable-vector-core1=false",
        "--cse",
        "--canonicalize",
        "--cce-simt-scope-outline",
        "--convert-scf-to-cf",
        "--cce-to-library-call",
        "--convert-cce-to-cce-intr",
        "--convert-func-memref-to-bare-ptr",
        "--convert-math-to-llvm",
        "--convert-func-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-cf-to-llvm",
        "--finalize-memref-to-llvm",
        "--cce-reconcile-bare-ptr-descriptors",
        "--reconcile-unrealized-casts",
        "--canonicalize",
    )


def _run(command: list[str], stage: str) -> None:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        return

    details = "\n".join(part for part in (result.stdout, result.stderr) if part)
    raise RuntimeError(
        f"{stage} failed with status {result.returncode}"
        + (f":\n{details}" if details else "")
    )


def compile_tilelang_ir(
    source: str,
    output: str | PathLike[str],
    *,
    target: str = "dav-351x",
    cce_aicore_arch: str = "dav-c310-vec",
    tile_opt: str | PathLike[str] | None = None,
    tile_translate: str | PathLike[str] | None = None,
    ccec: str | PathLike[str] | None = None,
) -> Path:
    """Compile textual TileLangIR into an Ascend relocatable object."""

    tile_opt_path = _find_executable(tile_opt, "TILE_OPT", "tile-opt")
    tile_translate_path = _find_executable(
        tile_translate, "TILE_TRANSLATE", "tile-translate"
    )
    ccec_path = _find_executable(ccec, "CCEC", "ccec")
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tilelang-simt-test-") as temp_dir:
        input_path = Path(temp_dir) / "kernel.tilelang.mlir"
        opt_path = Path(temp_dir) / "kernel.opt.mlir"
        llvm_path = Path(temp_dir) / "kernel.ll"
        input_path.write_text(source, encoding="utf-8")

        _run(
            [
                tile_opt_path,
                *_tilelang_opt_options(target),
                str(input_path),
                "-o",
                str(opt_path),
            ],
            "tile-opt",
        )
        _run(
            [
                tile_translate_path,
                "--cce-to-backend",
                str(opt_path),
                "-o",
                str(llvm_path),
            ],
            "tile-translate",
        )

        ccec_command = [
            ccec_path,
            f"--cce-aicore-arch={cce_aicore_arch}",
            "--cce-aicore-only",
            "-O2",
            "-cce-bitcode-is-aicore",
            "-mllvm",
            "--cce-vf-auto-sync=global",
        ]
        if "_mlir_ciface_" in llvm_path.read_text(encoding="utf-8"):
            lib_dir = Path(tile_opt_path).resolve().parent.parent / "lib"
            for name in ("libdevice.bc", "libdevice_simt.bc"):
                bitcode = lib_dir / name
                if bitcode.is_file():
                    ccec_command.extend(
                        ("-cce-link-aicore-ll-module", str(bitcode))
                    )
        ccec_command.extend(("-c", str(llvm_path), "-o", str(output_path)))
        _run(ccec_command, "ccec")

    if not output_path.is_file():
        raise RuntimeError(f"ccec reported success but did not create {output_path}")
    return output_path


def compile_tilelang_kernel(
    kernel: Any,
    output: str | PathLike[str],
    **kwargs: Any,
) -> Path:
    """Lower a TileLang test kernel and compile it into an Ascend object."""

    import tilelang

    source = tilelang.lower(kernel, target="tile").kernel_source
    return compile_tilelang_ir(source, output, **kwargs)


def require_symbols(
    obj: str | PathLike[str],
    symbols: tuple[str, ...],
    *,
    nm: str | PathLike[str] | None = None,
) -> None:
    """Require every named symbol to be present in an object file."""

    nm_path = _find_executable(nm, "NM", "nm")
    result = subprocess.run(
        [nm_path, os.fspath(obj)], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        details = "\n".join(part for part in (result.stdout, result.stderr) if part)
        raise RuntimeError(
            f"nm failed with status {result.returncode}"
            + (f":\n{details}" if details else "")
        )

    present = {
        fields[-1]
        for line in result.stdout.splitlines()
        if (fields := line.split())
    }
    missing = [symbol for symbol in symbols if symbol not in present]
    if missing:
        raise RuntimeError(f"{obj} is missing expected symbols: {missing}")
