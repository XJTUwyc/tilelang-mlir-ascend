"""Lowering and offline compilation tests for the validated SIMT gathers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing

from tilelangir.test.simt import utils


SOURCE_SIZE = 4096
WORK_ITEMS = 128
SIMD_VECTOR_SIZE = 64


def _simt_gather():
    @T.prim_func
    def main(
        src: T.Tensor((SOURCE_SIZE,), T.float32),
        indices: T.Tensor((WORK_ITEMS,), T.int32),
        out: T.Tensor((WORK_ITEMS,), T.float32),
    ):
        with T.Kernel(1, threads=1):
            with T.SimtVF(threads=32):
                for worker in T.Parallel(WORK_ITEMS):
                    out[worker] = src[indices[worker]]

    return main


def _mixed_simt_gather():
    @T.prim_func
    def main(
        src: T.Tensor((SOURCE_SIZE,), T.float32),
        indices: T.Tensor((WORK_ITEMS,), T.int32),
        out: T.Tensor((WORK_ITEMS,), T.float32),
    ):
        with T.Kernel(1, threads=1):
            zero_ub = T.alloc_shared((SIMD_VECTOR_SIZE,), "float32")

            with T.SimdVF():
                for lane in T.Parallel(SIMD_VECTOR_SIZE):
                    zero_ub[lane] = 0.0

            with T.SimtVF(threads=1024):
                for worker in T.Parallel(WORK_ITEMS):
                    out[worker] = src[indices[worker]]

    return main


GATHER_CASES = (
    pytest.param(_simt_gather, 1, 32, id="simt"),
    pytest.param(_mixed_simt_gather, 2, 1024, id="mixed-simd-simt"),
)


@pytest.mark.parametrize(("kernel_factory", "scope_count", "threads"), GATHER_CASES)
@tilelang.testing.requires_package("mlir")
def test_simt_gather_codegen(kernel_factory, scope_count, threads):
    source = tilelang.lower(kernel_factory(), target="tile").kernel_source

    assert source.count('"tilelang.scope"') == scope_count
    assert "mode = #tilelang.scope_mode<simt>" in source
    assert f"threads = {threads} : ui32" in source
    assert "arith.constant 128" in source
    assert source.count("memref.load") == 2
    assert "memref.store" in source
    assert source.count("scf.forall") == 1
    assert 'tilelang.logical_thread_axes = ["x"]' in source
    if scope_count == 2:
        assert "mode = #tilelang.scope_mode<simd>" in source


def test_compile_helper_uses_explicit_opt_translate_ccec_flow(monkeypatch, tmp_path):
    tool_root = tmp_path / "opentileas"
    bin_dir = tool_root / "bin"
    lib_dir = tool_root / "lib"
    bin_dir.mkdir(parents=True)
    lib_dir.mkdir()

    tile_opt = bin_dir / "tile-opt"
    tile_translate = bin_dir / "tile-translate"
    ccec = bin_dir / "ccec"
    for executable in (tile_opt, tile_translate, ccec):
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)
    libdevice = lib_dir / "libdevice.bc"
    libdevice.write_bytes(b"bitcode")

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index("-o") + 1])
        if command[0] == str(tile_translate.resolve()):
            output.write_text("define void @_mlir_ciface_main() {}", encoding="utf-8")
        elif command[0] == str(ccec.resolve()):
            output.write_bytes(b"\x7fELF")
        else:
            output.write_text("module {}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(utils.subprocess, "run", fake_run)
    output = tmp_path / "kernel.o"
    result = utils.compile_tilelang_ir(
        "module {}",
        output,
        tile_opt=tile_opt,
        tile_translate=tile_translate,
        ccec=ccec,
    )

    assert result == output.resolve()
    assert len(commands) == 3
    opt_command, translate_command, ccec_command = commands
    assert opt_command[0] == str(tile_opt.resolve())
    assert "--convert-tilelang-to-npu" in opt_command
    assert "--npu-plan-memory" in opt_command
    assert "--tilelang-materialize-simt-parallel" in opt_command
    assert "--tilelang-convert-simt-memref" in opt_command
    convert_npu_to_cce = (
        "--convert-npu-to-cce=target=dav-351x disable-vector-core1=false"
    )
    assert convert_npu_to_cce in opt_command
    assert "--cce-simt-scope-outline" in opt_command
    assert opt_command.index("--npu-plan-memory") < opt_command.index(
        "--tilelang-materialize-simt-parallel"
    ) < opt_command.index("--tilelang-convert-simt-memref") < opt_command.index(
        convert_npu_to_cce
    )
    assert opt_command.index("--finalize-memref-to-llvm") < opt_command.index(
        "--cce-reconcile-bare-ptr-descriptors"
    ) < opt_command.index("--reconcile-unrealized-casts")

    assert translate_command[0] == str(tile_translate.resolve())
    assert translate_command[1] == "--cce-to-backend"
    assert ccec_command[0] == str(ccec.resolve())
    assert "--cce-aicore-arch=dav-c310-vec" in ccec_command
    assert "-cce-enable-mix" not in ccec_command
    assert "-cce-link-aicore-ll-module" in ccec_command
    assert str(libdevice) in ccec_command
    assert all("opentileas" not in Path(command[0]).name for command in commands)


@pytest.mark.parametrize(
    ("kernel_factory", "expected_symbols"),
    (
        pytest.param(
            _simt_gather,
            ("main_vf_simt_simt_entry",),
            id="simt",
        ),
        pytest.param(
            _mixed_simt_gather,
            ("main_vf_simt_simt_entry", "main.vector.thread"),
            id="mixed-simd-simt",
        ),
    ),
)
def test_compile_simt_gather_to_ascend_object(
    kernel_factory,
    expected_symbols,
    tmp_path,
):
    pytest.importorskip("mlir")
    try:
        output = utils.compile_tilelang_kernel(
            kernel_factory(), tmp_path / "kernel.o"
        )
    except FileNotFoundError as error:
        pytest.skip(str(error))

    assert output.read_bytes().startswith(b"\x7fELF")
    utils.require_symbols(output, expected_symbols)


if __name__ == "__main__":
    tilelang.testing.main()
