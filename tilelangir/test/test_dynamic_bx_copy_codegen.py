"""Dynamic blockIdx.x-indexed ``tilelang.copy`` codegen tests."""

from __future__ import annotations

import tilelang
import tilelang.language as T
import tilelang.testing

def _kernel_source(artifact) -> str:
    source = getattr(artifact, "kernel_source", None)
    if source:
        return str(source)
    device_mod = getattr(artifact, "device_mod", None)
    if device_mod is not None and hasattr(device_mod, "inspect_source"):
        inspected = device_mod.inspect_source()
        if inspected:
            return str(inspected)
    raise AssertionError("lower(target='tile') did not produce inspectable MLIR source")


def _dynamic_bx_copy_kernel(M=128, N=256, BM=32, dtype="float32"):
    """GM↔shared copy whose region indices depend on ``blockIdx.x`` only (no For).

    ``M // BM`` (4) deliberately differs from the parameter count (2) so that the
    ``BlockIdx`` assertion below distinguishes the argument position from the
    launch grid extent.
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(M // BM) as bx:
            a_shared = T.alloc_shared((BM, N), dtype)
            T.copy(A[bx * BM : (bx + 1) * BM, 0:N], a_shared)
            T.copy(a_shared, B[bx * BM : (bx + 1) * BM, 0:N])

    return main


@tilelang.testing.requires_package("mlir")
def test_dynamic_bx_copy_emits_index_offset_and_operands():
    artifact = tilelang.lower(_dynamic_bx_copy_kernel(), target="tile")
    mlir = _kernel_source(artifact)

    assert "tilelang.copy" in mlir
    assert mlir.count("tilelang.copy") >= 2

    # Trailing blockIdx.x arg (golden: ``%arg0 : i32``).
    assert ": i32" in mlir
    # BlockIdx is the position of the bx argument (after A and B), not the grid.
    assert "BlockIdx = 2 : i64" in mlir
    assert "arith.index_cast" in mlir
    assert "arith.muli" in mlir
    assert "arith.addi" in mlir

    # Dynamic-offset reinterpret_cast for the GM tile view.
    assert "memref.reinterpret_cast" in mlir
    assert "offset: ?" in mlir or "offset: [%" in mlir

    assert any(
        "tilelang.copy" in line and "%" in line for line in mlir.splitlines()
    ), "expected tilelang.copy with memref SSA operands"


if __name__ == "__main__":
    tilelang.testing.main()
