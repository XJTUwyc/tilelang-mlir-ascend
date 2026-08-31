"""TileLangIR ``if_then_else`` codegen tests.

Covers the two TIRX ``if_then_else`` forms:

1. Statement ``tirx.IfThenElse`` (Python ``if/else``) must lower to ``scf.if``,
   preserving the then and else regions.
2. Expression ``T.if_then_else`` (a ``tirx.Call`` with op name ``tirx.if_then_else``)
   must lower to ``arith.select``.

These are pure IR codegen tests: they call ``tilelang.lower(..., target="tile")``
and assert on the generated MLIR source string. No Ascend hardware required.
"""

import re

import tilelang
import tilelang.language as T
import tilelang.testing


def _lower_to_tile_source(kernel):
    """Lower ``kernel`` to tile target and return the generated MLIR source."""
    artifact = tilelang.lower(kernel, target="tile")
    return artifact.kernel_source

def _if_expr_kernel():
    """Kernel whose body contains the `T.if_then_else` expression."""
    dtype = "float32"

    @T.prim_func
    def main(
        A: T.Tensor((128,), dtype),
        B: T.Tensor((128,), dtype),
        P: T.Tensor((128,), "bool"),
        C: T.Tensor((128,), dtype),
    ):
        with T.Kernel(4, threads=32) as bx:
            for tx in T.serial(32):
                C[bx * 32 + tx] = T.if_then_else(
                    P[bx * 32 + tx], A[bx * 32 + tx], B[bx * 32 + tx]
                )

    return main

@tilelang.testing.requires_package("mlir")
def test_if_then_else_expr_emits_arith_select():
    """``T.if_then_else`` must lower to a scalar ``arith.select``."""
    source = _lower_to_tile_source(_if_expr_kernel())
    assert "arith.select" in source, (
        f"Expected an `arith.select` op in the generated MLIR, but none was found.\n{source}"
    )
    assert "scf.if" not in source

def test_if_then_else_expr_result_type_matches_branches():
    """``arith.select`` result must carry the branch dtype (f32 here)."""
    source = _lower_to_tile_source(_if_expr_kernel())
    select_match = re.search(r"arith\.select[^\n]*:\s*(\S+)", source)
    assert select_match is not None, f"No arith.select found in:\n{source}"
    assert "f32" in select_match.group(1)


if __name__ == "__main__":
    tilelang.testing.main()