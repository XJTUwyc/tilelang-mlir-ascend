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


def _if_else_stmt_kernel():
    """Kernel whose body contains a Python native ``if/else`` statement.

    The TIR builder lowers a Python ``if/else`` to a statement-level
    ``tirx.IfThenElse`` with both then and else regions.
    """
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
                if P[bx * 32 + tx]:
                    C[bx * 32 + tx] = A[bx * 32 + tx]
                else:
                    C[bx * 32 + tx] = B[bx * 32 + tx]

    return main


def _if_only_stmt_kernel():
    """Kernel whose body contains a Python native ``if`` without ``else``.

    The TIR builder lowers a Python ``if`` without an else branch to a
    statement-level ``tirx.IfThenElse`` whose else region is absent.
    """
    dtype = "float32"

    @T.prim_func
    def main(
        A: T.Tensor((128,), dtype),
        P: T.Tensor((128,), "bool"),
        C: T.Tensor((128,), dtype),
    ):
        with T.Kernel(4, threads=32) as bx:
            for tx in T.serial(32):
                C[bx * 32 + tx] = A[bx * 32 + tx]
                if P[bx * 32 + tx]:
                    C[bx * 32 + tx] = A[bx * 32 + tx] * 2.0

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


@tilelang.testing.requires_package("mlir")
def test_if_then_else_stmt_emits_scf_if_with_else():
    """Python native ``if/else`` (statement-level ``tirx.IfThenElse``) must
    lower to ``scf.if`` preserving both the then and else regions."""
    source = _lower_to_tile_source(_if_else_stmt_kernel())
    if_match = re.search(r"scf\.if [^\{]*\{", source)
    assert if_match is not None, (
        f"Expected an `scf.if` op in the generated MLIR, but none was found.\n{source}"
    )
    else_match = re.search(r"\}\s*else\s*\{", source)
    assert else_match is not None, (
        f"Expected the `scf.if` op to have an else region, but none was found.\n{source}"
    )
    assert "arith.select" not in source


@tilelang.testing.requires_package("mlir")
def test_if_stmt_without_else_emits_scf_if():
    """Python native ``if`` without ``else`` must lower to an ``scf.if``
    without an else region."""
    source = _lower_to_tile_source(_if_only_stmt_kernel())
    assert re.search(r"scf\.if [^\{]*\{", source) is not None, (
        f"Expected an `scf.if` op in the generated MLIR, but none was found.\n{source}"
    )
    assert not re.search(r"\}\s*else\s*\{", source), (
        f"Did not expect an else region on `scf.if`, but one was found.\n{source}"
    )
    assert "arith.select" not in source


if __name__ == "__main__":
    tilelang.testing.main()
