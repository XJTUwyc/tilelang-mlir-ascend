"""TileLangIR ``tilelang.vexpdif`` codegen tests.

Verifies that ``T.vexpdif`` in the DSL lowers to a ``tilelang.vexpdif`` custom
dialect op (exp(src0 - src1)), rather than an inline ``linalg.generic`` body.
These are pure IR codegen tests: they call ``tilelang.lower(...,
target="tile")`` and assert on the generated MLIR source string.  No Ascend
hardware required.
"""

import re

import tilelang
import tilelang.language as T
import tilelang.testing


def _vexpdif_kernel(
    VL: int = 64,
    src_dtype: str = "float32",
    dst_dtype: str = "float32",
):
    """Build a minimal kernel that applies ``dst = exp(src0 - src1)``."""

    @T.prim_func
    def main(
        A: T.Tensor((1, VL), src_dtype),
        B: T.Tensor((1, VL), src_dtype),
        C: T.Tensor((1, VL), dst_dtype),
    ):
        with T.Kernel(1):
            a_shared = T.alloc_shared((1, VL), src_dtype)
            b_shared = T.alloc_shared((1, VL), src_dtype)
            c_shared = T.alloc_shared((1, VL), dst_dtype)
            T.copy(A[0:1, 0:VL], a_shared)
            T.copy(B[0:1, 0:VL], b_shared)

            with T.SimdVF():
                a_frag = T.alloc_fragment((VL,), src_dtype)
                b_frag = T.alloc_fragment((VL,), src_dtype)
                c_frag = T.alloc_fragment((VL,), dst_dtype)
                T.copy(a_shared[0, 0:VL], a_frag)
                T.copy(b_shared[0, 0:VL], b_frag)
                T.vexpdif(a_frag, b_frag, c_frag)
                T.copy(c_frag, c_shared[0, 0:VL])

            T.copy(c_shared, C[0:1, 0:VL])

    return main


def _vexpdif_nd_kernel(shape, src_dtype="float32", dst_dtype="float32"):
    """Build a kernel that applies vexpdif on N-dimensional fragments."""

    @T.prim_func
    def main(
        A: T.Tensor(shape, src_dtype),
        B: T.Tensor(shape, src_dtype),
        C: T.Tensor(shape, dst_dtype),
    ):
        with T.Kernel(1), T.SimdVF():
            af = T.alloc_fragment(shape, src_dtype)
            bf = T.alloc_fragment(shape, src_dtype)
            cf = T.alloc_fragment(shape, dst_dtype)
            T.vexpdif(af, bf, cf)

    return main


def _lower_to_tile_source(kernel) -> str:
    """Lower ``kernel`` to tile target and return the generated MLIR source."""
    artifact = tilelang.lower(kernel, target="tile")
    return str(artifact.kernel_source)


def _extract_vexpdif_op(source: str) -> str:
    """Return the ``tilelang.vexpdif`` operation emitted for ``T.vexpdif``."""
    match = re.search(
        r'"tilelang\.vexpdif"\([^)]*\)[^{]*:[^)]*\)', source
    )
    assert match is not None, (
        "Expected a `tilelang.vexpdif` op in the generated MLIR, but none was "
        f"found.\nSource:\n{source}"
    )
    return match.group(0)


def test_vexpdif_emits_tilelang_vexpdif():
    """``T.vexpdif`` must lower to a ``tilelang.vexpdif`` custom op."""
    source = _lower_to_tile_source(_vexpdif_kernel())
    assert "tilelang.vexpdif" in source
    assert "linalg.generic" not in source


def test_vexpdif_has_three_memref_operands():
    """vexpdif takes src0, src1, dst fragment memrefs (address space 2)."""
    source = _lower_to_tile_source(_vexpdif_kernel())
    op_text = _extract_vexpdif_op(source)
    assert op_text.count("memref<64xf32, 2>") >= 3, (
        f"Expected three fragment memref operands, got:\n{op_text}"
    )


def test_vexpdif_float16():
    """vexpdif on float16 fragments must lower to tilelang.vexpdif."""
    source = _lower_to_tile_source(
        _vexpdif_kernel(src_dtype="float16", dst_dtype="float16")
    )
    assert "tilelang.vexpdif" in source
    assert "f16" in source


def test_vexpdif_bfloat16():
    """bf16 operands must lower to tilelang.vexpdif."""
    source = _lower_to_tile_source(
        _vexpdif_nd_kernel((64,), src_dtype="bfloat16", dst_dtype="bfloat16")
    )
    assert "tilelang.vexpdif" in source
    assert "bf16" in source


def test_vexpdif_3d_operands():
    """3-D operands must lower to tilelang.vexpdif."""
    source = _lower_to_tile_source(
        _vexpdif_nd_kernel((2, 4, 8), src_dtype="float32", dst_dtype="float32")
    )
    assert "tilelang.vexpdif" in source


if __name__ == "__main__":
    tilelang.testing.main()
