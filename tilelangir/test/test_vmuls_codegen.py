"""TileLangIR ``tl.tileop.vmuls`` codegen tests.

Verifies that ``T.vmuls`` in the DSL lowers to a ``tilelang.vmuls`` custom
dialect op with the scalar captured as an explicit SSA operand (no extra memref
alloc for the scalar).  These are pure IR codegen tests: they call
``tilelang.lower(..., target="tile")`` and assert on the generated MLIR source
string.  No Ascend hardware required.
"""

import re

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing


def _vmuls_kernel(
    M: int = 1024,
    N: int = 256,
    block_M: int = 32,
    scalar_val: float = 2.0,
    dtype: str = "float32",
    dst_dtype: str = None,
):
    """Build a minimal vmuls kernel ``C = A * scalar``."""
    num_blocks = M // block_M
    dst_dtype = dst_dtype or dtype
    VL = 64

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dst_dtype),
    ):
        with T.Kernel(num_blocks) as bx:
            a_shared = T.alloc_shared((block_M, N), dtype)
            c_shared = T.alloc_shared((block_M, N), dst_dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, 0:N], a_shared)

            with T.SimdVF():
                for r in range(0, block_M):
                    for i in range(0, N // VL):
                        a_frag = T.alloc_frag((VL,), dtype)
                        c_frag = T.alloc_frag((VL,), dst_dtype)

                        T.copy(a_shared[r, i * VL : (i + 1) * VL], a_frag)

                        T.vmuls(a_frag, scalar_val, c_frag)

                        T.copy(c_frag, c_shared[r, i * VL : (i + 1) * VL])

            T.copy(c_shared, C[bx * block_M : (bx + 1) * block_M, 0:N])

    return main


def _lower_to_tile_source(kernel):
    """Lower ``kernel`` to tile target and return the generated MLIR source."""
    from tilelang import lower as _lower

    artifact = _lower(kernel, target="tile")
    return artifact.kernel_source


def _extract_vmuls_op(source: str) -> str:
    """Return the ``tilelang.vmuls`` operation emitted for ``T.vmuls``."""
    match = re.search(r'"tilelang\.vmuls"\([^)]*\)[^{]*:[^)]*\)', source)
    assert match is not None, (
        "Expected a ``tilelang.vmuls`` op in the generated MLIR, but none was "
        f"found.\nSource:\n{source}"
    )
    return match.group(0)


def test_vmuls_basic_lowers_to_tilelang_vmuls():
    """C = A * scalar must emit at least one ``tilelang.vmuls``."""
    source = _lower_to_tile_source(_vmuls_kernel())
    assert "tilelang.vmuls" in source
    assert "linalg.generic" not in source


def test_vmuls_operand_has_fragment_address_space():
    """The src/dst memrefs should live in address space 2 (fragment)."""
    source = _lower_to_tile_source(_vmuls_kernel())
    op_text = _extract_vmuls_op(source)
    assert "memref<64xf32, 2>" in op_text, (
        f"Expected fragment memref with address space 2, got:\n{op_text}"
    )


def test_vmuls_scalar_constant_is_present():
    """The scalar constant (e.g. 2.0) must appear as an ``arith.constant``."""
    source = _lower_to_tile_source(_vmuls_kernel(scalar_val=2.0))
    assert "arith.constant 2.000000e+00 : f32" in source, (
        "Expected scalar constant 2.0 in the generated MLIR.\n"
        f"Source:\n{source}"
    )


def test_vmuls_different_scalar_values():
    """Different scalar values must produce different constants in the MLIR."""
    source_2 = _lower_to_tile_source(_vmuls_kernel(scalar_val=2.0))
    assert "arith.constant 2.000000e+00 : f32" in source_2

    source_3 = _lower_to_tile_source(_vmuls_kernel(scalar_val=3.0))
    assert "arith.constant 3.000000e+00 : f32" in source_3


def test_vmuls_integer_dtype():
    """Integer vmuls must still lower to a ``tilelang.vmuls`` op."""
    source = _lower_to_tile_source(_vmuls_kernel(dtype="int32", scalar_val=2))
    assert "tilelang.vmuls" in source


def test_vmuls_mismatched_dtype_raises():
    """src/dst with different dtypes must raise TypeError."""
    with pytest.raises(TypeError):
        _lower_to_tile_source(_vmuls_kernel(dtype="float32", dst_dtype="float16"))


def _vmuls_2d_kernel(
    M: int = 128,
    N: int = 128,
    scalar_val: float = 2.0,
    dtype: str = "float32",
):
    """Build a rank-2 vmuls kernel ``C = A * scalar`` over a (M, N) fragment."""

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1), T.SimdVF():
            a_frag = T.alloc_frag((M, N), dtype)
            c_frag = T.alloc_frag((M, N), dtype)
            T.copy(A[0:M, 0:N], a_frag)
            T.vmuls(a_frag, scalar_val, c_frag)
            T.copy(c_frag, C[0:M, 0:N])

    return main


def test_vmuls_2d_lowers_to_tilelang_vmuls():
    """A rank-2 vmuls must emit a ``tilelang.vmuls`` op."""
    source = _lower_to_tile_source(_vmuls_2d_kernel())
    assert "tilelang.vmuls" in source


if __name__ == "__main__":
    tilelang.testing.main()
