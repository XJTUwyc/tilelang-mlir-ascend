"""TileLangIR ``tilelang.vcvt`` codegen tests.

Verifies that ``T.vcvt`` in the DSL lowers to a ``tilelang.vcvt`` custom
dialect op whose ``src_dtype``/``dst_dtype`` attributes preserve the frontend
TIRX dtype names (and therefore the signedness that signless memref element
types drop).  These are pure IR codegen tests: they call
``tilelang.lower(..., target="tile")`` and assert on the generated MLIR source
string.  No Ascend hardware required.
"""

from __future__ import annotations

import re

import tilelang
import tilelang.language as T
import tilelang.testing


def _vcvt_kernel(
    VL: int = 64,
    src_dtype: str = "float32",
    dst_dtype: str = "float16",
):
    """Build a minimal kernel that casts a 1-D fragment ``src -> dst``."""

    @T.prim_func
    def main(
        A: T.Tensor((1, VL), src_dtype),
        B: T.Tensor((1, VL), dst_dtype),
    ):
        with T.Kernel(1):
            a_shared = T.alloc_shared((1, VL), src_dtype)
            b_shared = T.alloc_shared((1, VL), dst_dtype)
            T.copy(A[0:1, 0:VL], a_shared)

            with T.SimdVF():
                a_frag = T.alloc_fragment((VL,), src_dtype)
                b_frag = T.alloc_fragment((VL,), dst_dtype)
                T.copy(a_shared[0, 0:VL], a_frag)
                T.vcvt(a_frag, b_frag, dst_dtype)
                T.copy(b_frag, b_shared[0, 0:VL])

            T.copy(b_shared, B[0:1, 0:VL])

    return main


def _lower_to_tile_source(kernel) -> str:
    """Lower ``kernel`` to tile target and return the generated MLIR source."""
    artifact = tilelang.lower(kernel, target="tile")
    return str(artifact.kernel_source)


def _extract_vcvt_op(source: str) -> str:
    """Return the ``tilelang.vcvt`` operation emitted for ``T.vcvt``."""
    match = re.search(r'"tilelang\.vcvt"\([^)]*\)[^{]*\{[^}]*\}[^:]*:[^)]*\)',
                      source)
    assert match is not None, (
        "Expected a `tilelang.vcvt` op in the generated MLIR, but none was "
        f"found.\nSource:\n{source}"
    )
    return match.group(0)


def _assert_dtypes(block: str, src_dtype: str, dst_dtype: str) -> None:
    """Assert the vcvt op carries the expected ``src_dtype``/``dst_dtype``."""
    assert f'src_dtype = "{src_dtype}"' in block, (
        f'Expected src_dtype = "{src_dtype}" in vcvt op.\nBlock:\n{block}'
    )
    assert f'dst_dtype = "{dst_dtype}"' in block, (
        f'Expected dst_dtype = "{dst_dtype}" in vcvt op.\nBlock:\n{block}'
    )


# ---------------------------------------------------------------------------
# float -> float
# ---------------------------------------------------------------------------

@tilelang.testing.requires_package("mlir")
def test_vcvt_f32_to_f16():
    source = _lower_to_tile_source(
        _vcvt_kernel(src_dtype="float32", dst_dtype="float16")
    )
    _assert_dtypes(_extract_vcvt_op(source), "float32", "float16")


@tilelang.testing.requires_package("mlir")
def test_vcvt_f16_to_f32():
    source = _lower_to_tile_source(
        _vcvt_kernel(src_dtype="float16", dst_dtype="float32")
    )
    _assert_dtypes(_extract_vcvt_op(source), "float16", "float32")


@tilelang.testing.requires_package("mlir")
def test_vcvt_f32_to_bf16():
    source = _lower_to_tile_source(
        _vcvt_kernel(src_dtype="float32", dst_dtype="bfloat16")
    )
    _assert_dtypes(_extract_vcvt_op(source), "float32", "bfloat16")


@tilelang.testing.requires_package("mlir")
def test_vcvt_bf16_to_f32():
    source = _lower_to_tile_source(
        _vcvt_kernel(src_dtype="bfloat16", dst_dtype="float32")
    )
    _assert_dtypes(_extract_vcvt_op(source), "bfloat16", "float32")


@tilelang.testing.requires_package("mlir")
def test_vcvt_bf16_to_f16():
    source = _lower_to_tile_source(
        _vcvt_kernel(src_dtype="bfloat16", dst_dtype="float16")
    )
    _assert_dtypes(_extract_vcvt_op(source), "bfloat16", "float16")


@tilelang.testing.requires_package("mlir")
def test_vcvt_f16_to_bf16():
    source = _lower_to_tile_source(
        _vcvt_kernel(src_dtype="float16", dst_dtype="bfloat16")
    )
    _assert_dtypes(_extract_vcvt_op(source), "float16", "bfloat16")


# ---------------------------------------------------------------------------
# float -> int : signed / unsigned
# ---------------------------------------------------------------------------

@tilelang.testing.requires_package("mlir")
def test_vcvt_f32_to_int32():
    source = _lower_to_tile_source(
        _vcvt_kernel(src_dtype="float32", dst_dtype="int32")
    )
    _assert_dtypes(_extract_vcvt_op(source), "float32", "int32")


@tilelang.testing.requires_package("mlir")
def test_vcvt_f32_to_uint8():
    source = _lower_to_tile_source(
        _vcvt_kernel(src_dtype="float32", dst_dtype="uint8")
    )
    _assert_dtypes(_extract_vcvt_op(source), "float32", "uint8")


# ---------------------------------------------------------------------------
# int -> float
# ---------------------------------------------------------------------------

@tilelang.testing.requires_package("mlir")
def test_vcvt_int32_to_f32():
    source = _lower_to_tile_source(
        _vcvt_kernel(src_dtype="int32", dst_dtype="float32")
    )
    _assert_dtypes(_extract_vcvt_op(source), "int32", "float32")


@tilelang.testing.requires_package("mlir")
def test_vcvt_int8_to_f16():
    source = _lower_to_tile_source(
        _vcvt_kernel(src_dtype="int8", dst_dtype="float16")
    )
    _assert_dtypes(_extract_vcvt_op(source), "int8", "float16")


@tilelang.testing.requires_package("mlir")
def test_vcvt_uint8_to_float32():
    source = _lower_to_tile_source(
        _vcvt_kernel(src_dtype="uint8", dst_dtype="float32")
    )
    _assert_dtypes(_extract_vcvt_op(source), "uint8", "float32")


# ---------------------------------------------------------------------------
# int -> int
# ---------------------------------------------------------------------------

@tilelang.testing.requires_package("mlir")
def test_vcvt_int32_to_int8():
    source = _lower_to_tile_source(
        _vcvt_kernel(src_dtype="int32", dst_dtype="int8")
    )
    _assert_dtypes(_extract_vcvt_op(source), "int32", "int8")


@tilelang.testing.requires_package("mlir")
def test_vcvt_int8_to_int32():
    source = _lower_to_tile_source(
        _vcvt_kernel(src_dtype="int8", dst_dtype="int32")
    )
    _assert_dtypes(_extract_vcvt_op(source), "int8", "int32")


@tilelang.testing.requires_package("mlir")
def test_vcvt_uint8_to_int32():
    source = _lower_to_tile_source(
        _vcvt_kernel(src_dtype="uint8", dst_dtype="int32")
    )
    _assert_dtypes(_extract_vcvt_op(source), "uint8", "int32")


@tilelang.testing.requires_package("mlir")
def test_vcvt_uint8_to_int8():
    """uint8 -> int8 is same-width; the op still records both dtype names."""
    source = _lower_to_tile_source(
        _vcvt_kernel(src_dtype="uint8", dst_dtype="int8")
    )
    _assert_dtypes(_extract_vcvt_op(source), "uint8", "int8")


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

@tilelang.testing.requires_package("mlir")
def test_vcvt_emits_tilelang_vcvt_op():
    """``T.vcvt`` must emit a ``tilelang.vcvt`` custom dialect op."""
    source = _lower_to_tile_source(
        _vcvt_kernel(src_dtype="float32", dst_dtype="float16")
    )
    assert "tilelang.vcvt" in source, (
        f"Expected a tilelang.vcvt op.\nSource:\n{source}"
    )
    assert "linalg.generic" not in source, (
        f"vcvt must no longer lower to linalg.generic.\nSource:\n{source}"
    )


if __name__ == "__main__":
    tilelang.testing.main()
