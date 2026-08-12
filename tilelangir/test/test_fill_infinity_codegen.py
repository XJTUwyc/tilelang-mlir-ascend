"""TileLangIR ``tl.tileop.fill`` + ``tl.infinity`` codegen tests."""

from __future__ import annotations

import tilelang
import tilelang.language as T
import tilelang.testing


def _fill_infinity_kernel(M=32, N=32, dtype="float32"):
    """T.fill + T.infinity triggers both _emit_fill and _emit_infinity."""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype)):
        with T.Kernel(1):
            a_shared = T.alloc_shared((M, N), dtype)
            T.fill(a_shared, T.infinity(dtype))
            T.copy(a_shared, A[0:M, 0:N])

    return main


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


@tilelang.testing.requires_package("mlir")
def test_fill_infinity_emits_linalg_fill_and_arith_constant():
    """f32 happy path: _emit_fill -> linalg.fill, _emit_infinity -> arith.constant(+inf)."""
    artifact = tilelang.lower(_fill_infinity_kernel(), target="tile")
    mlir = _kernel_source(artifact)

    assert "linalg.fill" in mlir, "_emit_fill did not emit linalg.fill"
    assert "arith.constant" in mlir, "_emit_infinity did not emit arith.constant"
    assert "0x7F800000" in mlir or "inf" in mlir, "FloatAttr is not +inf"
    assert "0xFF800000" not in mlir, "FloatAttr is -inf, expected +inf"


@tilelang.testing.requires_package("mlir")
def test_fill_infinity_dtype_float16():
    """f16 path: _dtype_type('float16') should map to F16Type, +inf = 0x7C00."""
    artifact = tilelang.lower(_fill_infinity_kernel(dtype="float16"), target="tile")
    mlir = _kernel_source(artifact)

    assert "arith.constant" in mlir
    assert "f16" in mlir, "_dtype_type did not return F16Type for 'float16'"
    # f16 +inf: 0x7C00
    assert "0x7C00" in mlir or "inf" in mlir, "FloatAttr is not +inf for f16"
    assert "0xFC00" not in mlir, "FloatAttr is -inf for f16, expected +inf"
    assert "linalg.fill" in mlir


if __name__ == "__main__":
    tilelang.testing.main()
