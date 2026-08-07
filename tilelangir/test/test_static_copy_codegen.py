"""TileLangIR static ``tilelang.copy`` operand codegen tests."""

from __future__ import annotations

import tilelang
import tilelang.language as T
import tilelang.testing


def _static_copy_kernel(M=32, N=256, VL=64, dtype="float32"):
    """Kernel whose region indices and extents are all compile-time constants."""

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1):
            a_shared = T.alloc_shared((M, N), dtype)
            b_shared = T.alloc_shared((M, N), dtype)
            c_shared = T.alloc_shared((M, N), dtype)

            T.copy(A[0:M, 0:N], a_shared)
            T.copy(B[0:M, 0:N], b_shared)

            with T.SimdVF():
                a_frag = T.alloc_fragment((VL,), dtype)
                b_frag = T.alloc_fragment((VL,), dtype)
                c_frag = T.alloc_fragment((VL,), dtype)

                T.copy(a_shared[0, 0:VL], a_frag)
                T.copy(b_shared[0, 0:VL], b_frag)
                T.vadd(a_frag, b_frag, c_frag)
                T.copy(c_frag, c_shared[0, 0:VL])

            T.copy(c_shared, C[0:M, 0:N])

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
def test_static_copy_emits_tilelang_copy_operands():
    artifact = tilelang.lower(_static_copy_kernel(), target="tile")
    mlir = _kernel_source(artifact)

    assert "memref.alloc" in mlir
    assert "linalg.add" in mlir
    assert "tilelang.copy" in mlir
    assert mlir.count("tilelang.copy") >= 1

    # Operand form may pretty-print as ``tilelang.copy %a, %b`` or
    # ``"tilelang.copy"(%a, %b)``.
    assert any(
        "tilelang.copy" in line and "%" in line for line in mlir.splitlines()
    ), "expected tilelang.copy with memref SSA operands"


if __name__ == "__main__":
    tilelang.testing.main()
