"""TileLangIR ``tl.tileop.vreduce_max`` codegen tests."""

import tilelang
import tilelang.language as T
import tilelang.testing


def _vreduce_max_kernel(
    M: int = 1024,
    N: int = 256,
    block_M: int = 32,
    dtype: str = "float32",
):
    num_blocks = M // block_M

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M,), dtype),
    ):
        with T.Kernel(num_blocks) as bx:
            a_shared = T.alloc_shared((block_M, N), dtype)
            c_shared = T.alloc_shared((block_M,), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, 0:N], a_shared)

            VL = 64

            with T.SimdVF():
                for r in range(0, block_M):
                    a_frag = T.alloc_frag((VL,), dtype)
                    dst_frag = T.alloc_frag((1,), dtype)

                    T.copy(a_shared[r, 0:VL], a_frag)

                    T.vreduce_max(a_frag, dst_frag)

                    T.copy(dst_frag, c_shared[r])

            T.copy(c_shared, C[bx * block_M : (bx + 1) * block_M])

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
def test_vreduce_max_emits_linalg_reduce():
    artifact = tilelang.lower(_vreduce_max_kernel(), target="tile")
    mlir = _kernel_source(artifact)

    assert "func.func" in mlir
    assert "linalg.reduce" in mlir
    assert "linalg.fill" in mlir
    assert "arith.maximumf" in mlir
    # No tensor conversions should remain.
    assert "bufferization.to_tensor" not in mlir
    assert "tensor.generate" not in mlir
    assert "materialize_in_destination" not in mlir


if __name__ == "__main__":
    tilelang.testing.main()
