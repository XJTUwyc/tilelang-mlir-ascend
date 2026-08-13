import tilelang
import tilelang.language as T

def vreduce_sum(M=1024, N=256, block_M=32):
    num_blocks = M // block_M
    dtype = "float32"

    @T.prim_func
    def main(
        A: T.Buffer((M, N), dtype),
        C: T.Buffer((M,), dtype),
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

                    T.vreduce_sum(a_frag, dst_frag)

                    T.copy(dst_frag, c_shared[r])

            T.copy(c_shared, C[bx * block_M : (bx + 1) * block_M])

    return main

if __name__ == "__main__":
    program = vreduce_sum()
    result = tilelang.lower(program, target="tile")
    mlir_source = result.kernel_source
    print(mlir_source)
    assert mlir_source is not None and len(mlir_source) > 0, "Lowered MLIR output is empty"
    assert "func.func" in mlir_source, "Missing func.func in lowered output"
    assert "linalg.reduce" in mlir_source, "Missing linalg.reduce in lowered output"
    print("All checks passed!")
