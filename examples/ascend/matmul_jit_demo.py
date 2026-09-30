"""Run a pure-AIC TileLang operator through the complete Tile JIT path.

The script lowers the matrix-multiplication DSL, compiles it with OpenTileAS
and CCEC, creates a JITKernel, launches it on the NPU, and checks its result
against PyTorch. No precompiled object or explicit launch description is
required.

Run from the repository root:

    OPENTILEAS_ROOT=/path/to/OpenTileAS \
    CCEC="$(command -v ccec)" \
    TILELANG_TILE_BUILD_DIR=/tmp/tilelang-jit-build \
    python examples/ascend/matmul_jit_demo.py
"""

import os

import torch

import tilelang
import tilelang.language as T


def matmul(m=64, n=64, k=64, block_m=64):
    """Build a pure-cube matrix multiplication PrimFunc."""
    num_blocks = m // block_m
    input_dtype = "bfloat16"
    accum_dtype = "float32"

    @T.prim_func
    def main(
        A: T.Buffer((m, k), input_dtype),
        B: T.Buffer((n, k), input_dtype),
        C: T.Buffer((m, n), accum_dtype),
    ):
        with T.Kernel(num_blocks) as bx:
            a_shared = T.alloc_shared((block_m, k), input_dtype)
            b_shared = T.alloc_shared((n, k), input_dtype)
            c_fragment = T.alloc_fragment((block_m, n), accum_dtype)

            T.copy(A[bx * block_m : (bx + 1) * block_m, 0:k], a_shared)
            T.copy(B[0:n, 0:k], b_shared)
            T.gemm(
                a_shared,
                b_shared,
                c_fragment,
                transpose_B=True,
                clear_accum=True,
            )
            T.copy(
                c_fragment,
                C[bx * block_m : (bx + 1) * block_m, 0:n],
            )

    return main


@tilelang.jit(
    out_idx=[2],
    target="tile",
    execution_backend="auto",
    verbose=True,
)
def build_matmul_jit(m=64, n=64, k=64, block_m=64):
    return matmul(m=m, n=n, k=k, block_m=block_m)


def main():
    import torch_npu  # noqa: F401

    m, n, k, block_m = 64, 64, 64, 64
    device_index = int(
        os.environ.get("OPENTILE_TEST_DEVICE", "0").rsplit(":", 1)[-1]
    )
    torch.npu.set_device(device_index)
    device = torch.device("npu", device_index)

    op = build_matmul_jit(m=m, n=n, k=k, block_m=block_m)
    generator = torch.Generator(device="cpu").manual_seed(20260914)
    a_cpu = torch.randn((m, k), dtype=torch.bfloat16, generator=generator)
    b_cpu = torch.randn((n, k), dtype=torch.bfloat16, generator=generator)

    output = op(a_cpu.to(device), b_cpu.to(device))
    torch.npu.synchronize()
    print(
        f"[MATMUL_JIT] launch passed: shape={tuple(output.shape)} "
        f"dtype={output.dtype} device={output.device}",
        flush=True,
    )

    output_cpu = output.cpu()
    reference = a_cpu.float() @ b_cpu.float().transpose(0, 1)
    print(
        f"[MATMUL_JIT] reference ready: shape={tuple(reference.shape)} "
        f"dtype={reference.dtype} device={reference.device}",
        flush=True,
    )
    diff = (output_cpu.float() - reference).abs()
    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()
    max_rel_err = (diff / reference.abs().clamp_min(1e-6)).max().item()
    atol, rtol = 1e-3, 1e-3
    num_bad = int((diff > atol + rtol * reference.abs()).sum())
    total = diff.numel()
    print(
        "[MATMUL_JIT] accuracy: "
        f"max_abs_err={max_abs_err:.6g} mean_abs_err={mean_abs_err:.6g} "
        f"max_rel_err={max_rel_err:.6g} | bad={num_bad}/{total} "
        f"(atol={atol}, rtol={rtol})",
        flush=True,
    )
    torch.testing.assert_close(
        output_cpu,
        reference,
        atol=atol,
        rtol=rtol,
        msg="Matmul JIT output mismatch against golden reference",
    )
    print(
        "[TILE_JIT_DEMO] mode=AIC op=matmul backend=tile_obj PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
