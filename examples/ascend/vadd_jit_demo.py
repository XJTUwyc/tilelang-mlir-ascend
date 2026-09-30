"""Run a pure-AIV TileLang operator through the complete Tile JIT path.

The script lowers the VADD DSL, compiles it with OpenTileAS and CCEC, creates
a JITKernel, launches it on the NPU, and checks its result against PyTorch.
No precompiled object or explicit launch description is required.

Run from the repository root:

    OPENTILEAS_ROOT=/path/to/OpenTileAS \
    CCEC="$(command -v ccec)" \
    TILELANG_TILE_BUILD_DIR=/tmp/tilelang-jit-build \
    python examples/ascend/vadd_jit_demo.py
"""

import os

import torch

import tilelang
import tilelang.language as T


def vadd(m=1024, n=256, block_m=32):
    """Build a pure-vector VADD PrimFunc."""
    num_blocks = m // block_m
    dtype = "float32"
    vector_length = 64

    @T.prim_func
    def main(
        A: T.Buffer((m, n), dtype),
        B: T.Buffer((m, n), dtype),
        C: T.Buffer((m, n), dtype),
    ):
        with T.Kernel(num_blocks) as bx:
            a_shared = T.alloc_shared((block_m, n), dtype)
            b_shared = T.alloc_shared((block_m, n), dtype)
            c_shared = T.alloc_shared((block_m, n), dtype)

            T.copy(A[bx * block_m : (bx + 1) * block_m, 0:n], a_shared)
            T.copy(B[bx * block_m : (bx + 1) * block_m, 0:n], b_shared)

            with T.SimdVF():
                for row in range(block_m):
                    for column in range(n // vector_length):
                        a_frag = T.alloc_frag((vector_length,), dtype)
                        b_frag = T.alloc_frag((vector_length,), dtype)
                        c_frag = T.alloc_frag((vector_length,), dtype)
                        begin = column * vector_length
                        end = (column + 1) * vector_length

                        T.copy(a_shared[row, begin:end], a_frag)
                        T.copy(b_shared[row, begin:end], b_frag)
                        T.vadd(a_frag, b_frag, c_frag)
                        T.copy(c_frag, c_shared[row, begin:end])

            T.copy(c_shared, C[bx * block_m : (bx + 1) * block_m, 0:n])

    return main


@tilelang.jit(
    out_idx=[2],
    target="tile",
    execution_backend="auto",
    verbose=True,
)
def build_vadd_jit(m=1024, n=256, block_m=32):
    return vadd(m=m, n=n, block_m=block_m)


def main():
    import torch_npu  # noqa: F401

    m, n, block_m = 1024, 256, 32
    device_index = int(
        os.environ.get("OPENTILE_TEST_DEVICE", "0").rsplit(":", 1)[-1]
    )
    torch.npu.set_device(device_index)
    device = torch.device("npu", device_index)

    op = build_vadd_jit(m=m, n=n, block_m=block_m)
    generator = torch.Generator(device="cpu").manual_seed(20260914)
    a_cpu = torch.randn((m, n), dtype=torch.float32, generator=generator)
    b_cpu = torch.randn((m, n), dtype=torch.float32, generator=generator)

    output = op(a_cpu.to(device), b_cpu.to(device))
    torch.npu.synchronize()
    print(
        f"[VADD_JIT] launch passed: shape={tuple(output.shape)} "
        f"dtype={output.dtype} device={output.device}",
        flush=True,
    )

    output_cpu = output.cpu()
    reference = a_cpu + b_cpu
    print(
        f"[VADD_JIT] reference ready: shape={tuple(reference.shape)} "
        f"dtype={reference.dtype} device={reference.device}",
        flush=True,
    )
    diff = (output_cpu.float() - reference.float()).abs()
    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()
    max_rel_err = (diff / reference.abs().clamp_min(1e-6)).max().item()
    atol, rtol = 1e-5, 1e-5
    num_bad = int((diff > atol + rtol * reference.abs()).sum())
    total = diff.numel()
    print(
        "[VADD_JIT] accuracy: "
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
        msg="VADD JIT output mismatch against golden reference",
    )
    print(
        "[TILE_JIT_DEMO] mode=AIV op=vadd backend=tile_obj PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
