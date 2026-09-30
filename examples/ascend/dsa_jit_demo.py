"""Run a MIX TileLang operator through the complete Tile JIT path.

The script lowers the DSA DSL, compiles it with OpenTileAS and CCEC, creates a
JITKernel, launches it on the NPU, and checks its result against PyTorch. No
precompiled object or explicit launch description is required. The call is:

    O = op(Q, KV, AttnSink, TopKIndices)

Run from the repository root:

    OPENTILEAS_ROOT=/path/to/OpenTileAS \
    CCEC="$(command -v ccec)" \
    TILELANG_TILE_BUILD_DIR=/tmp/tilelang-jit-build \
    python examples/ascend/dsa_jit_demo.py
"""

import math
import os

import torch

import tilelang
import tilelang.language as T


def dsa_demo(
    batch_size=1,
    seq_len=4096,
    seq_len_kv=4096,
    num_heads=64,
    dim=256,
    top_k=128,
):
    """Build a sparse-attention PrimFunc for the Ascend ``tile`` target."""

    block_heads = num_heads
    half_heads = block_heads // 2  # heads per vector core (1:2 split)
    block_top_k = 64
    dtype = "bfloat16"
    accum_dtype = "float32"
    indices_dtype = "int32"

    num_sparse_blocks = (top_k + block_top_k - 1) // block_top_k
    softmax_scale = 1.0 / math.sqrt(dim)

    @T.prim_func
    def main(
        Q: T.Buffer((batch_size, seq_len, num_heads, dim), dtype),
        KV: T.Buffer((batch_size, seq_len_kv, dim), dtype),
        AttnSink: T.Buffer((num_heads,), accum_dtype),
        TopKIndices: T.Buffer((batch_size, seq_len, top_k), indices_dtype),
        O: T.Buffer((batch_size, seq_len, num_heads, dim), accum_dtype),
    ):
        # Kernel axes map to logical cores: cid (blockIdx.x) and vid
        # (blockIdx.y, the 1:2 vec-core split).
        with T.Kernel(seq_len, 2) as (cid, vid):
            # L1 operands of the gemms stay full-shape.
            q_shared = T.alloc_shared((block_heads, dim), dtype)
            kv_shared = T.alloc_shared((block_top_k, dim), dtype)
            p_shared = T.alloc_shared((block_heads, block_top_k), dtype)
            valid_shared = T.alloc_shared((block_top_k,), accum_dtype)
            idxs_ub = T.alloc_shared((block_top_k,), indices_dtype)
            kv_ub = T.alloc_shared((block_top_k, dim), dtype)

            acc_s = T.alloc_fragment((block_heads, block_top_k), accum_dtype)
            acc_o = T.alloc_fragment((block_heads, dim), accum_dtype)

            # Dual-vec UB buffers: half-shape, private to each vector core.
            s_ub = T.alloc_shared((half_heads, block_top_k), accum_dtype)
            p_ub = T.alloc_shared((half_heads, block_top_k), dtype)
            o_ub = T.alloc_shared((half_heads, dim), accum_dtype)
            o_tmp_ub = T.alloc_shared((half_heads, dim), accum_dtype)
            m_ub = T.alloc_shared((half_heads,), accum_dtype)
            old_m_ub = T.alloc_shared((half_heads,), accum_dtype)
            row_sum_ub = T.alloc_shared((half_heads,), accum_dtype)
            l_ub = T.alloc_shared((half_heads,), accum_dtype)
            alpha_ub = T.alloc_shared((half_heads,), accum_dtype)

            T.copy(
                Q[
                    0,
                    cid,
                    0:num_heads,
                    0:dim,
                ],
                q_shared,
            )

            with T.SimdVF():
                T.fill(o_ub, 0.0)
                T.fill(l_ub, 1.0)

            T.copy(
                AttnSink[vid * half_heads : (vid + 1) * half_heads],
                m_ub,
            )

            for sparse_block in T.Pipelined(num_sparse_blocks, num_stages=2):
                for sparse_col in T.serial(block_top_k):
                    topk_col = sparse_block * block_top_k + sparse_col
                    if topk_col < top_k:
                        idxs_ub[sparse_col] = TopKIndices[
                            0,
                            cid,
                            topk_col,
                        ]
                    else:
                        idxs_ub[sparse_col] = -1

                for sparse_col in T.serial(block_top_k):
                    if idxs_ub[sparse_col] != -1:
                        valid_shared[sparse_col] = 1.0
                    else:
                        valid_shared[sparse_col] = 0.0

                # Zero kv_ub so rows left ungathered (invalid indices)
                # contribute exact zeros to the gemm.
                with T.SimdVF():
                    T.fill(kv_ub, T.cast(0, dtype))

                for sparse_col in T.serial(block_top_k):
                    if idxs_ub[sparse_col] != -1:
                        T.copy(
                            KV[0, idxs_ub[sparse_col], 0:dim],
                            kv_ub[sparse_col, 0:dim],
                        )

                # Full-tile UB->L1 (NZ) delivery of the gemm B operand.
                T.copy(kv_ub, kv_shared)

                # --- S = Q @ K^T ---
                T.gemm(
                    q_shared,
                    kv_shared,
                    acc_s,
                    transpose_B=True,
                    clear_accum=True,
                )
                # L0C -> UB, split heads across the two vec cores.
                T.copy(acc_s, s_ub, split_dim=0)

                # --- Step 1: scale scores, m_new = max(row_max(S), m_old) ---
                with T.SimdVF():
                    s_frag = T.alloc_frag((half_heads, block_top_k), accum_dtype)
                    row_max_frag = T.alloc_frag((half_heads,), accum_dtype)

                    T.copy(s_ub, s_frag)
                    T.vmuls(s_frag, softmax_scale, s_frag)
                    T.copy(s_frag, s_ub)

                    T.copy(m_ub, old_m_ub)
                    T.vreduce_max(s_frag, row_max_frag, dim=1)
                    T.copy(row_max_frag, m_ub)

                # --- Step 2: P = exp(S - m_new) * valid, row sums ---
                with T.SimdVF():
                    s_frag = T.alloc_frag((half_heads, block_top_k), accum_dtype)
                    p_frag = T.alloc_frag((half_heads, block_top_k), accum_dtype)
                    p_bf16_frag = T.alloc_frag((half_heads, block_top_k), dtype)
                    m_frag = T.alloc_frag((half_heads,), accum_dtype)
                    m_new_frag = T.alloc_frag((half_heads,), accum_dtype)
                    row_max_frag = T.alloc_frag((half_heads,), accum_dtype)
                    m_new_bc = T.alloc_frag((half_heads, block_top_k), accum_dtype)
                    valid_frag = T.alloc_frag((block_top_k,), accum_dtype)
                    valid_bc = T.alloc_frag((half_heads, block_top_k), accum_dtype)
                    row_sum_frag = T.alloc_frag((half_heads,), accum_dtype)

                    # Invalid KV rows have S = 0 exactly (zero-filled), and
                    # m_new >= 0, so exp(0 - m_new) <= 1 never overflows.
                    T.copy(s_ub, s_frag)
                    T.copy(m_ub, row_max_frag)
                    T.copy(old_m_ub, m_frag)
                    T.vmax(row_max_frag, m_frag, m_new_frag)
                    T.copy(m_new_frag, m_ub)
                    T.broadcast(m_new_frag, m_new_bc)
                    T.vexpdif(s_frag, m_new_bc, p_frag)

                    # Mask invalid columns post-exp: P *= valid (0/1).
                    T.copy(valid_shared, valid_frag)
                    T.broadcast(valid_frag, valid_bc)
                    T.vmul(p_frag, valid_bc, p_frag)

                    T.vcvt(p_frag, p_bf16_frag, dtype)
                    T.copy(p_bf16_frag, p_ub)
                    T.vreduce_sum(p_frag, row_sum_frag, dim=1)
                    T.copy(row_sum_frag, row_sum_ub)

                # --- Step 3: alpha = exp(m_old - m_new), l = l*alpha + row_sum ---
                with T.SimdVF():
                    m_frag = T.alloc_frag((half_heads,), accum_dtype)
                    m_new_frag = T.alloc_frag((half_heads,), accum_dtype)
                    l_frag = T.alloc_frag((half_heads,), accum_dtype)
                    alpha_frag = T.alloc_frag((half_heads,), accum_dtype)
                    row_sum_frag = T.alloc_frag((half_heads,), accum_dtype)

                    T.copy(old_m_ub, m_frag)
                    T.copy(m_ub, m_new_frag)
                    T.copy(l_ub, l_frag)
                    T.copy(row_sum_ub, row_sum_frag)
                    T.vexpdif(m_frag, m_new_frag, alpha_frag)
                    T.copy(alpha_frag, alpha_ub)
                    T.vmul(l_frag, alpha_frag, l_frag)
                    T.vadd(l_frag, row_sum_frag, l_frag)
                    T.copy(l_frag, l_ub)

                # --- O = O * alpha + P @ V ---
                # UB -> L1: merge the two vec-core halves back into the
                # full-head gemm A operand.
                T.copy(p_ub, p_shared, split_dim=0)
                T.gemm(
                    p_shared,
                    kv_shared,
                    acc_o,
                    transpose_B=False,
                    clear_accum=True,
                )
                # L0C -> UB, split heads across the two vec cores.
                T.copy(acc_o, o_tmp_ub, split_dim=0)

                with T.SimdVF():
                    o_frag = T.alloc_frag((half_heads, dim), accum_dtype)
                    o_tmp_frag = T.alloc_frag((half_heads, dim), accum_dtype)
                    alpha_frag = T.alloc_frag((half_heads,), accum_dtype)
                    alpha_bc = T.alloc_frag((half_heads, dim), accum_dtype)
                    T.copy(o_ub, o_frag)
                    T.copy(o_tmp_ub, o_tmp_frag)
                    T.copy(alpha_ub, alpha_frag)
                    T.broadcast(alpha_frag, alpha_bc)
                    T.vmul(o_frag, alpha_bc, o_frag)
                    T.vadd(o_frag, o_tmp_frag, o_frag)
                    T.copy(o_frag, o_ub)

            # --- O /= l, then store this core's half via the vid offset ---
            with T.SimdVF():
                o_frag = T.alloc_frag((half_heads, dim), accum_dtype)
                l_frag = T.alloc_frag((half_heads,), accum_dtype)
                l_bc = T.alloc_frag((half_heads, dim), accum_dtype)
                T.copy(o_ub, o_frag)
                T.copy(l_ub, l_frag)
                T.broadcast(l_frag, l_bc)
                T.vdiv(o_frag, l_bc, o_frag)
                T.copy(o_frag, o_ub)

            T.copy(
                o_ub,
                O[
                    0,
                    cid,
                    vid * half_heads : (vid + 1) * half_heads,
                    0:dim,
                ],
            )

    return main


@tilelang.jit(
    out_idx=[4],
    target="tile",
    execution_backend="auto",
    verbose=True,
)
def build_dsa_jit(
    batch_size=1,
    seq_len=4096,
    seq_len_kv=4096,
    num_heads=64,
    dim=256,
    top_k=128,
):
    return dsa_demo(
        batch_size=batch_size,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        num_heads=num_heads,
        dim=dim,
        top_k=top_k,
    )


def _dsa_reference(Q, KV, AttnSink, TopKIndices, dim, chunk=256):
    """按 seq 分块计算 DSA 闭式参考结果（float32）。

    与 kernel 的 online-softmax 更新等价，sink 项并入最终 max：
        m = max(AttnSink, max(S))
        O = sum(exp(S - m) * valid @ KV) / (exp(AttnSink - m)
                                            + sum(exp(S - m) * valid))
    分母中的 exp(AttnSink - m) 对应 l 初值 1.0 被逐块 alpha 累乘的闭环。
    """
    scale = 1.0 / math.sqrt(dim)
    seq_len = Q.shape[1]
    O_ref = torch.empty(
        (Q.shape[0], seq_len, Q.shape[2], dim),
        dtype=torch.float32,
        device=Q.device,
    )
    sink = AttnSink.float().view(1, -1, 1)
    num_chunks = (seq_len + chunk - 1) // chunk
    for start in range(0, seq_len, chunk):
        stop = min(start + chunk, seq_len)
        q = Q[0, start:stop].float()
        idx = TopKIndices[0, start:stop].long()
        kv = KV[0, idx].float()
        s = torch.einsum("chd,ckd->chk", q, kv) * scale
        valid = (idx >= 0).float().unsqueeze(1)
        m_final = torch.maximum(s.amax(-1, keepdim=True), sink)
        p = torch.exp(s - m_final) * valid
        denom = torch.exp(sink - m_final) + p.sum(-1, keepdim=True)
        O_ref[0, start:stop] = torch.einsum("chk,ckd->chd", p, kv) / denom
        print(
            f"[DSA_JIT] reference chunk {start // chunk + 1}/{num_chunks} done",
            flush=True,
        )
    return O_ref


def main():
    import torch_npu  # noqa: F401

    batch_size = 1
    seq_len = 4096
    seq_len_kv = 4096
    num_heads = 64
    dim = 256
    top_k = 128

    device_index = int(
        os.environ.get("OPENTILE_TEST_DEVICE", "0").rsplit(":", 1)[-1]
    )
    torch.npu.set_device(device_index)
    device = torch.device("npu", device_index)

    # 执行 DSL 构建、lowering、OpenTileAS/CCEC 编译，返回 JITKernel。
    op = build_dsa_jit(
        batch_size=batch_size,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        num_heads=num_heads,
        dim=dim,
        top_k=top_k,
    )

    generator = torch.Generator(device="cpu").manual_seed(20260907)

    Q = torch.randn(
        (batch_size, seq_len, num_heads, dim),
        dtype=torch.bfloat16,
        generator=generator,
    ).to(device)

    KV = torch.randn(
        (batch_size, seq_len_kv, dim),
        dtype=torch.bfloat16,
        generator=generator,
    ).to(device)

    AttnSink = torch.linspace(
        -1.0,
        1.0,
        num_heads,
        dtype=torch.float32,
    ).to(device)

    starts = torch.randint(
        seq_len_kv,
        (batch_size, seq_len, 1),
        generator=generator,
    )
    offsets = torch.arange(top_k).reshape(1, 1, top_k) * 31
    TopKIndices = ((starts + offsets) % seq_len_kv).to(torch.int32).to(device)

    # JITKernel 自动创建第五个参数 O，并完成 NPU 下发。
    O = op(Q, KV, AttnSink, TopKIndices)
    torch.npu.synchronize()
    print(
        "[DSA_JIT] launch passed: "
        f"shape={tuple(O.shape)} dtype={O.dtype} device={O.device}",
        flush=True,
    )

    # 计算参考结果并与 kernel 输出比对，打印误差统计。
    O_ref = _dsa_reference(Q, KV, AttnSink, TopKIndices, dim)
    print(
        f"[DSA_JIT] reference ready: shape={tuple(O_ref.shape)} "
        f"dtype={O_ref.dtype} device={O_ref.device}",
        flush=True,
    )

    diff = (O.float() - O_ref).abs()
    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()
    max_rel_err = (diff / O_ref.abs().clamp_min(1e-6)).max().item()
    atol, rtol = 1e-2, 2e-2
    num_bad = int((diff > atol + rtol * O_ref.abs()).sum())
    total = diff.numel()
    print(
        "[DSA_JIT] accuracy: "
        f"max_abs_err={max_abs_err:.6g} mean_abs_err={mean_abs_err:.6g} "
        f"max_rel_err={max_rel_err:.6g} | bad={num_bad}/{total} "
        f"(atol={atol}, rtol={rtol})",
        flush=True,
    )
    torch.testing.assert_close(
        O,
        O_ref,
        atol=atol,
        rtol=rtol,
        msg="DSA JIT output mismatch against golden reference",
    )
    print(
        "[TILE_JIT_DEMO] mode=MIX op=dsa backend=tile_obj PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()


# gen_fa_mlir.py 默认查找 flash_attention，加别名兼容
flash_attention = dsa_demo
