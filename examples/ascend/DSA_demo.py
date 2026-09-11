import math

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
    """Build a sparse-attention PrimFunc for the Ascend ``tile`` target.

    The p_ub -> p_shared L1 copy is issued between the P-producing
    scope and the row-sum reduce so it overlaps them on the MTE3 pipe.
    """

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
            # fp32 P tile staged for the reduce scope below.
            p_f32_ub = T.alloc_shared((half_heads, block_top_k), accum_dtype)
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

                # --- Step 2a: P = exp(S - m_new) * valid ---
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
                    T.copy(p_frag, p_f32_ub)

                # Early UB -> L1 delivery: overlaps the reduce below.
                T.copy(p_ub, p_shared, split_dim=0)

                # --- Step 2b: row sums ---
                with T.SimdVF():
                    p_frag = T.alloc_frag((half_heads, block_top_k), accum_dtype)
                    row_sum_frag = T.alloc_frag((half_heads,), accum_dtype)

                    T.copy(p_f32_ub, p_frag)
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
                # p_shared was already delivered after Step 2a.
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


if __name__ == "__main__":
    program = dsa_demo()
    tilelang.lower(program, target="tile")

# gen_fa_mlir.py 默认查找 flash_attention，加别名兼容
flash_attention = dsa_demo
