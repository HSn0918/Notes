#!/usr/bin/env python3
"""Stdlib-only CPU lab for LLM Linear/Advanced Algebra & SIMD vectorization intuitions:
1. Matrix multiplication 4 views & 2D Block GEMM Tiling (+ tiled Online Softmax Attention)
2. 2D/Multi-pair Orthogonal Rotation Matrices & RoPE relative-position dot-product invariance
3. Matrix associativity & DeepSeek MLA Weight Absorption numerical equivalence + HBM byte audit
4. SVD rank-1 outer-product expansion, Eckart-Young low-rank truncation & LoRA parameter ratio
5. 8-lane SIMD FMA dot product + Butterfly Tree Reduction & Tensor Stride / zero-copy transpose
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

SEED = 20261117
TOL = 1e-12


def max_abs_diff_1d(a: Sequence[float], b: Sequence[float]) -> float:
    assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def max_abs_diff_2d(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    assert len(a) == len(b), f"Row mismatch: {len(a)} vs {len(b)}"
    return max((max_abs_diff_1d(ra, rb) for ra, rb in zip(a, b)), default=0.0)


def rand_matrix(rng: random.Random, rows: int, cols: int, scale: float = 1.0) -> list[list[float]]:
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def rand_vector(rng: random.Random, dim: int, scale: float = 1.0) -> list[float]:
    return [rng.uniform(-scale, scale) for _ in range(dim)]


def zeros_2d(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def matmul_dense(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    """View 1: Standard row-by-column inner product C[i, j] = a[i, :] . b[:, j]."""
    m, k = len(a), len(a[0])
    k2, n = len(b), len(b[0])
    assert k == k2
    out = zeros_2d(m, n)
    for i in range(m):
        for j in range(n):
            out[i][j] = sum(a[i][p] * b[p][j] for p in range(k))
    return out


def matmul_outer_sum(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    """View 4: Sum of rank-1 column-by-row outer products: C = sum_p a[:, p] @ b[p, :]."""
    m, k = len(a), len(a[0])
    k2, n = len(b), len(b[0])
    assert k == k2
    out = zeros_2d(m, n)
    for p in range(k):
        col_a = [a[i][p] for i in range(m)]
        row_b = b[p]
        for i in range(m):
            aip = col_a[i]
            for j in range(n):
                out[i][j] += aip * row_b[j]
    return out


def matmul_tiled(
    a: Sequence[Sequence[float]],
    b: Sequence[Sequence[float]],
    bm: int,
    bn: int,
    bk: int,
) -> tuple[list[list[float]], int]:
    """2D Block GEMM simulating GPU CTA SRAM tiling + register accumulator."""
    m, k = len(a), len(a[0])
    k2, n = len(b), len(b[0])
    assert k == k2
    c_global = zeros_2d(m, n)
    tile_steps = 0

    for i0 in range(0, m, bm):
        i_end = min(i0 + bm, m)
        for j0 in range(0, n, bn):
            j_end = min(j0 + bn, n)
            # Register accumulator for tile C[i0:i_end, j0:j_end]
            acc = zeros_2d(i_end - i0, j_end - j0)
            for p0 in range(0, k, bk):
                p_end = min(p0 + bk, k)
                tile_steps += 1
                # Load A_tile and B_tile into simulated Shared Memory (SRAM)
                a_sram = [row[p0:p_end] for row in a[i0:i_end]]
                b_sram = [row[j0:j_end] for row in b[p0:p_end]]
                # Tensor Core / FMA accumulate along local K tile
                for ii in range(i_end - i0):
                    for pp in range(p_end - p0):
                        val_a = a_sram[ii][pp]
                        for jj in range(j_end - j0):
                            acc[ii][jj] += val_a * b_sram[pp][jj]
            # Write register accumulator back to HBM once
            for ii in range(i_end - i0):
                for jj in range(j_end - j0):
                    c_global[i0 + ii][j0 + jj] = acc[ii][jj]
    return c_global, tile_steps


def transpose_2d(mat: Sequence[Sequence[float]]) -> list[list[float]]:
    rows, cols = len(mat), len(mat[0])
    return [[mat[r][c] for r in range(rows)] for c in range(cols)]


def attention_dense(
    q: Sequence[Sequence[float]],
    k: Sequence[Sequence[float]],
    v: Sequence[Sequence[float]],
    scale: float,
) -> list[list[float]]:
    """Reference full-materialization Attention: softmax(Q K^T * scale) V."""
    scores = matmul_dense(q, transpose_2d(k))
    probs = []
    for row in scores:
        scaled = [s * scale for s in row]
        m = max(scaled)
        exps = [math.exp(x - m) for x in scaled]
        denom = sum(exps)
        probs.append([e / denom for e in exps])
    return matmul_dense(probs, v)


def attention_tiled_online_softmax(
    q: Sequence[Sequence[float]],
    k: Sequence[Sequence[float]],
    v: Sequence[Sequence[float]],
    scale: float,
    br: int,
    bc: int,
) -> list[list[float]]:
    """FlashAttention / FlashInfer style block-tiled Attention without N x N score matrix."""
    n_q, d = len(q), len(q[0])
    n_kv, d_v = len(v), len(v[0])
    out = zeros_2d(n_q, d_v)

    for i0 in range(0, n_q, br):
        i_end = min(i0 + br, n_q)
        q_tile = q[i0:i_end]
        m_state = [-math.inf] * (i_end - i0)
        d_state = [0.0] * (i_end - i0)
        o_state = zeros_2d(i_end - i0, d_v)

        for j0 in range(0, n_kv, bc):
            j_end = min(j0 + bc, n_kv)
            k_tile = k[j0:j_end]
            v_tile = v[j0:j_end]
            # Block QK^T on tile
            s_tile = matmul_dense(q_tile, transpose_2d(k_tile))
            for r in range(i_end - i0):
                row_scores = [s * scale for s in s_tile[r]]
                m_new = max(m_state[r], max(row_scores))
                alpha = math.exp(m_state[r] - m_new) if m_state[r] > -math.inf else 0.0
                p_tilde = [math.exp(s - m_new) for s in row_scores]
                d_state[r] = d_state[r] * alpha + sum(p_tilde)
                for col_v in range(d_v):
                    pv_sum = sum(p_tilde[c] * v_tile[c][col_v] for c in range(j_end - j0))
                    o_state[r][col_v] = o_state[r][col_v] * alpha + pv_sum
                m_state[r] = m_new

        for r in range(i_end - i0):
            inv_d = 1.0 / d_state[r]
            out[i0 + r] = [val * inv_d for val in o_state[r]]
    return out


# ---------------------------------------------------------------------------
# Experiment 1: Outer-Product Sum & 2D Block GEMM Tiling
# ---------------------------------------------------------------------------
def run_exp1_block_gemm(rng: random.Random) -> float:
    print("=== [Exp 1] Outer-Product View & 2D Block GEMM Tiling ===")
    # Intentionally non-multiples of tile size to exercise boundary tail tiles
    m, k, n = 29, 37, 31
    bm, bn, bk = 8, 8, 8
    a = rand_matrix(rng, m, k, scale=1.0)
    b = rand_matrix(rng, k, n, scale=1.0)

    c_ref = matmul_dense(a, b)
    c_outer = matmul_outer_sum(a, b)
    c_tiled, tile_steps = matmul_tiled(a, b, bm=bm, bn=bn, bk=bk)

    err_outer = max_abs_diff_2d(c_ref, c_outer)
    err_tiled = max_abs_diff_2d(c_ref, c_tiled)

    # Tiled Attention (Block GEMM + Online Softmax)
    q = rand_matrix(rng, 19, 16, scale=0.8)
    k_mat = rand_matrix(rng, 23, 16, scale=0.8)
    v_mat = rand_matrix(rng, 23, 16, scale=0.8)
    scale = 1.0 / math.sqrt(16.0)
    att_ref = attention_dense(q, k_mat, v_mat, scale)
    att_tiled = attention_tiled_online_softmax(q, k_mat, v_mat, scale, br=6, bc=7)
    err_att = max_abs_diff_2d(att_ref, att_tiled)

    max_err = max(err_outer, err_tiled, err_att)
    print(
        f"GEMM shape=({m}x{k})@({k}x{n}), tile=({bm},{bn},{bk}), sram_tile_loads={tile_steps}"
    )
    print(
        f"  outer_sum_err={err_outer:.3e} | block_tiling_err={err_tiled:.3e} | "
        f"tiled_online_attn_err={err_att:.3e}"
    )
    assert max_err < TOL, f"Exp 1 error exceeded tolerance: {max_err}"
    return max_err


# ---------------------------------------------------------------------------
# Experiment 2: Orthogonal Rotation Matrices & RoPE Relative Position
# ---------------------------------------------------------------------------
def build_rope_matrix(dim: int, pos: int, base: float = 10000.0) -> list[list[float]]:
    """Construct block-diagonal orthogonal matrix R_pos in R^{dim x dim}."""
    assert dim % 2 == 0
    r = zeros_2d(dim, dim)
    for pair_idx in range(dim // 2):
        theta = base ** (-2.0 * pair_idx / dim)
        angle = pos * theta
        c, s = math.cos(angle), math.sin(angle)
        i0 = 2 * pair_idx
        r[i0][i0] = c
        r[i0][i0 + 1] = -s
        r[i0 + 1][i0] = s
        r[i0 + 1][i0 + 1] = c
    return r


def apply_rope_fast(vec: Sequence[float], pos: int, base: float = 10000.0) -> list[float]:
    """O(d) elementwise RoPE rotation used in Llama / FlashInfer kernels."""
    dim = len(vec)
    out = [0.0] * dim
    for pair_idx in range(dim // 2):
        theta = base ** (-2.0 * pair_idx / dim)
        angle = pos * theta
        c, s = math.cos(angle), math.sin(angle)
        x0, x1 = vec[2 * pair_idx], vec[2 * pair_idx + 1]
        out[2 * pair_idx] = x0 * c - x1 * s
        out[2 * pair_idx + 1] = x0 * s + x1 * c
    return out


def matvec(mat: Sequence[Sequence[float]], vec: Sequence[float]) -> list[float]:
    return [dot(row, vec) for row in mat]


def l2_norm(vec: Sequence[float]) -> float:
    return math.sqrt(dot(vec, vec))


def run_exp2_rope_rotation(rng: random.Random) -> float:
    print("\n=== [Exp 2] Orthogonal Rotation & RoPE Relative Position Invariance ===")
    dim = 64
    q = rand_vector(rng, dim, scale=1.5)
    k = rand_vector(rng, dim, scale=1.5)

    # Check R_m^T @ R_m == I
    r_sample = build_rope_matrix(dim, pos=17)
    gram = matmul_dense(transpose_2d(r_sample), r_sample)
    ident = [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]
    err_ortho = max_abs_diff_2d(gram, ident)

    max_rel_err = 0.0
    max_norm_err = 0.0
    max_fast_err = 0.0
    position_pairs = [(0, 0), (3, 11), (42, 15), (128, 512), (2047, 2040)]

    for m_pos, n_pos in position_pairs:
        r_m = build_rope_matrix(dim, m_pos)
        r_n = build_rope_matrix(dim, n_pos)
        r_diff = build_rope_matrix(dim, n_pos - m_pos)

        rq = matvec(r_m, q)
        rk = matvec(r_n, k)
        rq_fast = apply_rope_fast(q, m_pos)

        lhs = dot(rq, rk)
        rhs = dot(q, matvec(r_diff, k))
        max_rel_err = max(max_rel_err, abs(lhs - rhs))
        max_norm_err = max(max_norm_err, abs(l2_norm(rq) - l2_norm(q)))
        max_fast_err = max(max_fast_err, max_abs_diff_1d(rq, rq_fast))

    # Also verify translation invariance: shifting both m and n by +delta keeps <R_m q, R_n k> identical
    q_s1 = apply_rope_fast(q, 10)
    k_s1 = apply_rope_fast(k, 25)
    q_s2 = apply_rope_fast(q, 10 + 300)
    k_s2 = apply_rope_fast(k, 25 + 300)
    err_shift = abs(dot(q_s1, k_s1) - dot(q_s2, k_s2))

    max_err = max(err_ortho, max_rel_err, max_norm_err, max_fast_err, err_shift)
    print(
        f"head_dim={dim} ({dim // 2} 2D subspaces), tested pairs={len(position_pairs)}, "
        f"shift_dot(10,25)==(310,325): {dot(q_s1, k_s1):+.6f}"
    )
    print(
        f"  R^T R==I err={err_ortho:.3e} | norm_preserve_err={max_norm_err:.3e} | "
        f"<R_m q, R_n k>==<q, R_{{n-m}} k> err={max_rel_err:.3e} | shift_err={err_shift:.3e}"
    )
    assert max_err < TOL, f"Exp 2 error exceeded tolerance: {max_err}"
    return max_err


# ---------------------------------------------------------------------------
# Experiment 3: DeepSeek MLA Associativity & Weight Absorption
# ---------------------------------------------------------------------------
def run_exp3_mla_absorption(rng: random.Random) -> float:
    print("\n=== [Exp 3] Matrix Associativity & DeepSeek MLA Weight Absorption ===")
    num_heads = 8
    kv_len = 24
    d_c = 32      # Latent KV dimension (512 in DeepSeek-V3)
    d_nope = 16   # Non-RoPE head dim (128 in DeepSeek-V3)
    d_r = 8       # Decoupled RoPE dim (64 in DeepSeek-V3)
    d_v = 16      # Value head dim (128 in DeepSeek-V3)
    sm_scale = 1.0 / math.sqrt(d_nope + d_r)

    # Decode step: 1 query token with H heads; kv_len cached latent tokens
    q_nope = [rand_vector(rng, d_nope, scale=0.6) for _ in range(num_heads)]
    q_pe = [rand_vector(rng, d_r, scale=0.6) for _ in range(num_heads)]
    c_kv = rand_matrix(rng, kv_len, d_c, scale=0.6)  # Shared latent KV across all heads!
    k_pe = rand_matrix(rng, kv_len, d_r, scale=0.6)  # Shared RoPE key across all heads!

    w_uk = [rand_matrix(rng, d_c, d_nope, scale=0.4) for _ in range(num_heads)]
    w_uv = [rand_matrix(rng, d_c, d_v, scale=0.4) for _ in range(num_heads)]

    out_unabsorbed: list[list[float]] = []
    out_absorbed: list[list[float]] = []
    max_score_err = 0.0

    for h in range(num_heads):
        # Path A (Unabsorbed): Decompress K_nope[t] = c_kv[t] @ W_UK[h], V[t] = c_kv[t] @ W_UV[h]
        k_nope_h = matmul_dense(c_kv, w_uk[h])  # [kv_len, d_nope]
        v_h = matmul_dense(c_kv, w_uv[h])       # [kv_len, d_v]
        scores_a = [
            (dot(q_nope[h], k_nope_h[t]) + dot(q_pe[h], k_pe[t])) * sm_scale
            for t in range(kv_len)
        ]
        m_a = max(scores_a)
        exp_a = [math.exp(s - m_a) for s in scores_a]
        denom_a = sum(exp_a)
        probs_a = [e / denom_a for e in exp_a]
        out_a = [
            sum(probs_a[t] * v_h[t][ idx] for t in range(kv_len))
            for idx in range(d_v)
        ]
        out_unabsorbed.append(out_a)

        # Path B (Absorbed): q_tilde = q_nope[h] @ W_UK[h]^T in R^{d_c}; attend directly to c_kv!
        q_tilde = matvec(w_uk[h], q_nope[h])  # Equivalent to row vec q_nope @ W_UK^T, len = d_c
        scores_b = [
            (dot(q_tilde, c_kv[t]) + dot(q_pe[h], k_pe[t])) * sm_scale
            for t in range(kv_len)
        ]
        max_score_err = max(max_score_err, max_abs_diff_1d(scores_a, scores_b))

        m_b = max(scores_b)
        exp_b = [math.exp(s - m_b) for s in scores_b]
        denom_b = sum(exp_b)
        probs_b = [e / denom_b for e in exp_b]
        # Latent output o_tilde in R^{d_c}, then multiply W_UV[h] once!
        o_tilde = [
            sum(probs_b[t] * c_kv[t][c_idx] for t in range(kv_len))
            for c_idx in range(d_c)
        ]
        out_b = [
            sum(o_tilde[c_idx] * w_uv[h][c_idx][v_idx] for c_idx in range(d_c))
            for v_idx in range(d_v)
        ]
        out_absorbed.append(out_b)

    err_out = max_abs_diff_2d(out_unabsorbed, out_absorbed)

    # HBM Bandwidth Audit at DeepSeek-V3/R1 architectural dimensions (BF16 = 2 bytes, seq_len = 8192)
    ds_heads, ds_dnope, ds_dr, ds_dv, ds_dc = 128, 128, 64, 128, 512
    ds_seq = 8192
    bytes_per_elem = 2  # BF16
    # Standard uncompressed MHA cache per token: H * ((d_nope + d_r) + d_v) * 2B
    mha_bytes_per_tok = ds_heads * ((ds_dnope + ds_dr) + ds_dv) * bytes_per_elem
    # Decompressed K_nope + V + shared k_pe per token
    decomp_bytes_per_tok = (ds_heads * (ds_dnope + ds_dv) + ds_dr) * bytes_per_elem
    # MLA absorbed cache per token: (d_c + d_r) * 2B
    mla_bytes_per_tok = (ds_dc + ds_dr) * bytes_per_elem

    max_err = max(max_score_err, err_out)
    print(
        f"Numerical check (H={num_heads}, T={kv_len}, d_c={d_c}, d_nope={d_nope}, d_r={d_r}): "
        f"score_err={max_score_err:.3e} | out_err={err_out:.3e}"
    )
    print(
        f"DeepSeek-V3 scale (H=128, T={ds_seq}, BF16) KV HBM read per Decode step:\n"
        f"  Full MHA KV     : {mha_bytes_per_tok * ds_seq / (1024**2):7.2f} MiB "
        f"({mha_bytes_per_tok:,} B/tok)\n"
        f"  Unabsorbed KV   : {decomp_bytes_per_tok * ds_seq / (1024**2):7.2f} MiB "
        f"({decomp_bytes_per_tok:,} B/tok)\n"
        f"  MLA Absorbed KV : {mla_bytes_per_tok * ds_seq / (1024**2):7.2f} MiB "
        f"({mla_bytes_per_tok:,} B/tok) -> {mha_bytes_per_tok / mla_bytes_per_tok:.2f}x smaller vs Full MHA "
        f"({decomp_bytes_per_tok / mla_bytes_per_tok:.2f}x vs Unabsorbed)"
    )
    assert max_err < TOL, f"Exp 3 error exceeded tolerance: {max_err}"
    return max_err


# ---------------------------------------------------------------------------
# Experiment 4: SVD Outer-Product Expansion & Eckart-Young Low-Rank Truncation
# ---------------------------------------------------------------------------
def orthonormalize_columns(mat: Sequence[Sequence[float]]) -> list[list[float]]:
    """Modified Gram-Schmidt producing orthonormal columns Q in R^{m x n} (m >= n)."""
    m, n = len(mat), len(mat[0])
    cols = [[mat[r][c] for r in range(m)] for c in range(n)]
    q_cols: list[list[float]] = []
    for c in range(n):
        v = list(cols[c])
        for u in q_cols:
            proj = dot(u, v)
            v = [vi - proj * ui for vi, ui in zip(v, u)]
        norm = l2_norm(v)
        q_cols.append([vi / norm for vi in v])
    return [[q_cols[c][r] for c in range(n)] for r in range(m)]


def frobenius_norm_diff(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    sq = 0.0
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            d = x - y
            sq += d * d
    return math.sqrt(sq)


def run_exp4_svd_low_rank(rng: random.Random) -> float:
    print("\n=== [Exp 4] SVD Rank-1 Outer Products, Eckart-Young & LoRA Compression ===")
    dim = 32
    u_ortho = orthonormalize_columns(rand_matrix(rng, dim, dim, scale=1.0))
    v_ortho = orthonormalize_columns(rand_matrix(rng, dim, dim, scale=1.0))
    # Rapidly decaying singular spectrum mimicking LLM weight update Delta W
    sigmas = [10.0 * (0.65 ** i) for i in range(dim)]

    # W = U @ diag(sigmas) @ V^T
    u_scaled = [[u_ortho[r][i] * sigmas[i] for i in range(dim)] for r in range(dim)]
    w_full = matmul_dense(u_scaled, transpose_2d(v_ortho))

    # Reconstruct via sum of rank-1 outer products: sum_{i=1}^k sigma_i u_i v_i^T
    total_energy = sum(s * s for s in sigmas)
    w_accum = zeros_2d(dim, dim)
    max_ey_err = 0.0
    checkpoints = {1, 2, 4, 8, 16, 32}

    print("  rank_k | energy_kept | empirical ||W - W_k||_F | theoretical sqrt(sum_{i>k} s_i^2) | abs_diff")
    for idx in range(dim):
        s = sigmas[idx]
        u_col = [u_ortho[r][idx] for r in range(dim)]
        v_col = [v_ortho[r][idx] for r in range(dim)]
        for r in range(dim):
            ur = s * u_col[r]
            for c in range(dim):
                w_accum[r][c] += ur * v_col[c]

        k = idx + 1
        if k in checkpoints:
            emp_frob = frobenius_norm_diff(w_full, w_accum)
            theo_frob = math.sqrt(sum(sv * sv for sv in sigmas[k:]))
            ey_diff = abs(emp_frob - theo_frob)
            max_ey_err = max(max_ey_err, ey_diff)
            kept_pct = 100.0 * sum(sv * sv for sv in sigmas[:k]) / total_energy
            print(
                f"  k={k:4d} |   {kept_pct:8.4f}% |          {emp_frob:14.6e} |                    {theo_frob:14.6e} | {ey_diff:.2e}"
            )

    err_full_svd = max_abs_diff_2d(w_full, w_accum)

    # Verify LoRA B @ A == sum of r rank-1 outer products + parameter saving ratio at d=4096, r=16
    r_lora = 4
    b_lora = [[u_ortho[row][i] * math.sqrt(sigmas[i]) for i in range(r_lora)] for row in range(dim)]
    a_lora = [[math.sqrt(sigmas[i]) * v_ortho[col][i] for col in range(dim)] for i in range(r_lora)]
    delta_w_ba = matmul_dense(b_lora, a_lora)
    w_rank4 = zeros_2d(dim, dim)
    for i in range(r_lora):
        for row in range(dim):
            for col in range(dim):
                w_rank4[row][col] += sigmas[i] * u_ortho[row][i] * v_ortho[col][i]
    err_lora_equiv = max_abs_diff_2d(delta_w_ba, w_rank4)

    d_llm, r_llm = 4096, 16
    ratio_pct = 100.0 * (d_llm * r_llm + r_llm * d_llm) / (d_llm * d_llm)
    print(
        f"  LoRA equivalence (B@A == sum_{{i=1}}^4 s_i u_i v_i^T) err={err_lora_equiv:.3e} | "
        f"LLM d={d_llm}, r={r_llm} trainable params={ratio_pct:.3f}% (128x fewer)"
    )

    max_err = max(max_ey_err, err_full_svd, err_lora_equiv)
    assert max_err < TOL, f"Exp 4 error exceeded tolerance: {max_err}"
    return max_err


# ---------------------------------------------------------------------------
# Experiment 5: 8-Lane SIMD Vectorized Dot Product + Butterfly Reduction & Stride
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SimdVec8:
    """Simulated 8-lane SIMD register (e.g., 256-bit AVX2 8xFP32 or 128-bit 8xFP16 vec_t)."""

    lanes: tuple[float, float, float, float, float, float, float, float]

    @staticmethod
    def splat(val: float = 0.0) -> "SimdVec8":
        return SimdVec8((val,) * 8)

    @staticmethod
    def load_contiguous(buf: Sequence[float], offset: int, mask: Sequence[bool] | None = None) -> "SimdVec8":
        if mask is None:
            return SimdVec8(tuple(buf[offset : offset + 8]))  # type: ignore[arg-type]
        vals = [
            buf[offset + lane] if (mask[lane] and offset + lane < len(buf)) else 0.0
            for lane in range(8)
        ]
        return SimdVec8(tuple(vals))  # type: ignore[arg-type]

    @staticmethod
    def fma(acc: "SimdVec8", a: "SimdVec8", b: "SimdVec8") -> "SimdVec8":
        return SimdVec8(tuple(acc.lanes[i] + a.lanes[i] * b.lanes[i] for i in range(8)))  # type: ignore[arg-type]

    def butterfly_reduce_sum(self) -> float:
        """3-step XOR butterfly tree reduction (offsets 4 -> 2 -> 1), matching __shfl_xor_sync."""
        cur = list(self.lanes)
        for xor_mask in (4, 2, 1):
            cur = [cur[lane] + cur[lane ^ xor_mask] for lane in range(8)]
        # Every lane now holds the exact total horizontal sum!
        assert max(abs(x - cur[0]) for x in cur) < TOL
        return cur[0]


@dataclass(frozen=True)
class StridedTensorView:
    """Physical 1D storage + (shape, strides) metadata demonstrating zero-copy transpose."""

    storage: list[float]
    shape: tuple[int, ...]
    strides: tuple[int, ...]

    def get(self, *indices: int) -> float:
        phys_offset = sum(idx * st for idx, st in zip(indices, self.strides))
        return self.storage[phys_offset]

    def transpose(self, dim0: int, dim1: int) -> "StridedTensorView":
        new_shape = list(self.shape)
        new_strides = list(self.strides)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        new_strides[dim0], new_strides[dim1] = new_strides[dim1], new_strides[dim0]
        # Zero bytes copied: reuses the exact same `self.storage` list object!
        return StridedTensorView(self.storage, tuple(new_shape), tuple(new_strides))


def simd8_dot_product(a: Sequence[float], b: Sequence[float]) -> tuple[float, int, int]:
    """Compute dot product via 8-lane FMA accumulators + tail mask + butterfly reduction."""
    assert len(a) == len(b)
    n = len(a)
    acc = SimdVec8.splat(0.0)
    full_vecs = n // 8
    tail = n % 8

    for idx in range(full_vecs):
        base = idx * 8
        va = SimdVec8.load_contiguous(a, base)
        vb = SimdVec8.load_contiguous(b, base)
        acc = SimdVec8.fma(acc, va, vb)

    masked_loads = 0
    if tail > 0:
        base = full_vecs * 8
        mask = [lane < tail for lane in range(8)]
        va = SimdVec8.load_contiguous(a, base, mask=mask)
        vb = SimdVec8.load_contiguous(b, base, mask=mask)
        acc = SimdVec8.fma(acc, va, vb)
        masked_loads = 1

    return acc.butterfly_reduce_sum(), full_vecs, masked_loads


def run_exp5_simd_and_strides(rng: random.Random) -> float:
    print("\n=== [Exp 5] 8-Lane SIMD FMA + Butterfly Reduction & Tensor Stride View ===")
    vec_len = 133  # 16 full 8-lane vectors + 5 tail elements via predicate mask
    x = rand_vector(rng, vec_len, scale=2.0)
    y = rand_vector(rng, vec_len, scale=2.0)

    ref_dot = dot(x, y)
    simd_dot, full_vecs, masked_loads = simd8_dot_product(x, y)
    err_simd = abs(ref_dot - simd_dot)

    # NHD [seq=16, heads=4, dim=8] physical layout vs zero-copy HND transpose
    seq_len, heads, head_dim = 16, 4, 8
    raw_buf = rand_vector(rng, seq_len * heads * head_dim, scale=1.0)
    nhd_view = StridedTensorView(
        storage=raw_buf,
        shape=(seq_len, heads, head_dim),
        strides=(heads * head_dim, head_dim, 1),
    )
    hnd_view = nhd_view.transpose(0, 1)  # shape=(4, 16, 8), strides=(8, 32, 1)
    dn_view = nhd_view.transpose(0, 2)   # shape=(8, 4, 16), strides=(1, 8, 32) -> non-contiguous last dim!

    assert nhd_view.storage is hnd_view.storage
    err_view = 0.0
    for t in range(seq_len):
        for h in range(heads):
            for d in range(head_dim):
                err_view = max(err_view, abs(nhd_view.get(t, h, d) - hnd_view.get(h, t, d)))

    print(
        f"SIMD8 dot (D={vec_len} = {full_vecs}x8 + {vec_len % 8} masked tail): "
        f"scalar={ref_dot:+.8f}, simd8={simd_dot:+.8f}, abs_err={err_simd:.3e}"
    )
    print(
        f"Zero-copy transpose check (same buffer={nhd_view.storage is hnd_view.storage}, err={err_view:.1e}):\n"
        f"  NHD shape={nhd_view.shape}, strides={nhd_view.strides} -> inner stride={nhd_view.strides[-1]} (SIMD/LDG.128 contiguous)\n"
        f"  HND shape={hnd_view.shape}, strides={hnd_view.strides} -> inner stride={hnd_view.strides[-1]} (per-head token jump={hnd_view.strides[1]})\n"
        f"  DHN shape={dn_view.shape}, strides={dn_view.strides} -> inner stride={dn_view.strides[-1]} (strided gather penalty!)"
    )

    max_err = max(err_simd, err_view)
    assert max_err < TOL, f"Exp 5 error exceeded tolerance: {max_err}"
    return max_err


def main() -> None:
    rng = random.Random(SEED)
    errs = [
        run_exp1_block_gemm(rng),
        run_exp2_rope_rotation(rng),
        run_exp3_mla_absorption(rng),
        run_exp4_svd_low_rank(rng),
        run_exp5_simd_and_strides(rng),
    ]
    worst = max(errs)
    print(f"\nMAX ABS ERROR ACROSS ALL 5 EXPERIMENTS: {worst:.3e} (< {TOL:.0e})")
    assert worst < TOL
    print("SELF-CHECK: PASS")


if __name__ == "__main__":
    main()
