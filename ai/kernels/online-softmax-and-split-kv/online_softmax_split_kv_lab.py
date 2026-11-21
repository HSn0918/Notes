#!/usr/bin/env python3
"""Stdlib-only CPU lab for FlashInfer Paged KV addressing, base-2 Online Softmax,
Split-KV Decode, and Cascade Shared-Prefix state merging.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

SEED = 20261117
LOG2_E = math.log2(math.e)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class PagedKVCache:
    """Minimal NHD Paged KV Cache matching FlashInfer's `paged_kv_t` metadata.

    Storage layout (NHD): [num_pages, page_size, num_kv_heads, head_dim].
    CSR metadata:
      - indptr[b] .. indptr[b + 1]: slice of physical page IDs in `indices`
      - last_page_len[b]: valid token count in the final page of request `b`
    """

    num_kv_heads: int
    page_size: int
    head_dim: int
    indptr: list[int]
    indices: list[int]
    last_page_len: list[int]
    k_pool: list[list[list[list[float]]]]
    v_pool: list[list[list[list[float]]]]

    @property
    def batch_size(self) -> int:
        return len(self.last_page_len)

    def kv_len(self, batch_idx: int) -> int:
        num_pages = self.indptr[batch_idx + 1] - self.indptr[batch_idx]
        if num_pages == 0:
            return 0
        return (num_pages - 1) * self.page_size + self.last_page_len[batch_idx]

    def locate(self, batch_idx: int, token_pos: int) -> tuple[int, int, int]:
        """Map logical token position to (page_iter, phys_page_id, entry_idx)."""
        if not 0 <= token_pos < self.kv_len(batch_idx):
            raise IndexError(f"token_pos={token_pos} out of range for batch={batch_idx}")
        page_offset, entry_idx = divmod(token_pos, self.page_size)
        page_iter = self.indptr[batch_idx] + page_offset
        phys_page_id = self.indices[page_iter]
        return page_iter, phys_page_id, entry_idx

    def elem_offset_nhd(
        self, phys_page_id: int, kv_head_idx: int, entry_idx: int, feat_idx: int = 0
    ) -> int:
        """Exact `paged_kv_t::get_elem_offset` in NHD layout."""
        stride_h = self.head_dim
        stride_n = self.num_kv_heads * self.head_dim
        stride_page = self.page_size * stride_n
        return (
            phys_page_id * stride_page
            + entry_idx * stride_n
            + kv_head_idx * stride_h
            + feat_idx
        )

    def get_kv(
        self, batch_idx: int, token_pos: int, kv_head_idx: int
    ) -> tuple[list[float], list[float]]:
        _, phys_page_id, entry_idx = self.locate(batch_idx, token_pos)
        return (
            self.k_pool[phys_page_id][entry_idx][kv_head_idx],
            self.v_pool[phys_page_id][entry_idx][kv_head_idx],
        )

    def materialize_dense(
        self, batch_idx: int
    ) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
        """Reconstruct [kv_len, num_kv_heads, head_dim] tensors for reference check."""
        length = self.kv_len(batch_idx)
        k_dense: list[list[list[float]]] = []
        v_dense: list[list[list[float]]] = []
        for pos in range(length):
            k_heads = []
            v_heads = []
            for h in range(self.num_kv_heads):
                k_vec, v_vec = self.get_kv(batch_idx, pos, h)
                k_heads.append(list(k_vec))
                v_heads.append(list(v_vec))
            k_dense.append(k_heads)
            v_dense.append(v_heads)
        return k_dense, v_dense


@dataclass
class AttentionState:
    """FlashInfer-style `state_t` using base-2 log-max `m_log2` for exp2 rescaling."""

    head_dim: int
    m_log2: float = -math.inf
    d: float = 1.0
    o: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.o:
            self.o = [0.0] * self.head_dim

    def update_tile(
        self, scores_log2: Sequence[float], v_vecs: Sequence[Sequence[float]]
    ) -> None:
        if not scores_log2:
            return
        m_prev = self.m_log2
        m_new = max(m_prev, max(scores_log2))
        if m_new == -math.inf:
            return
        o_scale = 0.0 if m_prev == -math.inf else 2.0 ** (m_prev - m_new)
        self.d *= o_scale
        for idx in range(self.head_dim):
            self.o[idx] *= o_scale

        for score, v_vec in zip(scores_log2, v_vecs):
            w = 0.0 if score == -math.inf else 2.0 ** (score - m_new)
            self.d += w
            for idx in range(self.head_dim):
                self.o[idx] += w * v_vec[idx]
        self.m_log2 = m_new

    def merge_raw(self, other: AttentionState) -> None:
        """Associative merge of two unnormalized `state_t` accumulators."""
        if other.m_log2 == -math.inf:
            if self.m_log2 == -math.inf:
                self.d = 0.0
            return
        m_prev = self.m_log2
        d_prev = self.d
        m_new = max(m_prev, other.m_log2)
        f1 = 0.0 if m_prev == -math.inf else 2.0 ** (m_prev - m_new)
        f2 = 2.0 ** (other.m_log2 - m_new)
        self.d = d_prev * f1 + other.d * f2
        for idx in range(self.head_dim):
            self.o[idx] = self.o[idx] * f1 + other.o[idx] * f2
        self.m_log2 = m_new

    def merge_normalized(self, v_other: Sequence[float], s_other_log2: float) -> None:
        """Merge a normalized partial output + base-2 LSE (`MergeStatesKernel` style).

        Matches `st.merge(v, s, 1)` in `cascade.cuh`:
        since `v_other = o_other / d_other` and `s_other = m_other + log2(d_other)`,
        setting `other_m = s_other` and `other_d = 1.0` yields:
        `exp2(s_other - m_new) * v_other = exp2(m_other - m_new) * o_other`.
        """
        if s_other_log2 == -math.inf:
            if self.m_log2 == -math.inf:
                self.d = 0.0
            return
        m_prev = self.m_log2
        d_prev = self.d
        m_new = max(m_prev, s_other_log2)
        f1 = 0.0 if m_prev == -math.inf else 2.0 ** (m_prev - m_new)
        f2 = 2.0 ** (s_other_log2 - m_new)
        self.d = d_prev * f1 + 1.0 * f2
        for idx in range(self.head_dim):
            self.o[idx] = self.o[idx] * f1 + v_other[idx] * f2
        self.m_log2 = m_new

    def normalized_output(self) -> list[float]:
        if self.m_log2 == -math.inf or self.d <= 0.0:
            return [0.0] * self.head_dim
        d_rcp = 1.0 / self.d
        return [val * d_rcp for val in self.o]

    def get_lse(self) -> float:
        if self.m_log2 == -math.inf or self.d <= 0.0:
            return -math.inf
        return self.m_log2 + math.log2(self.d)


def dense_reference_gqa(
    q_heads: Sequence[Sequence[float]],
    k_dense: Sequence[Sequence[Sequence[float]]],
    v_dense: Sequence[Sequence[Sequence[float]]],
    sm_scale: float,
) -> tuple[list[list[float]], list[float]]:
    """Standard 2-pass safe-softmax GQA reference."""
    num_qo_heads = len(q_heads)
    num_kv_heads = len(k_dense[0])
    group_size = num_qo_heads // num_kv_heads
    head_dim = len(q_heads[0])

    outputs: list[list[float]] = []
    lse_log2: list[float] = []
    for qo_h in range(num_qo_heads):
        kv_h = qo_h // group_size
        q_vec = q_heads[qo_h]
        logits = [dot(q_vec, k_dense[t][kv_h]) * sm_scale for t in range(len(k_dense))]
        m_nat = max(logits)
        weights = [math.exp(x - m_nat) for x in logits]
        denom = sum(weights)
        probs = [w / denom for w in weights]
        out_vec = [0.0] * head_dim
        for t, p in enumerate(probs):
            v_vec = v_dense[t][kv_h]
            for d_idx in range(head_dim):
                out_vec[d_idx] += p * v_vec[d_idx]
        outputs.append(out_vec)
        lse_log2.append((m_nat + math.log(denom)) * LOG2_E)
    return outputs, lse_log2


def compute_paged_chunk_state(
    q_heads: Sequence[Sequence[float]],
    paged_kv: PagedKVCache,
    batch_idx: int,
    start_pos: int,
    end_pos: int,
    sm_scale: float,
    tile_size: int = 3,
) -> list[AttentionState]:
    """Compute per-QO-head `AttentionState` over paged tokens `[start_pos, end_pos)`."""
    num_qo_heads = len(q_heads)
    group_size = num_qo_heads // paged_kv.num_kv_heads
    sm_scale_log2 = sm_scale * LOG2_E
    states = [AttentionState(paged_kv.head_dim) for _ in range(num_qo_heads)]

    for tile_start in range(start_pos, end_pos, tile_size):
        tile_end = min(tile_start + tile_size, end_pos)
        for kv_h in range(paged_kv.num_kv_heads):
            k_tile: list[list[float]] = []
            v_tile: list[list[float]] = []
            for pos in range(tile_start, tile_end):
                k_vec, v_vec = paged_kv.get_kv(batch_idx, pos, kv_h)
                k_tile.append(k_vec)
                v_tile.append(v_vec)
            for g in range(group_size):
                qo_h = kv_h * group_size + g
                scores_log2 = [dot(q_heads[qo_h], k_vec) * sm_scale_log2 for k_vec in k_tile]
                states[qo_h].update_tile(scores_log2, v_tile)
    return states


def split_kv_decode(
    q_heads: Sequence[Sequence[float]],
    paged_kv: PagedKVCache,
    batch_idx: int,
    sm_scale: float,
    kv_chunk_size: int,
    merge_order: str = "forward",
    rng: random.Random | None = None,
) -> tuple[
    list[tuple[int, int, list[AttentionState], list[list[float]], list[float]]],
    list[list[float]],
    list[float],
]:
    """Simulate Stage-1 Split-KV CTAs + Stage-2 `MergeStates`."""
    kv_len = paged_kv.kv_len(batch_idx)
    num_qo_heads = len(q_heads)
    chunks = [
        (start, min(start + kv_chunk_size, kv_len))
        for start in range(0, kv_len, kv_chunk_size)
    ]

    # Stage 1: independent CTAs write normalized partial (V_chunk, S_chunk) to workspace
    stage1_partials: list[
        tuple[int, int, list[AttentionState], list[list[float]], list[float]]
    ] = []
    for c_start, c_end in chunks:
        chunk_states = compute_paged_chunk_state(
            q_heads, paged_kv, batch_idx, c_start, c_end, sm_scale, tile_size=3
        )
        v_chunk = [st.normalized_output() for st in chunk_states]
        s_chunk = [st.get_lse() for st in chunk_states]
        stage1_partials.append((c_start, c_end, chunk_states, v_chunk, s_chunk))

    indices = list(range(len(stage1_partials)))
    if merge_order == "reverse":
        indices.reverse()
    elif merge_order == "shuffled":
        assert rng is not None
        rng.shuffle(indices)
    elif merge_order != "forward":
        raise ValueError(f"Unknown merge_order={merge_order}")

    # Stage 2: MergeStatesKernel reduces partial states in arbitrary associative order
    merged_states = [AttentionState(paged_kv.head_dim) for _ in range(num_qo_heads)]
    for idx in indices:
        _, _, _, v_chunk, s_chunk = stage1_partials[idx]
        for qo_h in range(num_qo_heads):
            merged_states[qo_h].merge_normalized(v_chunk[qo_h], s_chunk[qo_h])

    merged_out = [st.normalized_output() for st in merged_states]
    merged_lse = [st.get_lse() for st in merged_states]
    return stage1_partials, merged_out, merged_lse


def build_synthetic_batch(
    rng: random.Random,
    num_qo_heads: int = 8,
    num_kv_heads: int = 2,
    head_dim: int = 6,
    page_size: int = 4,
    seq_lens: Sequence[int] = (5, 19, 12),
) -> tuple[
    list[list[list[float]]],
    PagedKVCache,
    list[list[list[list[float]]]],
    list[list[list[list[float]]]],
]:
    """Allocate shuffled physical pages and populate Q + Paged KV Cache."""
    pages_needed = [math.ceil(length / page_size) for length in seq_lens]
    total_pages = sum(pages_needed) + 2  # keep 2 unused spare pages in pool
    shuffled_pages = list(range(total_pages))
    rng.shuffle(shuffled_pages)

    k_pool = [
        [
            [[0.0 for _ in range(head_dim)] for _ in range(num_kv_heads)]
            for _ in range(page_size)
        ]
        for _ in range(total_pages)
    ]
    v_pool = [
        [
            [[0.0 for _ in range(head_dim)] for _ in range(num_kv_heads)]
            for _ in range(page_size)
        ]
        for _ in range(total_pages)
    ]

    indptr = [0]
    indices: list[int] = []
    last_page_len: list[int] = []
    gt_k_all: list[list[list[list[float]]]] = []
    gt_v_all: list[list[list[list[float]]]] = []
    cursor = 0

    for length, num_pages in zip(seq_lens, pages_needed):
        req_pages = shuffled_pages[cursor : cursor + num_pages]
        cursor += num_pages
        indices.extend(req_pages)
        indptr.append(indptr[-1] + num_pages)
        rem = length % page_size
        last_page_len.append(page_size if (length > 0 and rem == 0) else rem)

        gt_k_seq: list[list[list[float]]] = []
        gt_v_seq: list[list[list[float]]] = []
        for pos in range(length):
            p_off, entry = divmod(pos, page_size)
            phys_p = req_pages[p_off]
            k_heads = []
            v_heads = []
            for kv_h in range(num_kv_heads):
                k_vec = [rng.uniform(-1.0, 1.0) for _ in range(head_dim)]
                v_vec = [rng.uniform(-1.0, 1.0) for _ in range(head_dim)]
                k_pool[phys_p][entry][kv_h] = k_vec
                v_pool[phys_p][entry][kv_h] = v_vec
                k_heads.append(k_vec)
                v_heads.append(v_vec)
            gt_k_seq.append(k_heads)
            gt_v_seq.append(v_heads)
        gt_k_all.append(gt_k_seq)
        gt_v_all.append(gt_v_seq)

    q_batch = [
        [[rng.uniform(-1.0, 1.0) for _ in range(head_dim)] for _ in range(num_qo_heads)]
        for _ in seq_lens
    ]
    paged_kv = PagedKVCache(
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        head_dim=head_dim,
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        k_pool=k_pool,
        v_pool=v_pool,
    )
    return q_batch, paged_kv, gt_k_all, gt_v_all


def max_abs_diff_2d(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    return max(abs(x - y) for row_a, row_b in zip(a, b) for x, y in zip(row_a, row_b))


def max_abs_diff_1d(a: Sequence[float], b: Sequence[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def check_empty_state_nan_guard(head_dim: int) -> None:
    """Verify two empty (-inf) states merge to zero without NaN (state.cuh invariant)."""
    a = AttentionState(head_dim)
    b = AttentionState(head_dim)
    a.merge_raw(b)
    assert a.m_log2 == -math.inf and a.d == 0.0
    assert all(x == 0.0 for x in a.normalized_output())
    c = AttentionState(head_dim)
    c.merge_normalized([0.0] * head_dim, -math.inf)
    assert c.m_log2 == -math.inf and all(x == 0.0 for x in c.normalized_output())


def run_cascade_shared_prefix_check(rng: random.Random) -> float:
    """Verify MultiLevelCascadeAttentionWrapper's prefix+suffix state merge."""
    num_qo_heads = 8
    num_kv_heads = 2
    head_dim = 6
    page_size = 4
    sm_scale = 1.0 / math.sqrt(head_dim)

    # Unified PagedKVCache with 3 levels:
    # Level 0 (shared system prompt): len=7, pages=[5, 2]
    # Level 1 (shared few-shot ctx):  len=5, pages=[9, 1]
    # Level 2 (unique per request):   req0 len=6 pages=[4, 7], req1 len=3 pages=[0]
    total_pages = 10
    k_pool = [
        [
            [[rng.uniform(-1.0, 1.0) for _ in range(head_dim)] for _ in range(num_kv_heads)]
            for _ in range(page_size)
        ]
        for _ in range(total_pages)
    ]
    v_pool = [
        [
            [[rng.uniform(-1.0, 1.0) for _ in range(head_dim)] for _ in range(num_kv_heads)]
            for _ in range(page_size)
        ]
        for _ in range(total_pages)
    ]

    l0_kv = PagedKVCache(num_kv_heads, page_size, head_dim, [0, 2], [5, 2], [3], k_pool, v_pool)
    l1_kv = PagedKVCache(num_kv_heads, page_size, head_dim, [0, 2], [9, 1], [1], k_pool, v_pool)
    l2_kv = PagedKVCache(
        num_kv_heads, page_size, head_dim, [0, 2, 3], [4, 7, 0], [2, 3], k_pool, v_pool
    )

    l0_k, l0_v = l0_kv.materialize_dense(0)
    l1_k, l1_v = l1_kv.materialize_dense(0)

    q_batch = [
        [[rng.uniform(-1.0, 1.0) for _ in range(head_dim)] for _ in range(num_qo_heads)]
        for _ in range(2)
    ]

    max_err = 0.0
    print("\n=== Cascade Shared-Prefix Merge (3 Levels) ===")
    for req_idx in range(2):
        q_heads = q_batch[req_idx]
        l2_k, l2_v = l2_kv.materialize_dense(req_idx)
        full_k = l0_k + l1_k + l2_k
        full_v = l0_v + l1_v + l2_v
        ref_out, ref_lse = dense_reference_gqa(q_heads, full_k, full_v, sm_scale)

        st_l0 = compute_paged_chunk_state(q_heads, l0_kv, 0, 0, l0_kv.kv_len(0), sm_scale)
        st_l1 = compute_paged_chunk_state(q_heads, l1_kv, 0, 0, l1_kv.kv_len(0), sm_scale)
        st_l2 = compute_paged_chunk_state(
            q_heads, l2_kv, req_idx, 0, l2_kv.kv_len(req_idx), sm_scale
        )

        # MultiLevelCascadeAttentionWrapper.run() computes bottom level first, then
        # merges upper shared levels via merge_state_in_place(out, lse, out_i, lse_i)
        merged_states = [AttentionState(head_dim) for _ in range(num_qo_heads)]
        for h in range(num_qo_heads):
            merged_states[h].merge_normalized(st_l2[h].normalized_output(), st_l2[h].get_lse())
            merged_states[h].merge_normalized(st_l0[h].normalized_output(), st_l0[h].get_lse())
            merged_states[h].merge_normalized(st_l1[h].normalized_output(), st_l1[h].get_lse())

        cas_out = [st.normalized_output() for st in merged_states]
        cas_lse = [st.get_lse() for st in merged_states]
        err_out = max_abs_diff_2d(cas_out, ref_out)
        err_lse = max_abs_diff_1d(cas_lse, ref_lse)
        max_err = max(max_err, err_out, err_lse)
        print(
            f"req={req_idx} total_kv={len(full_k)} (L0={len(l0_k)}, L1={len(l1_k)}, L2={len(l2_k)}) "
            f"| L0_lse_h0={st_l0[0].get_lse():+.4f} L2_lse_h0={st_l2[0].get_lse():+.4f} "
            f"| max_out_err={err_out:.3e} | max_lse_err={err_lse:.3e}"
        )
        assert err_out < 1e-12 and err_lse < 1e-12
    return max_err


def main() -> None:
    rng = random.Random(SEED)
    seq_lens = (5, 19, 12)
    page_size = 4
    num_qo_heads = 8
    num_kv_heads = 2
    head_dim = 6
    sm_scale = 1.0 / math.sqrt(head_dim)
    kv_chunk_size = 6

    check_empty_state_nan_guard(head_dim)
    q_batch, paged_kv, gt_k_all, gt_v_all = build_synthetic_batch(
        rng,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        seq_lens=seq_lens,
    )

    print("=== Paged KV Cache Metadata (NHD Layout) ===")
    print(f"page_size={page_size}, indptr={paged_kv.indptr}, last_page_len={paged_kv.last_page_len}")
    print(f"indices={paged_kv.indices}")
    for b in range(paged_kv.batch_size):
        k_rec, v_rec = paged_kv.materialize_dense(b)
        assert k_rec == gt_k_all[b] and v_rec == gt_v_all[b], "Paged addressing mismatch"
        pages = paged_kv.indices[paged_kv.indptr[b] : paged_kv.indptr[b + 1]]
        last_tok = paged_kv.kv_len(b) - 1
        p_iter, phys_p, entry = paged_kv.locate(b, last_tok)
        off = paged_kv.elem_offset_nhd(phys_p, kv_head_idx=1, entry_idx=entry)
        print(
            f"req={b}: kv_len={paged_kv.kv_len(b):2d}, pages={pages}, "
            f"last_tok=(page_iter={p_iter}, phys_page={phys_p}, entry={entry}, head1_elem_offset={off})"
        )

    global_max_err = 0.0
    print("\n=== Online Softmax & Split-KV Decode (GQA 8Q:2KV, kv_chunk_size=6) ===")
    for b in range(paged_kv.batch_size):
        ref_out, ref_lse = dense_reference_gqa(q_batch[b], gt_k_all[b], gt_v_all[b], sm_scale)

        single_states = compute_paged_chunk_state(
            q_batch[b], paged_kv, b, 0, paged_kv.kv_len(b), sm_scale, tile_size=3
        )
        single_out = [st.normalized_output() for st in single_states]
        single_lse = [st.get_lse() for st in single_states]
        single_err = max(max_abs_diff_2d(single_out, ref_out), max_abs_diff_1d(single_lse, ref_lse))

        partials, fwd_out, fwd_lse = split_kv_decode(
            q_batch[b], paged_kv, b, sm_scale, kv_chunk_size, merge_order="forward"
        )
        _, rev_out, rev_lse = split_kv_decode(
            q_batch[b], paged_kv, b, sm_scale, kv_chunk_size, merge_order="reverse"
        )
        _, shuf_out, shuf_lse = split_kv_decode(
            q_batch[b], paged_kv, b, sm_scale, kv_chunk_size, merge_order="shuffled", rng=rng
        )

        # Verify raw unnormalized `state_t::merge` associativity
        raw_merged = AttentionState(head_dim)
        for _, _, chunk_states, _, _ in reversed(partials):
            raw_merged.merge_raw(chunk_states[0])
        raw_err_h0 = max(
            max_abs_diff_1d(raw_merged.normalized_output(), ref_out[0]),
            abs(raw_merged.get_lse() - ref_lse[0]),
        )

        fwd_err = max(max_abs_diff_2d(fwd_out, ref_out), max_abs_diff_1d(fwd_lse, ref_lse))
        rev_err = max(max_abs_diff_2d(rev_out, ref_out), max_abs_diff_1d(rev_lse, ref_lse))
        shuf_err = max(max_abs_diff_2d(shuf_out, ref_out), max_abs_diff_1d(shuf_lse, ref_lse))
        global_max_err = max(global_max_err, single_err, fwd_err, rev_err, shuf_err, raw_err_h0)

        assert single_err < 1e-12
        assert fwd_err < 1e-12 and rev_err < 1e-12 and shuf_err < 1e-12 and raw_err_h0 < 1e-12

        print(
            f"req={b} (kv_len={paged_kv.kv_len(b):2d}, chunks={len(partials)}): "
            f"single_err={single_err:.2e}, split_fwd={fwd_err:.2e}, "
            f"split_rev={rev_err:.2e}, split_shuf={shuf_err:.2e}, raw_merge_h0={raw_err_h0:.2e}"
        )
        if b == 1:
            print("  -- Intermediate Stage-1 Partial States for req=1, qo_head=0 --")
            print("  chunk_range | m_log2    | d        | lse_log2     | o_partial[0..2]")
            for c_start, c_end, chunk_states, v_chunk, s_chunk in partials:
                st0 = chunk_states[0]
                preview = ", ".join(f"{x:+.5f}" for x in v_chunk[0][:3])
                print(
                    f"  [{c_start:2d}, {c_end:2d})   | {st0.m_log2:+9.5f} | {st0.d:8.5f} | "
                    f"{s_chunk[0]:+12.6f} | [{preview}]"
                )
            merged_preview = ", ".join(f"{x:+.5f}" for x in fwd_out[0][:3])
            print(
                f"  merged_h0   | {raw_merged.m_log2:+9.5f} | {raw_merged.d:8.5f} | "
                f"{fwd_lse[0]:+12.6f} | [{merged_preview}]"
            )

    cas_err = run_cascade_shared_prefix_check(rng)
    global_max_err = max(global_max_err, cas_err)
    assert global_max_err < 1e-12
    print(f"\nMAX ABS ERROR ACROSS ALL CHECKS: {global_max_err:.3e} (< 1e-12)")
    print("SELF-CHECK: PASS")


if __name__ == "__main__":
    main()
