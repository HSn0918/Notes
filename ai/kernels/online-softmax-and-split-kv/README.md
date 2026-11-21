# 把 KV 切成四块，再合并一次注意力结果

取长度为 19 的请求，每 6 个位置分一块。脚本把四块分别计算，再按正序、逆序和乱序合并，逐项对照一次读完 KV 的结果。

## 运行与第一处观察

需要 Python 3.9+ 标准库。从仓库根目录运行：

```bash
python3 ai/kernels/online-softmax-and-split-kv/online_softmax_split_kv_lab.py
```

先核对末尾的 `SELF-CHECK: PASS`，再对照下面各项的误差。

## 从分页地址读到归约误差

- 3 条不等长序列（`kv_len` 分别为 `5`、`19`、`12`，`page_size=4`，包含未满尾页与满尾页）的物理页随机打乱存放，重建后与原始 Dense K/V 完全一致。
- 空状态（`-inf`）合并保护检查通过，不出现 `NaN`。
- `req=1`（`kv_len=19`）被切分为 4 个 KV chunk，打印每个 Stage-1 chunk 的中间状态表（`m_log2`、`d`、`lse_log2`、`o_partial[0..2]`）及 Stage-2 合并后的最终结果。
- 单遍 Online Softmax、正序/逆序/随机乱序 Split-KV 归约、未归一化 `merge_raw` 以及 3 层 Cascade 合并结果相对 Dense GQA 参考值的最大绝对误差均 `< 1e-12`。
- 末尾输出 `SELF-CHECK: PASS`。

## 各组实验验证的关系

1. `PagedKVCache` 的 `NHD` 布局与 CSR 分页寻址（`indptr`、`indices`、`last_page_len`）。
2. `AttentionState`（对应 `include/flashinfer/attention/state.cuh` 中的 `state_t`）采用以 $2$ 为底的对数缩放（`m_log2`、`d`、`o` 与 `2**(m_prev - m_new)`），并与双遍安全 Softmax GQA（8 QO Heads : 2 KV Heads）对齐。
3. **Split-KV Decode**：将单条请求的分页 KV 切分到多个模拟 CTA（`kv_chunk_size=6`），分别以正序、逆序、随机打乱顺序执行 Stage-2 `MergeStates` 归约。
4. **Cascade Shared-Prefix Merge**：在统一页池中构造 3 层前缀共享（L0 系统提示 + L1 少样本上下文 + L2 请求独有后缀），验证 `merge_state_in_place` 式的分层合并等价于完整拼接 KV 的全量计算。

## 两块注意力状态如何合并

在传统两遍 Softmax 中，必须先扫描全部 KV 得到全局最大值 $m$ 与分母 $d$，再回头计算概率加权和 $O$。FlashInfer 将状态压缩为三元组 $(m, d, o)$（其中 $m$ 乘以 $\log_2 e$ 以直接调用硬件快速指令 `ptx::exp2`），并通过结合律实现任意区间切分的无损合并：

$$
m_{\text{new}} = \max(m_1, m_2), \quad d_{\text{new}} = d_1 \cdot 2^{m_1 - m_{\text{new}}} + d_2 \cdot 2^{m_2 - m_{\text{new}}}
$$

$$
o_{\text{new}} = o_1 \cdot 2^{m_1 - m_{\text{new}}} + o_2 \cdot 2^{m_2 - m_{\text{new}}}
$$

当 Stage-1 Kernel 输出已经归一化的局部结果 $v_i = o_i / d_i$ 与以 $2$ 为底的对数和 $s_i = m_i + \log_2 d_i$ 时（即 FlashInfer `MergeStatesKernel` 与 `MergeStateInPlaceKernel` 的输入格式），Stage-2 合并只需令 `other_d = 1.0` 并调用同一 `state_t::merge(v_i, s_i, 1.0)` 即可精确恢复全局归约结果。

## 数值精度与适用环境

- Python 3.9+；仅使用 Python 标准库（`math`、`random`、`dataclasses`）。
- Mac 或 Linux CPU 均可运行，耗时 `< 1s`；不需要 PyTorch、CUDA、网络或 GPU。

- 本实验使用 Python IEEE-754 `float64` 标量运算，因此能稳定达到 `< 1e-12`（实测 `8.882e-16`）的误差上界；真实 GPU Kernel 中 Q/K/V 通常为 `FP16`/`BF16`/`FP8`/`NVFP4`，Tensor Core 矩阵乘（MMA/WGMMA）内部累加与存储转换会产生 `1e-3 ~ 1e-2` 量级的舍入差异。
- 当启用 Split-KV 时，若不固定切分粒度（对比 `BatchDecodeWithPagedKVCacheWrapper.plan()` 的 `fixed_split_size` / `disable_split_kv`），不同 Batch Size 下不同的归约树顺序在低精度浮点加法下会导致微小的非确定性（non-determinism），而本实验仅在 `float64` 下验证数学结合律与误差上界。
- 本脚本模拟逻辑索引与数值归约，不包含 `cp.async` 多级流水线、Shared Memory Swizzle、CUDA Graph 静态缓冲或 SM90 TMA 异步传输等硬件执行细节。

## 自检问题

### Q：为什么 Stage-1 输出已经除以 `d_i` 之后，Stage-2 仍然能用 `s_i = m_i + log2(d_i)` 无损还原原始权重和？

> [!question]- 参考答案（点击展开）
>
> 因为归一化输出满足 $v_i = o_i / d_i$，而 $\text{LSE}_2(s_i) = m_i + \log_2 d_i$ 等价于 $2^{s_i} = d_i \cdot 2^{m_i}$。当以新最大值 $s_{\text{new}}$ 做缩放时，权重因子 $2^{s_i - s_{\text{new}}} \cdot v_i = 2^{-s_{\text{new}}} \cdot (d_i 2^{m_i}) \cdot (o_i / d_i) = 2^{m_i - s_{\text{new}}} \cdot o_i$，分子分母中的 $d_i$ 恰好抵消。因此 `cascade.cuh` 中 `MergeStatesKernel` 直接传入 `st.merge(v, s, 1)` 即可同时完成重缩放与累加。

### Q：在 FlashInfer `state_t::merge()` 中，为什么不能直接写 `ptx_exp2(m_prev - m)` 而不加 `-math::inf` 截断？

> [!question]- 参考答案（点击展开）
>
> 当某个 KV chunk 完全被 Mask 遮蔽或为空时，对应状态保持初始值 `m = -inf`。如果两个空状态合并，`m_prev - m` 会算出 `(-inf) - (-inf) = NaN`，进而污染整个 `d` 与 `o`。`include/flashinfer/attention/state.cuh` 通过 `max(m_prev - m, -math::inf)`（利用 CUDA `max` 对 NaN 的处理）以及对 `d_sum == 0` 的保护，把空状态缩放系数安全压成 `0`，本实验在 `check_empty_state_nan_guard` 中同样验证了这一边界条件。

### Q：Cascade Attention 的 3 层合并顺序改变时，数学上为什么仍然等价？

> [!question]- 参考答案（点击展开）
>
> Attention 的分母求和操作与分子加权求和操作在实数域上满足交换律与结合律，而将状态提升为三元组 $(m, d, o)$（或归一化后的 $(v, s)$）后，其合并算子 $\oplus$ 依然是结合且可交换的幺半群运算（单位元为 $(-\infty, 0, \vec{0})$）。因此无论像 `MultiLevelCascadeAttentionWrapper.run()` 那样先算底层后缀再依次 `merge_state_in_place` 上层公共前缀，还是先合并前缀再合并后缀，精确数学结果完全一致。

## 验证记录

- 验证日期与环境：`2026-11-17`，Mac CPU，Python 3.14.7（通过 `/tmp` 副本运行验证）。
- 实测关键输出摘要：

```text
=== Paged KV Cache Metadata (NHD Layout) ===
page_size=4, indptr=[0, 2, 7, 10], last_page_len=[1, 3, 4]
indices=[7, 10, 4, 5, 1, 6, 11, 8, 9, 3]
req=0: kv_len= 5, pages=[7, 10], last_tok=(page_iter=1, phys_page=10, entry=0, head1_elem_offset=486)
req=1: kv_len=19, pages=[4, 5, 1, 6, 11], last_tok=(page_iter=6, phys_page=11, entry=2, head1_elem_offset=558)
req=2: kv_len=12, pages=[8, 9, 3], last_tok=(page_iter=9, phys_page=3, entry=3, head1_elem_offset=186)

=== Online Softmax & Split-KV Decode (GQA 8Q:2KV, kv_chunk_size=6) ===
req=0 (kv_len= 5, chunks=1): single_err=4.44e-16, split_fwd=4.44e-16, split_rev=4.44e-16, split_shuf=4.44e-16, raw_merge_h0=1.67e-16
req=1 (kv_len=19, chunks=4): single_err=8.88e-16, split_fwd=8.88e-16, split_rev=8.88e-16, split_shuf=8.88e-16, raw_merge_h0=8.88e-16
  -- Intermediate Stage-1 Partial States for req=1, qo_head=0 --
  chunk_range | m_log2    | d        | lse_log2     | o_partial[0..2]
  [ 0,  6)   |  +0.50197 |  5.26944 |    +2.899616 | [+0.36666, +0.27121, +0.12871]
  [ 6, 12)   |  +0.54656 |  4.81916 |    +2.815346 | [+0.18229, +0.18334, +0.37059]
  [12, 18)   |  +0.47979 |  4.65683 |    +2.699135 | [+0.04549, +0.03679, +0.16132]
  [18, 19)   |  +0.10250 |  1.00000 |    +0.102501 | [+0.61992, +0.84759, -0.82273]
  merged_h0   |  +0.54656 | 15.10946 |    +4.463944 | [+0.22567, +0.20224, +0.16917]
req=2 (kv_len=12, chunks=2): single_err=8.88e-16, split_fwd=8.88e-16, split_rev=8.88e-16, split_shuf=8.88e-16, raw_merge_h0=4.44e-16

=== Cascade Shared-Prefix Merge (3 Levels) ===
req=0 total_kv=18 (L0=7, L1=5, L2=6) | L0_lse_h0=+3.1322 L2_lse_h0=+2.4288 | max_out_err=8.327e-17 | max_lse_err=8.882e-16
req=1 total_kv=15 (L0=7, L1=5, L2=3) | L0_lse_h0=+3.0151 L2_lse_h0=+1.6571 | max_out_err=1.388e-16 | max_lse_err=8.882e-16

MAX ABS ERROR ACROSS ALL CHECKS: 8.882e-16 (< 1e-12)
SELF-CHECK: PASS
```

[FlashInfer 分页 Decode 源码](../flashinfer-paged-kv-and-decode-source.md) · [返回算子索引](../README.md)
