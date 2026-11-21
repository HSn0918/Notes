#ai #llm #线性代数 #数学基础 #simd #cuda

相关笔记：[[advanced-algebra-for-llm]] | [[llm-fundamentals]] | [[llm-learning-path]] | [[simd-and-gpu-simt-foundations]] | [[gpu-execution-and-memory-model]] | [LLM 数学直觉与 SIMD 并行离线实验](llm-math-and-simd-lab/README.md)

# 面向大模型的线性代数：从几何直觉到张量与分块计算

## 概述

很多工程背景的学习者一准备深入 FlashAttention / FlashInfer / vLLM 底层，就想先把数学系《线性代数》整本啃完，结果卡在行列式展开、伴随矩阵求逆和若尔当标准型里半途而废。事实上，**现代大模型几乎从不计算行列式，也从不显式求大矩阵的逆**；真正反复用到的只有四件高度聚焦的武器：

| 模块 | 在 LLM 里对应什么 | 在硬件上对应什么 | 配套实验 |
| --- | --- | --- | --- |
| 点积 = 投影 | Attention 打分 $QK^T/\sqrt{d}$、RMSNorm | SIMT 归约、Warp Shuffle | Exp 5（SIMD 点积） |
| 矩阵乘的四种视角 | QKV / MLP 投影、GEMM | Tensor Core MMA、rank-1 更新 | Exp 1（外积视角） |
| 分块矩阵乘 Tiling | FlashAttention / Block GEMM | SRAM 缓存 Tile、寄存器累加器 | Exp 1（分块等价） |
| Shape / Stride | NHD / HND KV Cache 布局 | 128-bit 合并访存、零拷贝转置 | Exp 5（Stride 视图） |

本笔记全部采用"白话比喻 + $2 \times 2$ 小数字手算 + 算子代码映射"的方式，跳过抽象空间的公理化定义，直接建立能支撑后续 [[simd-and-gpu-simt-foundations]]、[[gpu-execution-and-memory-model]] 与 FlashInfer 源码阅读的工程直觉。

---

## 一、点积：投影、相似度与方差缩放

### 几何直觉：点积在问"你对齐了多少"

两个向量的点积 $\langle q, k \rangle = \sum_i q_i k_i$ 的几何意义是：**把 $q$ 投影到 $k$ 方向上，投影长度乘以 $k$ 的长度**。方向越一致，点积越大；垂直则为 0；相反为负。

小数字手算：设 $q = [1, 2]$，$k = [2, 1]$，则 $q \cdot k = 1 \times 2 + 2 \times 1 = 4$；若 $k' = [-2, 1]$，则 $q \cdot k' = -2 + 2 = 0$（正交，互不相关）。

Attention 里 $s_j = \frac{q \cdot k_j}{\sqrt{d}}$ 的本质就是：**当前 Token 用自己的 Query 向量，去和历史每个 Token 的 Key 向量做一次"对齐程度打分"**。打分越高，后面加权求和时该 Token 的 Value 占比越大。

### RMSNorm：先把"音量"归一，再谈方向

> 基础名词：**RMSNorm** 是 $\text{RMSNorm}(x) = x / \sqrt{\frac{1}{d}\sum_i x_i^2}$（可能再乘一个可学习的逐维增益 $\gamma$）。它只缩放向量的整体"音量"（模长），不改变方向。

为什么 LLM（Llama、Qwen、DeepSeek 等）普遍用它而不是 BatchNorm / LayerNorm？因为 Transformer 的 Hidden State 模长会随层数漂移，先统一音量、再让注意力专注方向对齐，深层网络才稳定。它只需要一次平方和归约，没有均值/方差两套统计量，对 GPU 的 Warp 归约极其友好（FlashInfer `norm.cuh` 的 `RMSNormKernel`，源码导读见 [[flashinfer-norm-and-sampling-source]]）。

### 为什么除以 $\sqrt{d}$：方差缩放

设 $q, k$ 的各维分量独立、均值 0、且经过 RMSNorm 后满足 $\mathbb{E}[q_i^2] = 1$，则：

$$\mathrm{Var}(q \cdot k) = \sum_{i=1}^{d} \mathrm{Var}(q_i k_i) = d$$

也就是说，**维度越高，点积的波动范围越大**（标准差 $\sqrt{d}$）。Softmax 对输入尺度极其敏感：分数普遍偏大时分布会变得极尖（几乎 one-hot），梯度消失。除以 $\sqrt{d}$ 把方差重新拉回 1，等于把 Softmax 的"温度"稳定住。

这就是 $\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d})V$ 里那个 $\sqrt{d}$ 的全部来源——**它不是深奥设计，只是一个方差归一化**。

---

## 二、矩阵乘法的四种黄金视角

设 $A \in \mathbb{R}^{m \times k}$，$B \in \mathbb{R}^{k \times n}$。用同一个 $2 \times 3$ 乘 $3 \times 2$ 的例子贯穿：

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \quad B = \begin{bmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{bmatrix} \implies C = AB = \begin{bmatrix} 58 & 64 \\ 139 & 154 \end{bmatrix}$$

### 视角 1：行 × 列内积（教科书定义）

$$C[i][j] = \langle A[i,:],\, B[:,j] \rangle$$

手算：$C[0][0] = 1 \cdot 7 + 2 \cdot 9 + 3 \cdot 11 = 58$。

这是定义，但对 GPU 却是最差的遍历方式之一：每算一个元素都要跳着读 $B$ 的一列（最内维 stride 大，破坏合并访存，见第五节）。它适合当**参考实现**（实验里的 `matmul_dense`），不适合当主力 Kernel。

### 视角 2：行 × 矩阵（输出行 = B 行向量的线性组合）

$$C[i,:] = A[i][0]\,B[0,:] + A[i][1]\,B[1,:] + \cdots + A[i][k{-}1]\,B[k{-}1,:]$$

手算：$C[0,:] = 1\cdot[7,8] + 2\cdot[9,10] + 3\cdot[11,12] = [58,64]$。✓

这个视角下，输出的一整行是 $B$ 的**连续行**按系数加权求和——内存访问是连续的。它对应"把 $B$ 当成 $k$ 个 $n$ 维基向量，输出行是坐标 $A[i,:]$ 下的组合"。

### 视角 3：矩阵 × 列（输出列 = A 列向量的线性组合）

$$C[:,j] = B[0][j]\,A[:,0] + B[1][j]\,A[:,1] + \cdots$$

手算：$C[:,0] = 7\cdot[1,4] + 9\cdot[2,5] + 11\cdot[3,6] = [58,139]$。✓

### 视角 4：列 × 行外积之和（rank-1 展开）

$$C = \sum_{p=0}^{k-1} A[:,p] \otimes B[p,:]$$

手算：

```text
outer([1,4],[7,8])  = [[ 7,  8],[28, 32]]
outer([2,5],[9,10]) = [[18, 20],[45, 50]]
outer([3,6],[11,12])= [[33, 36],[66, 72]]
求和                 = [[58, 64],[139,154]]  ✓
```

这是理解高性能计算最重要的一视角：

- **Tensor Core 的 MMA 指令**本质就是"一次吃一小块 A 切片 × B 切片，累加进寄存器里的 C Tile"——正是视角 4 的硬件化。
- **Decode 阶段的 GEMV**（$M=1$ 的矩阵向量乘）就是"每次读一行权重，与 $x$ 做外积累加"。
- **LoRA 的 $\Delta W = BA$** 与 [[advanced-algebra-for-llm]] 中的 SVD 低秩展开，都是"少数几个 rank-1 外积之和"。

四种视角数学上严格等价（都是同一个 $C$），但**对应完全不同的内存遍历顺序与硬件效率**——选哪个视角写 Kernel，决定了你离 Roofline 屋顶有多远。

---

## 三、分块矩阵乘（Block GEMM / Tiling）

### 为什么切小块与全量相乘严格等价

矩阵乘的全部运算就是 $m \times n \times k$ 次乘法与加法，而**加法满足结合律、乘法对加法满足分配律**，所以：

- 沿 $K$ 维任意切片求和，顺序无所谓；
- 沿 $M$ / $N$ 维切块，各块互不重叠，各算各的。

小数字手算（沿 $K$ 切成 $2+1$ 两段）：

```text
先累加前 2 列的外积： outer([1,4],[7,8]) + outer([2,5],[9,10])
                    = [[7+18, 8+20],[28+45, 32+50]] = [[25, 28],[73, 82]]
再累加最后 1 列：     + outer([3,6],[11,12]) = [[25+33, 28+36],[73+66, 82+72]]
                    = [[58, 64],[139,154]]  ✓ 与全量一致
```

这就是 [[gpu-execution-and-memory-model]] 中 Online Softmax 分块状态合并 $\oplus$ 能成立的最朴素版本：**只要"合并算子"满足结合律，任意分块、任意顺序都等价**。

### Tiling 的硬件动机：把"远水"变"近水"

GPU 的存储金字塔里，HBM（几十 GB、约数 TB/s）远而慢，SRAM/Shared Memory（每 SM 约 100–228 KB、极高带宽）近而小。朴素视角 1 的问题在于：**每算一个 $C$ 元素都要从 HBM 重新读一遍 $B$ 的一列，权重数据被重复读取了 $m$ 次**。

分块 Tiling 的做法：

1. 把 $C$ 的一个 Tile（如 $64 \times 64$）留在**寄存器**里当累加器；
2. 把对应的 $A$、$B$ 小块通过 `cp.async` 流水搬进 **SRAM**；
3. 在片上把该 Tile 累加完，最后一次性写回 HBM。

这样每份 $A$、$B$ 数据从 HBM 读一次，就能被 $M$ / $N$ 方向上的一整批 Tile 复用，算术强度（FLOP/Byte）大致随 Tile 边长线性上升——这正是 Prefill GEMM 能冲进 Compute-bound 屋顶平坦区的原因（定量推导见 [[gpu-execution-and-memory-model]]）。

### 与真实算子的映射

- 实验脚本（[LLM 数学直觉与 SIMD 并行离线实验](llm-math-and-simd-lab/README.md)）Exp 1 用故意**非整除**的形状 `29×37 @ 37×31`、`tile=8×8×8` 验证分块与全量的最大误差 `< 1e-12`，并统计 SRAM Tile 装载次数（80 次）——尾块（tail tile）处理是真实 Kernel 里最容易写错边界的地方。
- FlashAttention / FlashInfer 的 Prefill Kernel 就是"分块 GEMM + Online Softmax"的组合：外层沿 $S_q$ / $S_{kv}$ 双重分块，内层用视角 4 在 SRAM 里完成 $QK^T$ 与 $PV$，全程不让 $N \times N$ 注意力矩阵落盘 HBM（实验 Exp 1 的 `attention_tiled_online_softmax` 复现了这一结构）。

---

## 四、QKV 与 MLP 的投影几何

### Linear 层：换一组坐标轴看同一个向量

一个 `nn.Linear(in, out)` 存的是矩阵 $W \in \mathbb{R}^{out \times in}$（加 bias）。$y = Wx$ 的几何意义是：**把 $x$ 用一组新基（$W$ 的行向量）重新表达，得到它在 $out$ 维空间里的坐标**。

- $W$ 的每个行向量是一根"探测器"的方向；$y_j = \langle w_j, x \rangle + b_j$ 就是 $x$ 在第 $j$ 根探测器上的投影响应。
- 小例子：$x = [1, 0]$，$W = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$，则 $y = [2, 0]$——只是沿各轴被缩放；若 $W$ 是旋转矩阵（见 [[advanced-algebra-for-llm]] 的 RoPE），则只转方向不改长度。

### QKV：同一份 Hidden State 的三种"提问角度"

每个 Token 的 Hidden State 经过三个独立矩阵 $W_Q, W_K, W_V$ 投影成 Q/K/V：

- **Query**：本 Token 在问什么（探测器方向）；
- **Key**：历史每个 Token 能被怎样检索（索引方向）；
- **Value**：检索命中后真正交付的信息内容。

三套投影刻意**不共享矩阵**，因为"怎么找"和"找到后给什么"本来就是两种不同的变换。多头注意力再把 $d_{model}$ 切成 $H$ 份，每份 $d_{head} = d_{model}/H$ 在**各自的低维子空间**里独立做注意力——这正是"秩与低维子空间"直觉的第一次登场（详见 [[advanced-algebra-for-llm]]）。

### MLP：升维 → 非线性筛选 → 投回

MLP 先把向量投影到 $4 \times d_{model}$ 的高维空间，过激活函数（如 SwiGLU），再投回来。直觉：**高维空间里有更多"互不干扰的方向"，非线性筛选相当于在每个方向上做门控**，让网络能表达比单层线性更复杂的函数。

---

## 五、Shape / Stride：张量的内存视图

### 形状是逻辑的，步长才是物理的

> 基础名词：**Stride（步长）** 是"沿某一维前进 1 个逻辑索引，物理地址跳过多少个元素"。访问 `tensor[i_0, i_1, ..., i_n]` 的物理偏移为 $\sum_j i_j \times stride_j$。

关键推论：**形状变了不一定要动数据**。只要改 `shape` 和 `stride` 这两组元数据，同一个一维 buffer 可以被解释成任何布局——这就是 PyTorch `.view()` / `.transpose()` 零拷贝的原理。

实验 Exp 5 用 `[seq=16, heads=4, dim=8]` 的小 buffer 验证了三种视图共享**同一份存储**（`storage is hnd_view.storage`，误差为 0）：

| 视图 | 逻辑形状 | Stride | 最内维特征 |
| --- | --- | --- | --- |
| NHD | `(16, 4, 8)` | `(32, 8, 1)` | 同一 Token 同一 Head 的 $d$ 维连续 → SIMD/128-bit 友好 |
| HND | `(4, 16, 8)` | `(8, 32, 1)` | 同一 Head 的相邻 Token 跳 32 元素 |
| DHN | `(8, 4, 16)` | `(1, 8, 32)` | 最内维跨 32 元素 → strided gather，访存惩罚 |

### 为什么布局直接决定性能

CPU SIMD 与 GPU 都靠"**一批车道同时搬运连续地址**"吃饭（详见 [[simd-and-gpu-simt-foundations]]）：

- 最内维 `stride = 1`：一个 Warp 的 32 个线程读连续地址，内存控制器合并成最少的缓存行事务，`vec_t<T, vec_size>` 还能单指令搬 128-bit（16 字节，如 FP16 一口气读 8 个元素）。
- 最内维 `stride > 1`：每次请求散落在不同缓存行，有效带宽可能暴跌一个数量级。

所以"布局选择"从来不是小事：

- FlashInfer `paged_kv_t` 提供 **NHD**（与线性投影输出天然对齐，免转置）与 **HND**（同一 Head 的连续 Token 紧凑，对低比特量化与部分 Tensor Core 路径友好）两种布局，Kernel 借 stride 公式统一寻址——细节见 [[flashinfer-paged-kv-and-decode-source]]。
- 实验里 NHD → HND 的转置**零字节拷贝**（只交换 `shape`/`stride`），而如果真要 DHN 布局喂给 Kernel，就会吃 strided gather 惩罚——这也是为什么不少引擎宁可专门写一个 Kernel，也不随手 `transpose()`。

### 二维回顾：行主序与列主序

矩阵 `M` 形状 `[1024, 1024]` 行主序时 `stride = (1024, 1)`：沿列方向走是连续的（`stride=1`），沿行方向走每次跳 1024。C/C++/PyTorch 默认行主序，Fortran/MATLAB 默认列主序——同一段"连续访存"代码换个生态可能就从最快变成最慢，根源只在 stride。

---

## 自检清单

- [ ] 能用"投影"解释点积，并说出 RMSNorm 为什么只缩放模长不改方向。
- [ ] 能推导 $QK^T/\sqrt{d}$ 中 $\sqrt{d}$ 来自点积方差 $\approx d$。
- [ ] 能不查资料写出矩阵乘四种视角的公式，并说出各自对应的硬件访问模式。
- [ ] 能解释分块 GEMM 为什么严格等价（加法结合律 + 分配律），以及 Tiling 如何提升算术强度。
- [ ] 能写出 `offset = Σ idx × stride`，并解释零拷贝转置为什么不动数据。
- [ ] 能说出 NHD 与 HND 布局各自的访存特征与适用场景。

## 面试要点

### Q：为什么 Attention 打分要除以 $\sqrt{d_k}$，不除会发生什么？

> [!question]- 参考答案（点击展开）
>
> 设 Q/K 各维独立同分布、均值 0，则点积的方差随维度线性增长（$\mathrm{Var}(q \cdot k) \approx d_k$）。不除以 $\sqrt{d_k}$ 时，高维下分数普遍偏大，Softmax 输出趋近 one-hot：一方面前向梯度过小（几乎只剩最大项有梯度），另一方面分布过尖使采样失去多样性。除以 $\sqrt{d_k}$ 把方差重新归一到 1，等价于固定 Softmax 的温度。

### Q：矩阵乘 $C = AB$ 有哪几种等价视角？各自适合什么硬件场景？

> [!question]- 参考答案（点击展开）
>
> 1. 行×列内积：教科书定义，适合做参考实现，但列遍历破坏合并访存。
> 2. 行×矩阵：输出行是 B 行向量的线性组合，行遍历连续，适合流式逐行计算。
> 3. 矩阵×列：输出列是 A 列向量的线性组合，与 GEMV（Decode 小批量）对应。
> 4. 列×行外积之和：沿 K 维累加 rank-1 外积，是 Tensor Core MMA 分块累加、LoRA 低秩展开与 SVD 的共同骨架。
>
> 四者数学等价，差别只在内存遍历顺序与数据复用方式。

### Q：分块矩阵乘（Tiling）为什么能大幅提升 GEMM 性能？正确性靠什么保证？

> [!question]- 参考答案（点击展开）
>
> 性能：把 C 的一个 Tile 常驻寄存器、A/B 小块搬进 SRAM 后在片上复用，使每份权重数据只需从 HBM 读一次即可服务 $M$/$N$ 方向一整批 Tile，算术强度随 Tile 边长近似线性上升，从而把 GEMM 从 HBM 带宽受限推向计算受限。
>
> 正确性：矩阵乘的累加只依赖加法结合律与分配律，沿 K 维任意切片、沿 M/N 维任意切块都与全量相乘严格等价（实验 Exp 1 在 29×37×31 的非整除形状上验证误差 < 1e-12）。

### Q：`tensor.transpose()` 在 PyTorch 里为什么是 O(1) 的？它什么时候会变成性能陷阱？

> [!question]- 参考答案（点击展开）
>
> 转置只交换 `shape` 与 `stride` 两组元数据，底层 storage 一个字节都不动，所以是 O(1) 视图操作。陷阱在于转置后最内维 stride 不再为 1：后续 Kernel 若直接消费该视图，访存退化为 strided gather，有效带宽大幅下降；因此高性能路径要么选一个本来就匹配目标布局的 Kernel（如 FlashInfer 的 NHD/HND 双布局），要么显式做一次 `.contiguous()` 拷贝换取后续合并访存。

## 参考与版本边界

- 本文为数学直觉笔记，不绑定具体库版本；涉及的算子名（`RMSNormKernel`、`vec_t`、`paged_kv_t` 的 NHD/HND）以 [[flashinfer-norm-and-sampling-source]] 与 [[flashinfer-paged-kv-and-decode-source]] 中的版本边界（FlashInfer v0.7.3, commit `82caa5668`）为准。
- 配套可运行验证：[LLM 数学直觉与 SIMD 并行离线实验](llm-math-and-simd-lab/README.md)（Exp 1 分块 GEMM 与分块 Attention、Exp 5 SIMD 点积与 Stride 视图）。
- 深入方向：结合律/正交旋转/特征谱/SVD 见 [[advanced-algebra-for-llm]]；从 SIMD 到 SIMT 的硬件映射见 [[simd-and-gpu-simt-foundations]]；Roofline 定量分析见 [[gpu-execution-and-memory-model]]。
