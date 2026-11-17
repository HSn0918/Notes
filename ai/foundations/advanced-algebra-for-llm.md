#ai #llm #高等代数 #数学基础 #rope #svd #lora #mla

相关笔记：[[linear-algebra-for-llm]] | [[lora-and-qlora]] | [[llm-post-training]] | [[flashinfer-ragged-prefill-and-mla-source]] | [[sglang-srt-radix-and-scheduler]] | [[simd-and-gpu-simt-foundations]] | [LLM 数学直觉与 SIMD 并行离线实验](../experiments/llm-math-and-simd-lab/README.md)

# 面向大模型的高等代数：子空间、正交旋转、特征谱与低秩分解

## 概述

如果说 [[linear-algebra-for-llm]] 解决的是"矩阵乘怎么算、张量怎么放"，本笔记回答的是另一类问题：**为什么大模型里的几个关键设计（MLA 权重吸收、RoPE、残差连接、LoRA）在数学上是成立的**。它们分别对应高等代数的四件武器：

| 模块 | 一句话 | 在 LLM 里对应什么 | 配套实验 |
| --- | --- | --- | --- |
| 秩与低维子空间 | 高维数据常活在低维子空间里 | 多头注意力的头维度切分、MLA 压缩 KV | Exp 4（能量谱） |
| 结合律 $(AB)x = A(Bx)$ | 结果不变，成本天差地别 | DeepSeek MLA 权重吸收省 KV 带宽 | Exp 3（吸收等价 + 字节审计） |
| 正交旋转 | 只转方向，不改长度与内积 | RoPE 相对位置编码 5 分钟手推 | Exp 2（旋转不变量） |
| 特征谱与条件数 | 每层放大多少倍决定网络死活 | 残差连接与 Pre-Norm 的必要性 | — |
| SVD 低秩近似 | 最优的"少参数重建" | LoRA / QLoRA、KV 低秩压缩 | Exp 4（Eckart–Young） |

与上一篇相同，全部用"白话比喻 + 小数字手算 + 实验验证"推进，不引入抽象空间公理。

---

## 一、秩与低维子空间：大模型相信"有效信息是低维的"

### 秩的直觉

矩阵的秩是它包含的**独立方向数**。一个 $4096 \times 4096$ 的矩阵理论上有 4096 个独立方向，但真实训练出来的权重矩阵、注意力头、乃至激活流形的"有效秩"往往远小于名义维度——大量方向几乎是冗余或噪声。

小例子：$3 \times 3$ 矩阵 $\begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 3 & 6 & 9 \end{bmatrix}$ 三行全是第一行的倍数，秩为 1——9 个数字其实只由 3 个独立数字生成。

### 在 LLM 里的两处落地

1. **多头注意力本身就是"低维子空间假设"**：$d_{model}=4096$ 被切成 32 个 $d_{head}=128$ 的子空间，每个头相信"在某个 128 维子空间里单独做注意力就能捕捉一种关系模式"。如果信息真的均匀铺满 4096 维，这个设计就不会有效。
2. **DeepSeek MLA 把它推到极致**：所有 Head 的 K/V 不再各自独立缓存，而是每个 Token 只缓存一个 512 维（+64 维 RoPE）的**共享低秩潜在向量** $c_{kv}$，各 Head 的 K/V 由 $c_{kv}$ 经各自的投影矩阵解压生成——本质上是押注"128 个 Head 的 K/V 联合分布在远低于 $128 \times 128$ 维的子空间里"。

这套押注省下多少带宽，正是下一节结合律要回答的问题。

---

## 二、矩阵乘法结合律与 DeepSeek MLA 权重吸收

### 结合律：结果不变，成本天差地别

矩阵乘法满足结合律：$(AB)x = A(Bx)$，但两种求值顺序的计算量不同：

- **先乘 $AB$ 再乘 $x$**：构成 $AB$ 需要 $O(k^3)$（这里 $2^3=8$ 次乘法），再乘 $x$ 又要 $O(k^2)$；
- **先乘 $Bx$ 再乘 $A$**：两次矩阵向量乘，共 $O(k^2)$ 即可（$4+4=8$ 次乘法）。

当 $x$ 是一个向量（Decode 阶段每步每请求只来 1 个 Query）时，**"把矩阵吸收进向量一侧、推迟另一个矩阵"几乎总是更便宜**。

### MLA 的吸收推导（Decode 场景）

DeepSeek MLA 中，每 Token 缓存的是压缩 latent $c_t \in \mathbb{R}^{512}$（各 Head 共享）与解耦 RoPE 键 $k^{pe}_t \in \mathbb{R}^{64}$，Head $h$ 的未压缩 K/V 由各自的 $W^{UK}_h \in \mathbb{R}^{512 \times 128}$、$W^{UV}_h$ 从 $c_t$ 解压。

**不吸收路径**（先把 K/V 解压出来）：

$$s_t = \big(q^{nope}_h\big)^{\top} \underbrace{\big(W^{UK}_h c_t\big)}_{K_t} + \big(q^{pe}_h\big)^{\top} k^{pe}_t, \qquad o_h = \sum_t p_t \underbrace{\big(W^{UV}_h c_t\big)}_{V_t}$$

问题：每个 Head、每个历史 Token 都要做一次 $512 \times 128$ 的矩阵向量乘来解压——Decode 每步要扫全部 $T$ 个历史 Token，等于**每步把解压矩阵乘 $T \times H$ 遍**，而且解压后的 K/V 会把缓存带宽打回未压缩状态。

**吸收路径**（用结合律换序）：

$$s_t = \big(\underbrace{(W^{UK}_h)^{\top} q^{nope}_h}_{\tilde{q}_h \in \mathbb{R}^{512}}\big)^{\top} c_t + \big(q^{pe}_h\big)^{\top} k^{pe}_t$$

$$o_h = \sum_t p_t \big(W^{UV}_h c_t\big) = W^{UV}_h \underbrace{\Big(\sum_t p_t\, c_t\Big)}_{\tilde{o}_h \in \mathbb{R}^{512}}$$

- Query 侧：每步每 Head 只做**一次** $W^{UK\top}_h q$ 得到 $\tilde{q}_h$，之后直接对 512 维的 $c_t$ 做点积；
- Value 侧：先在 latent 空间完成加权求和 $\tilde{o}_h$，**最后乘一次** $W^{UV}_h$——因为 $W^{UV}$ 是线性的，可以穿过求和号。

数学上与不吸收路径**零误差等价**（实验 Exp 3 在 $H=8, T=24$ 下验证 `score_err=out_err=5.551e-17`），但 HBM 每步要读的字节天差地别（DeepSeek-V3 量级，$H=128$、$T=8192$、BF16）：

| 缓存形态 | 每 Token 字节 | Decode 每步总读取 |
| --- | --- | --- |
| 全量 MHA KV（$128 \times (128+64+128)$ 维） | 81,920 B | 640 MiB |
| 不吸收的解压缓存 | 65,664 B | 513 MiB |
| MLA 吸收后 latent | **1,152 B** | **9 MiB（71× 更小）** |

这就是 [[advanced-algebra-for-llm]] 标题里"结合律算力魔法"的含义：**一行代数换序，把 Decode Attention 的带宽需求砍了 71 倍**。

### 为什么 Prefill 不急着吸收

Prefill 一次处理成百上千个 Query Token，算术强度本来就高（Compute-bound，见 [[gpu-execution-and-memory-model]] 的 Roofline 定量），解压 K/V 一次就够整个 Tile 复用；吸收主要拯救的是**每步都要重扫全部 KV 的 Decode**。FlashInfer 的 MLA Kernel 因此同时提供两条路径（见 [[flashinfer-ragged-prefill-and-mla-source]]）；SGLang 在 Decode 阶段启用吸收并调用 `BatchMLAPagedAttentionWrapper`（见 [[sglang-srt-radix-and-scheduler]]）。

---

## 三、正交旋转与 RoPE：5 分钟手推相对位置编码

### 正交矩阵：只旋转，不拉伸

> 基础名词：**正交矩阵** $R$ 满足 $R^{\top} R = I$。几何上它只做旋转/反射，**保持所有向量长度与任意两向量夹角（内积）不变**。

二维旋转块：

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}, \qquad R(90^\circ)\begin{bmatrix}1\\0\end{bmatrix} = \begin{bmatrix}0\\1\end{bmatrix}$$

把 $x$ 转 90° 得到 $[0,1]$，长度不变——这就是全部直觉。

### RoPE：每个特征对一根独立的"时钟指针"

RoPE 把 $d$ 维向量切成 $d/2$ 个二维对，第 $i$ 对用频率 $\theta_i = \text{base}^{-2i/d}$（base 常取 10000）旋转 $pos \cdot \theta_i$ 弧度：

$$\begin{bmatrix} x'_{2i} \\ x'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(pos\,\theta_i) & -\sin(pos\,\theta_i) \\ \sin(pos\,\theta_i) & \cos(pos\,\theta_i) \end{bmatrix} \begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}$$

直觉：**每个二维对是一根钟表指针，位置越靠后指针转得越多；$\theta_i$ 随 $i$ 衰减，低维指针转得快（像秒针，分辨精细位置），高维指针转得慢（像时针，编码长程周期）**。

### 相对位置不变性（核心性质）

对 Query 在位置 $m$、Key 在位置 $n$ 的打分：

$$\langle R_m q,\ R_n k \rangle = q^{\top} R_m^{\top} R_n\, k$$

关键在于同一频率的旋转矩阵可交换，且 $R_m^{\top} R_n = R_{n-m}$（转 3° 再转回 5° 等价于净转 2°），于是：

$$\langle R_m q,\ R_n k \rangle = q^{\top} R_{n-m}\, k = \langle q,\ R_{n-m} k \rangle$$

**打分只取决于相对距离 $n - m$，与绝对位置无关**。注意力天然需要平移不变性，RoPE 用"旋转 + 内积"零参数地实现了它。实验 Exp 2 验证了三件事：$R^{\top}R = I$（误差 `2.22e-16`）、长度保持（误差 `0`）、以及 5 组 $(m,n)$ 配对下 $\langle R_m q, R_n k\rangle = \langle q, R_{n-m} k\rangle$ 与"两边同平移 $\delta$ 打分不变"（误差 `< 1e-13`）。

### 工程落地

实际 Kernel **从不构造 $d \times d$ 大矩阵**，而是逐对做 O(d) 的旋转（实验里的 `apply_rope_fast`）；FlashInfer 甚至在 Decode Kernel 加载 K Tile 时**在线融合** RoPE 旋转，避免单独启动一次 Kernel 多读写一遍 HBM（`decode.cuh` 的 `PosEncodingMode::kRoPELlama` 路径，见 [[flashinfer-paged-kv-and-decode-source]]）。MLA 的解耦 RoPE 维度（64 维 $k^{pe}$）之所以单独缓存，正是因为 RoPE 的旋转**不能**被权重吸收（它是逐位置非线性映射），与第二节的吸收路径互斥共存。

---

## 四、特征值谱与条件数：百层网络为什么能活

### 谱半径：每层的"放大倍数"

把每层近似看作线性映射 $y = Wx$，则向量模长最多被放大 $\sigma_{\max}(W)$ 倍（最大奇异值，谱范数）。100 层堆叠后：

- $\sigma_{\max} > 1$：$1.1^{100} \approx 13780$——激活与梯度**指数爆炸**；
- $\sigma_{\max} < 1$：$0.9^{100} \approx 2.7 \times 10^{-5}$——信息与梯度**指数消失**。

纯乘法链条对每层增益极其敏感，这就是深网络 without 机制辅助训不动的本质。

### 残差连接：把乘法变加法

残差把 $y = f(x)$ 改成 $y = x + f(x)$：即使 $f$ 的增益略小于 1，主体信息仍以"加法"原样向前传递，梯度也可以沿恒等分支无衰减回传。深网络从"100 次连乘"变成"100 次加法微扰"，才有了百层可训练的资格。

### Pre-Norm：把每层的扰动约束在单位尺度

Pre-Norm（先 Norm 再进子层，$x_{l+1} = x_l + f(\text{Norm}(x_l))$）保证**进入每层的输入尺度有界**，于是每层注入的扰动幅度有界、堆叠 $L$ 层只线性增长而不是随层数乘法放大。这也是现代 LLM（Llama 系、DeepSeek 系）普遍选择 RMSNorm + Pre-Norm + 残差的组合原因——三者共同把谱半径问题从"指数"降级为"有界累加"。

### 条件数

条件数 $\kappa(W) = \sigma_{\max}/\sigma_{\min}$ 衡量矩阵"各方向放大倍数的不均匀度"：$\kappa$ 越大，输入的微小变化在窄方向上被放得越大，数值计算与优化越敏感。训练中 weight decay、归一化与稳定的学习率本质上都在抑制病态（大 $\kappa$）方向的增长。

---

## 五、SVD 与 LoRA：最优的"少参数重建"

### SVD：把矩阵拆成"按重要性排序的外积"

> 基础名词：**奇异值分解（SVD）** 把任意矩阵写成 $W = U \Sigma V^{\top} = \sum_i \sigma_i u_i v_i^{\top}$：一组相互正交的"输出方向" $u_i$、一组正交"输入方向" $v_i$，以及按大小排序的"强度" $\sigma_i$。

这正是 [[linear-algebra-for-llm]] 视角 4（外积之和）的完美落地：**任何矩阵 = 一堆按重要性排序的 rank-1 外积**。

### Eckart–Young 定理：截断即最优

保留前 $k$ 项得到 $W_k = \sum_{i \le k} \sigma_i u_i v_i^{\top}$，则 $W_k$ 是所有秩 $\le k$ 矩阵中 Frobenius 距离最小的，且误差有闭式解：

$$\|W - W_k\|_F = \sqrt{\textstyle\sum_{i > k} \sigma_i^2}$$

实验 Exp 4 用快速衰减谱（$\sigma_i = 10 \cdot 0.65^i$）验证：保留 8 / 32 个方向就守住 99.90% / 100% 的能量，实测误差与理论闭式逐项吻合（`abs_diff ≈ 1e-15`）。

### LoRA：训练一个低秩增量，而不是全量矩阵

微调不要求重写整个 $W$，只求一个低秩增量 $\Delta W$。既然 SVD 告诉我们"好矩阵可低秩近似"，LoRA 干脆把 $\Delta W$ 参数化为两个瘦矩阵的乘积：

$$W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times d},\ r \ll d$$

可训练参数量 $2dr$ 相对全量 $d^2$ 的比例为 $2r/d$：$d=4096, r=16$ 时仅 **0.781%（128 倍缩减）**。实验同时验证了 $BA$ 与"取 SVD 前 $r$ 个外积之和"精确等价（误差 `2.776e-16`）——**LoRA 的表达能力上界就是秩 $r$ 的最优低秩近似**，这也是选 $r$ 时的理论依据。QLoRA 再叠加量化冻结基座，工程细节见 [[lora-and-qlora]]。

同样的思想反向用在推理侧：MLA 的 512 维 latent KV 压缩（第二节）、以及各类 KV 低秩/量化方案，都在押注同一个数学事实——**注意力相关的矩阵族具有良好的低秩结构**。

---

## 自检清单

- [ ] 能说出 $(AB)x = A(Bx)$ 在什么场景下（矩阵 × 向量）吸收一侧更便宜，为什么。
- [ ] 能手推 MLA 吸收：$\tilde{q} = (W^{UK})^{\top} q$ 与把 $W^{UV}$ 推迟到加权求和之后，并解释为何与原路径零误差等价。
- [ ] 能背出 DeepSeek-V3 缓存形态的字节对比（81,920 → 1,152 B/token，约 71×）。
- [ ] 能手推 $\langle R_m q, R_n k\rangle = \langle q, R_{n-m} k \rangle$，并解释为什么每个二维对用不同频率 $\theta_i$。
- [ ] 能解释为什么 MLA 的 RoPE 维度必须与可吸收的 nope 部分解耦缓存。
- [ ] 能说出谱半径、残差、Pre-Norm 三者如何共同决定百层网络的稳定性。
- [ ] 能陈述 Eckart–Young 结论与 LoRA 参数比 $2r/d$，并解释 LoRA 与 SVD 的关系。

## 面试要点

### Q：MLA 权重吸收为什么在 Decode 阶段收益最大？数学依据是什么？

> [!question]- 参考答案（点击展开）
>
> 依据是矩阵乘法结合律 $(AB)x = A(Bx)$：不吸收时每步要对 $T \times H$ 个历史 Token 逐个做 $512 \times 128$ 解压矩阵乘，并把解压后的 K/V 读一遍；吸收后 Query 侧只需每步一次 $(W^{UK})^{\top}q$，Value 侧把 $W^{UV}$ 穿过求和号推迟到加权求和之后只乘一次，Kernel 直接在压缩 latent $c_{kv}$ 上计算注意力。DeepSeek-V3 量级下每 Token 缓存从 81,920 B 降到 1,152 B（约 71×）。Decode 每步都要重扫全部 KV 且算术强度极低（带宽受限），因此该换序的带宽收益全部兑现在 Decode；Prefill 计算受限，通常走解压路径。

### Q：RoPE 如何做到"注意力打分只依赖相对位置"？为什么工程实现不构造完整旋转矩阵？

> [!question]- 参考答案（点击展开）
>
> RoPE 把向量切成 $d/2$ 个二维对，第 $i$ 对按位置旋转 $pos \cdot \theta_i$。由于同一频率的旋转矩阵可交换且 $R_m^{\top} R_n = R_{n-m}$，打分 $\langle R_m q, R_n k \rangle = \langle q, R_{n-m} k \rangle$ 只依赖相对距离 $n-m$。工程实现逐对做 O(d) 旋转即可，构造 $d \times d$ 矩阵既浪费计算也浪费带宽；FlashInfer 甚至把旋转在线融合进 Decode Kernel 的 K Tile 加载阶段，省去独立 Kernel 的 HBM 往返。另外 RoPE 旋转依赖绝对位置、不能穿过权重吸收，因此 MLA 把 64 维 RoPE 键与可吸收的 latent 部分解耦缓存。

### Q：为什么现代 LLM 几乎都采用 Pre-Norm + RMSNorm + 残差的组合？

> [!question]- 参考答案（点击展开）
>
> 深网络的核心风险是逐层增益连乘导致的梯度爆炸/消失（谱半径 $\sigma_{\max}$ 偏离 1 时 $L$ 层呈指数放大）。残差把信息传递从乘法链改为加法链，梯度可沿恒等分支无衰减回传；Pre-Norm 使每层输入尺度有界，从而每层注入的扰动有界、$L$ 层总扰动线性而非指数增长；RMSNorm 只做模长归一、无均值统计与中心化偏置，计算量小且对硬件归约友好。三者组合把"百层堆叠"从数值悬崖变成工程默认。

### Q：LoRA 与 SVD 是什么关系？如何据此选择秩 $r$？

> [!question]- 参考答案（点击展开）
>
> LoRA 把增量 $\Delta W = BA$ 约束在秩 $\le r$ 的矩阵空间内，而 SVD + Eckart–Young 定理表明秩 $r$ 矩阵对目标的最佳逼近就是保留前 $r$ 个奇异方向、误差为 $\sqrt{\sum_{i>r}\sigma_i^2}$。因此 LoRA 的表达上界由权重增量的奇异谱衰减速度决定：谱衰减快（能量集中于前几个方向）时小 $r$ 即可，谱平坦时需要更大 $r$。实验可用能量保留率（如保留 8 个方向守 99.9%）指导 $r$ 的数量级选择；参数开销为 $2r/d$（$d=4096, r=16$ 时约 0.78%）。

## 参考与版本边界

- 本文为数学直觉笔记，公式为标准结果（Online Softmax 引 Milakov & Gimelshein 2018 思路、Eckart–Young 为经典定理）；涉及的算子与工程路径（`BatchMLAPagedAttentionWrapper`、MLA 双路径、RoPE 在线融合）以 [[flashinfer-ragged-prefill-and-mla-source]] 与 [[sglang-srt-radix-and-scheduler]] 记载的版本边界（FlashInfer v0.7.3, commit `82caa5668`，审查日期 `2026-11-17`）为准。
- 配套可运行验证：[LLM 数学直觉与 SIMD 并行离线实验](../experiments/llm-math-and-simd-lab/README.md)（Exp 2 RoPE 旋转不变量、Exp 3 MLA 吸收等价与 HBM 字节审计、Exp 4 SVD 与 LoRA 等价）。
- 深入方向：低秩近似的算子实现见 [[flashinfer-ragged-prefill-and-mla-source]]；LoRA/QLoRA 工程实践见 [[lora-and-qlora]]；Decode 带宽受限的定量分析见 [[gpu-execution-and-memory-model]]。
