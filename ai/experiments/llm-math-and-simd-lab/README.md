# LLM 数学直觉与 SIMD 并行离线实验

> 返回 [AI Experiments](../README.md)

相关笔记：[[linear-algebra-for-llm]] | [[advanced-algebra-for-llm]] | [[simd-and-gpu-simt-foundations]] | [[gpu-execution-and-memory-model]]

## 目标

脱离 GPU 与第三方依赖，用纯 Python 标准库在 CPU 上验证 [[linear-algebra-for-llm]] 与 [[advanced-algebra-for-llm]] 的全部核心数值不变量：

1. **矩阵乘四种视角与 2D Block GEMM Tiling**：行×列内积、行×矩阵、矩阵×列、列×行外积之和严格等价；分块（含非整除尾块）与全量相乘严格等价；分块 Online Softmax Attention 与全量 Materialization Attention 严格等价。
2. **正交旋转与 RoPE 相对位置不变量**：$R^{\top}R = I$、模长保持、$\langle R_m q, R_n k\rangle = \langle q, R_{n-m} k\rangle$、两侧同平移 $\delta$ 打分不变。
3. **结合律与 DeepSeek MLA 权重吸收**：吸收路径（$\tilde{q} = (W^{UK})^{\top}q$、$W^{UV}$ 推迟到加权求和后）与解压路径的分数与输出零误差等价，并按 DeepSeek-V3 尺寸审计 Decode 每步 KV HBM 读取字节数。
4. **SVD rank-1 外积展开与 Eckart–Young**：截断误差实测值与理论闭式 $\sqrt{\sum_{i>k}\sigma_i^2}$ 逐项吻合；LoRA $BA$ 与前 $r$ 个外积之和等价。
5. **8-lane SIMD FMA 点积 + 蝴蝶树形归约 + 张量 Stride 零拷贝转置**：掩码尾块处理、3 步 XOR 蝴蝶归约、NHD/HND/DHN 三视图共享同一 buffer。

## 环境

- Python 3.9+；仅使用标准库（`math`、`random`、`dataclasses`、`typing`）。
- Mac / Linux / Windows CPU 均可运行，耗时约 1 秒；不需要 PyTorch、NumPy、CUDA 或网络。
- 固定随机种子 `SEED = 20261117`，输出可复现；全部断言容差 `TOL = 1e-12`。

## 命令

从仓库根目录运行：

```bash
python3 ai/experiments/llm-math-and-simd-lab/llm_math_simd_lab.py
```

## 预期现象

- 5 个实验依次输出，每项误差均 `< 1e-12`，末尾打印 `SELF-CHECK: PASS`：

```text
=== [Exp 1] Outer-Product View & 2D Block GEMM Tiling ===
GEMM shape=(29x37)@(37x31), tile=(8,8,8), sram_tile_loads=80
  outer_sum_err=2.665e-15 | block_tiling_err=2.665e-15 | tiled_online_attn_err=8.327e-17

=== [Exp 2] Orthogonal Rotation & RoPE Relative Position Invariance ===
head_dim=64 (32 2D subspaces), tested pairs=5, shift_dot(10,25)==(310,325): -5.950486
  R^T R==I err=2.220e-16 | norm_preserve_err=0.000e+00 | <R_m q, R_n k>==<q, R_{n-m} k> err=2.398e-14 | shift_err=2.220e-14

=== [Exp 3] Matrix Associativity & DeepSeek MLA Weight Absorption ===
Numerical check (H=8, T=24, d_c=32, d_nope=16, d_r=8): score_err=5.551e-17 | out_err=5.551e-17
DeepSeek-V3 scale (H=128, T=8192, BF16) KV HBM read per Decode step:
  Full MHA KV     :  640.00 MiB (81,920 B/tok)
  Unabsorbed KV   :  513.00 MiB (65,664 B/tok)
  MLA Absorbed KV :    9.00 MiB (1,152 B/tok) -> 71.11x smaller vs Full MHA (57.00x vs Unabsorbed)

=== [Exp 4] SVD Rank-1 Outer Products, Eckart-Young & LoRA Compression ===
  rank_k | energy_kept | empirical ||W - W_k||_F | theoretical sqrt(sum_{i>k} s_i^2) | abs_diff
  k=   1 |    57.7500% |            8.553372e+00 |                      8.553372e+00 | 5.33e-15
  k=   2 |    82.1494% |            5.559692e+00 |                      5.559692e+00 | 1.78e-15
  k=   4 |    96.8136% |            2.348970e+00 |                      2.348970e+00 | 0.00e+00
  k=   8 |    99.8985% |            4.193058e-01 |                      4.193058e-01 | 0.00e+00
  k=  16 |    99.9999% |            1.336095e-02 |                      1.336095e-02 | 9.89e-17
  k=  32 |   100.0000% |            3.383865e-15 |                      0.000000e+00 | 3.38e-15
  LoRA equivalence (B@A == sum_{i=1}^4 s_i u_i v_i^T) err=2.776e-16 | LLM d=4096, r=16 trainable params=0.781% (128x fewer)

=== [Exp 5] 8-Lane SIMD FMA + Butterfly Reduction & Tensor Stride View ===
SIMD8 dot (D=133 = 16x8 + 5 masked tail): scalar=-0.08084232, simd8=-0.08084232, abs_err=1.027e-15
Zero-copy transpose check (same buffer=True, err=0.0e+00):
  NHD shape=(16, 4, 8), strides=(32, 8, 1) -> inner stride=1 (SIMD/LDG.128 contiguous)
  HND shape=(4, 16, 8), strides=(8, 32, 1) -> inner stride=1 (per-head token jump=32)
  DHN shape=(8, 4, 16), strides=(1, 8, 32) -> inner stride=32 (strided gather penalty!)

MAX ABS ERROR ACROSS ALL 5 EXPERIMENTS: 2.398e-14 (< 1e-12)
SELF-CHECK: PASS
```

- Exp 1 特意使用非 8 倍数的形状 `29×37×31` 与 `br=6, bc=7` 的 Attention 分块，覆盖边界尾块路径。
- Exp 3 的字节审计数字（81,920 → 1,152 B/token，71.11×）与 [[advanced-algebra-for-llm]] 手算一致。
- Exp 5 打印 `same buffer=True`，证明零拷贝转置只改 `shape/stride` 元数据。

## 解释

- **为什么分块与吸收是"严格等价"而非"近似"**：矩阵乘的累加只依赖加法结合律与分配律，分块只是重新分组求和；MLA 吸收只是对 $(AB)x$ 与 $A(Bx)$ 换序，均为实数域恒等变形，故在 float64 下误差仅为浮点舍入量级（`1e-14 ~ 1e-16`）。
- **为什么 MLA 能砍 71× 带宽**：Decode 每步的 HBM 读取量由"每 Token 缓存多少字节"决定。全量 MHA 每 Token 缓存 $H(d_{nope}+d_r+d_v) = 128 \times 320$ 维；吸收后只缓存共享 latent $(d_c + d_r) = 576$ 维，且 Query 侧的解压矩阵吸收进每步只算一次的 $\tilde{q}$，$W^{UV}$ 推迟到加权求和之后只乘一次。
- **为什么蝴蝶归约是 $\log_2 W$ 步**：XOR 配对（offset 4→2→1）每步把未配对 lane 数减半，8 lane 只需 3 步；这正对应 GPU Warp 内 `__shfl_xor_sync` 以 offset 16→8→4→2→1 共 5 步完成 32 lane 归约（见 [[simd-and-gpu-simt-foundations]]）。
- **为什么 Stride 决定访存性能**：最内维 `stride=1` 的 NHD/HND 视图可被 SIMD 车道与 GPU 128-bit 合并访存高效消费；DHN 的最内维跨 32 元素，模拟 strided gather 惩罚。

## 已知限制与硬件差异

- 本实验在 IEEE-754 `float64` 标量下运行，误差上界 `1e-12` 是数学等价性的验证；真实 GPU Kernel 使用 FP16/BF16/FP8/NVFP4 与 Tensor Core MMA，内部累加与存储转换会引入 `1e-3 ~ 1e-2` 量级舍入差异，不能把本实验结论直接当作低精度内核的精度保证。
- `SimdVec8` / `StridedTensorView` 是纯 Python 模拟：模拟的是"8 车道同时 FMA""(shape, strides) 元数据寻址"这些**逻辑模型**，不包含真实 SIMD 指令调度、Cache Line 合并、SRAM 容量约束与 Bank Conflict 等微架构行为。
- Exp 3 的 DeepSeek-V3 字节审计只统计 Decode Attention 读取的 KV Cache 字节数，不含权重读取、Workspace 往返与多级存储（L2 命中）效应。
- RoPE 验证使用 base=10000 的 Llama 风格实现与逐对旋转（`apply_rope_fast`），不覆盖 YaRN / NTK 缩放等变体。

## 自检问题

### Q：为什么 LoRA 的 $BA$ 可以与"取 SVD 前 $r$ 个 rank-1 外积之和"精确等价？这对选择 $r$ 意味着什么？

> [!question]- 参考答案（点击展开）
>
> 设 $\Delta W$ 的 SVD 为 $\sum_i \sigma_i u_i v_i^{\top}$，取 $B = [\sigma_1 u_1, \dots, \sigma_r u_r]$（每列缩放 $\sigma_i$）、$A = [v_1, \dots, v_r]^{\top}$，则 $BA = \sum_{i \le r} \sigma_i u_i v_i^{\top}$，即秩 $r$ 的最优（Eckart–Young）低秩近似。因此 LoRA 的表达上界由权重增量的奇异谱衰减速度决定：谱快衰减时小 $r$ 即可（实验中 $r=8$ 已保留 99.90% 能量），谱平坦时需增大 $r$；参数开销为 $2r/d$。

### Q：实验中吸收路径的 HBM 审计为什么按"每 Token 字节数"计算，而不是按矩阵乘 FLOPs？

> [!question]- 参考答案（点击展开）
>
> Decode Attention 的算术强度极低（GQA 下约 $g$ FLOP/Byte），是典型的 HBM 带宽受限算子，性能由"每步从显存读回多少 KV 字节"决定。每 Token 缓存字节数：全量 MHA 为 $H(d_{nope}+d_r+d_v) \times 2\text{B}$，MLA 吸收后为 $(d_c + d_r) \times 2\text{B}$，比值即带宽收益（128×320/576 ≈ 71×）。吸收路径省的正是"解压后的 K/V 根本不需要存在于缓存与显存"这一层字节。

### Q：`SimdVec8.butterfly_reduce_sum()` 的 XOR 配对归约，与 GPU Warp 的 `__shfl_xor_sync` 有什么对应关系？

> [!question]- 参考答案（点击展开）
>
> 两者是同一算法在不同宽度的实例：`lane ^ offset` 的 XOR 配对让每步参与归约的 lane 数减半，8 lane 用 offset 4→2→1 共 $\log_2 8 = 3$ 步；32 lane 的 Warp 用 offset 16→8→4→2→1 共 5 步，且数据只经寄存器交换、不经过 Shared Memory。FlashInfer 的 `math::shfl_xor_sync`（PTX `shfl.sync.bfly.b32`）即该范式的硬件实现。

## 验证状态

- 验证日期与环境：`2026-11-18`，Mac（Apple Silicon）CPU，Python 3.14.7，无 GPU、无第三方依赖。
- 实测结果：5 组实验全部通过 `TOL = 1e-12` 断言，全库最大绝对误差 `2.398e-14`，输出 `SELF-CHECK: PASS`（完整输出见上文"预期现象"）。
- 已知限制：见上文"已知限制与硬件差异"。
