# LLM Foundations

> 返回 [AI 学习索引](../README.md)

共同基础是矩阵乘、张量形状和一次前向计算。做训练，接着学生成梯度和更新参数；做推理，转向布局、访存和缓存。两条线不需要互相等待读完。

## 为什么学大模型无需盲啃整本数学系教材

很多工程背景的学习者一准备深入大模型底层（如 FlashAttention、FlashInfer、RoPE、LoRA、DeepSeek MLA），就误以为必须先把数学系《线性代数》和《高等代数》厚教材从头啃完——结果卡在抽象线性空间公理、n 阶行列式展开、克拉默法则、伴随矩阵求逆和若尔当标准型（Jordan Form）里半途而废。

事实上，**现代大模型训练与 GPU 算子几乎从不计算行列式，也绝不显式求大矩阵的逆**。看懂大模型架构演进与高性能 CUDA 内核，真正反复用到的只有高度聚焦的两组核心代数武器：

1. **线性代数层（聚焦几何投影、分块累加与物理内存布局）**：理解点积的投影意义与方差缩放（$QK^T / \sqrt{d_k}$、RMSNorm）、**矩阵乘法的“列×行外积之和”与分块矩阵乘（Block GEMM / Tiling）**（为什么切小块放进 GPU SRAM / Tensor Core 局部累加与全量相乘严格等价），以及高维张量的 **`Shape` vs `Stride`（步长）** 如何直接决定 CPU SIMD 连续车道加载与 GPU 128-bit（`float4` / `vec_t`）合并访存。
2. **高等代数层（聚焦结合律算力魔法、正交旋转、特征谱与低秩分解）**：理解秩与**矩阵乘法结合律 $(AB)x = A(Bx)$**（DeepSeek MLA 权重吸收如何在数学零误差前提下省掉巨量 KV Cache 带宽）、**正交旋转矩阵保长保内积性质**（5 分钟手推 RoPE 相对位置编码）、**特征值谱半径与条件数**（为什么百层网络必须靠残差连接与 Pre-Norm 存活），以及 **奇异值分解 SVD**（LoRA 微调与 KV 低秩压缩的理论根基）。

本目录把上述真正决定工程上限的数学本质浓缩为两篇“白话比喻 + $2 \times 2$ 小数字手算 + 算子代码映射”的专属笔记，并配套零第三方依赖的离线实验脚本，让你跳过掉书袋证明、直达大模型底层核心。

## 推荐阅读顺序

**训练方向的最短路径**：[面向大模型的线性代数](linear-algebra-for-llm.md) → [一次参数更新：从矩阵梯度到有效 token](training-step-and-gradients.md) → [分片梯度 CPU 实验](../experiments/sharded-gradient/README.md) → [万卡集群调度与模型训练](../training/README.md)。先过 `dW = XᵀG` 和全局 batch 两关，再按需回读 SVD、RoPE 与 MLA。

### 0. 专属数学直觉（打通公式推导与硬件内存布局）

1. [[linear-algebra-for-llm|面向大模型的线性代数：从几何直觉到张量与分块计算]] —— 向量点积与 RMSNorm、矩阵乘法四种黄金视角与分块 GEMM Tiling、QKV/MLP 投影几何、高维张量 `Shape/Stride` 与 SIMD/CUDA 合并访存。
2. [[advanced-algebra-for-llm|面向大模型的高等代数：子空间、正交旋转、特征谱与低秩分解]] —— 秩与低维子空间、结合律与 DeepSeek MLA 权重吸收、正交旋转矩阵与 RoPE 推导、特征值谱与残差/Pre-Norm、SVD 与 LoRA 低秩近似。
   - **配套动手实验**：[LLM 数学直觉与 SIMD 并行离线实验](../experiments/llm-math-and-simd-lab/README.md)（纯 Python 标准库运行，带 `< 1e-12` 误差断言验证分块矩阵乘、RoPE、MLA 吸收、SVD 截断与 8-lane SIMD 归约）。

### 1. 全局生命周期路线（建立宏观技术坐标系）

3. [[llm-learning-path|LLM 零基础学习路线]] —— 厘清 Pre-training、SFT、LoRA、Preference / Online RL 与 vLLM / SGLang 推理系统各自解决什么问题，规划从零基础到算子/推理、后训练与端到端实践的阶段标准。

### 2. 核心数据流概念（串联模型内部关键对象）

4. [[llm-fundamentals|LLM 基础概念]] —— 沿着 `Prompt → Tokenizer → Embedding → Transformer Block (RMSNorm / QKV / RoPE / Attention / MLP) → Logits → Sampling` 走通真实数据流，明确 Checkpoint、Adapter 与 Inference Server 边界。

## 下一步去哪里

- 立志深耕 **CUDA 生态、GPU 并行模型与 FlashInfer 算子源码** → 进入 [GPU 算子与 FlashInfer 学习索引](../kernels/README.md)
- 想打通 **FlashInfer × vLLM V1 × SGLang SRT 全栈推理系统与分布式架构** → 进入 [LLM Inference 学习索引](../inference/README.md)（总路线见 [[flashinfer-vllm-sglang-learning-path]]）
- 想深入 **LoRA / QLoRA、SFT、DPO、PPO 与 GRPO 训练闭环** → 进入 [LLM Post-training](../post-training/README.md)
