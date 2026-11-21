# 全库知识速查索引

> ⬆ 返回 [知识库首页](../README.md) · 架构说明见 [仓库架构](architecture.md) · 按标签查看 [全库标签聚合索引](tags.md)
>
> 本文件由 `python3 scripts/search_notes.py --rebuild-catalog` 自动扫描生成并同步重建 `.traces/notes_fts.sqlite3` FTS5 索引（不手工维护脆弱的精确篇数统计）。

## 仓库入口与规范

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [AI Infra 学习笔记](../README.md) | — | — | 从单卡训练到万卡集群，要同时看模型怎样更新、GPU 怎样分配，以及跨机通信花了多少时间。 |
| [Learning Notes 协作规则](../AGENTS.md) | — | — | 顶层收拢为四大知识支柱与文档/工具目录： |
| [Learning Notes 项目说明](../CLAUDE.md) | — | — | 本仓库按四大知识支柱组织个人技术学习笔记，兼容 Obsidian 与 GitHub，全部采用纯文本 Markdown + Python 标准库维护，不包含任何前端依赖或 node_modules。 |

## AI / LLM Systems

### ai

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [模型训练与推理](../ai/README.md) | — | — | 刚开始补基础时，用矩阵乘法算出一个线性层的输出，再跟着损失、梯度与参数更新手算一个训练步。 |
| [AI Infra 实战路线：交付一个能训练、能恢复、能服务的模型](../ai/ai-infra-fieldwork.md) | [[ai-infra-fieldwork]] | `#ai` `#ai-infra` `#学习路线` | 适合会运行 Python 脚本、刚开始做训练平台的人。先拿一份可比的单步更新，再把它交给多进程和调度器；最后判断恢复后是否重复算了数据，以及服务延迟量的到底是什么。以下命令都从仓库根目录执行。没... |

### ai/agents

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [跟一次模型工具调用](../ai/agents/README.md) | — | — | 打开工具调用源码导读，选一个例子，只记三样东西：模型请求了什么参数、工具实际返回什么、模型下一轮看到了什么。这样才能分清是工具执行错了，还是模型拿到正确结果后解释错了。 |
| [Agent 开发学习路线](../ai/agents/agent-development-learning-path.md) | [[agent-development-learning-path]] | `#ai` `#agent` `#rag` `#面试` `#学习计划` | Agent 开发面试的三件大事——RAG（可信知识）、ReAct / Tool Loop（动作闭环）、Plan / Graph Workflow（复杂任务编排）——不是三个孤立名词，而是一条能落... |
| [Agent 开发框架源码导读](../ai/agents/agent-development-source.md) | [[agent-development-source]] | `#ai` `#agent` `#rag` `#源码导读` | 本篇是 Agent 开发框架源码导读，目标不是把每个框架都读完，而是抓住生产 Agent 的共同骨架：状态图 / loop runtime / tool schema / handoff / c... |
| [生产级 Agent 开发](../ai/agents/production-agent-development.md) | [[production-agent-development]] | `#ai` `#agent` `#production` `#observability` `#安全` | 生产级 Agent 不是“能调用工具的 Demo”，而是一个受控的自动化系统：LLM 只负责理解、规划和生成结构化决策；真正决定能否上线的是状态管理、工具权限、幂等、可观测、评估、灰度、回滚和人... |

### ai/foundations

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [模型基础：看懂一次矩阵乘法](../ai/foundations/README.md) | — | — | 从分块矩阵乘实验开始。矩阵边长不能整除分块宽度时，最后一小块也要参与计算。先对比整块乘和分块乘的结果，再手算一个元素的位置。 |
| [在 CPU 上验证矩阵分块、RoPE 和低秩分解](../ai/foundations/llm-math-and-simd-lab/README.md) | — | — | 先看第一组乘法。把同一个矩阵切成小块再乘，最后的边缘块凑不满 8 行时，结果还应和整块相乘一致。输出里的 block_tiling_err 就是两种写法的最大误差。 |
| [面向大模型的高等代数：子空间、正交旋转、特征谱与低秩分解](../ai/foundations/advanced-algebra-for-llm.md) | [[advanced-algebra-for-llm]] | `#ai` `#llm` `#高等代数` `#数学基础` `#rope` `#svd` `#lora` `#mla` | 如果说 linear-algebra-for-llm 解决的是"矩阵乘怎么算、张量怎么放"，本笔记回答的是另一类问题：为什么大模型里的几个关键设计（MLA 权重吸收、RoPE、残差连接、LoRA... |
| [面向大模型的线性代数：从几何直觉到张量与分块计算](../ai/foundations/linear-algebra-for-llm.md) | [[linear-algebra-for-llm]] | `#ai` `#llm` `#线性代数` `#数学基础` `#simd` `#cuda` | 很多工程背景的学习者一准备深入 FlashAttention / FlashInfer / vLLM 底层，就想先把数学系《线性代数》整本啃完，结果卡在行列式展开、伴随矩阵求逆和若尔当标准型里半... |
| [LLM 基础概念](../ai/foundations/llm-fundamentals.md) | [[llm-fundamentals]] | `#ai` `#llm` `#transformer` `#基础概念` | 本笔记解释后续 LLM 推理与训练都会反复遇到的基础名词，并把它们串成一条真实的数据流。目标不是一次记住全部定义，而是每遇到一个对象，都知道它来自哪里、要去哪里。 |
| [LLM 零基础学习路线](../ai/foundations/llm-learning-path.md) | [[llm-learning-path]] | `#ai` `#llm` `#学习路线` `#vllm` `#sglang` `#lora` `#post-training` `#rl` | 这是一条面向零基础读者的 LLM 学习主线。目标不是先记住工具名称，而是理解一个模型从“文本如何变成数字”，到“如何训练、微调、生成、评测和部署”的完整生命周期。 |
| [一次参数更新：从矩阵梯度到有效 token](../ai/foundations/training-step-and-gradients.md) | [[training-step-and-gradients]] | `#ai` `#机器学习` `#线性代数` `#分布式训练` | 把语言模型先缩小成一个预测头：每个有效位置给出一行特征，模型要为下一 token 的真实类别打高分。训练系统无论使用一张卡还是多张卡，都应先说明自己在最小化哪一个标量。下面假设无类别权重、无标签... |

### ai/inference

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [推理服务：先量用户等了多久](../ai/inference/README.md) | — | — | 做流式请求实验：固定模型和请求集，分别看首次响应、生成结束、实际输出 token 数。没有服务时可先跑客户端自测，它不给出引擎排名。 |
| [比较 vLLM 和 SGLang 的流式响应](../ai/inference/openai-compatible-benchmark/README.md) | — | — | 同一批提示分别发给两个服务。先看请求多久收到第一个 SSE 事件，再看整批请求每秒返回多少个输出 token。两个指标对应不同的使用体验。 |
| [同一份 JSON 为什么会有两个前缀哈希](../ai/inference/prefix-input-determinism/README.md) | — | — | 把相同字段按不同顺序写入两个 JSON。先比较序列化后的字节和 SHA-256，再比较按键名排序后的结果。 |
| [用五个候选词观察温度和 top-p](../ai/inference/sampling/README.md) | — | — | 同一组 logits 可以产生不同的下一词分布。先降低温度，再缩小 top-p，观察改变的是概率还是候选集合。 |
| [FlashInfer × vLLM × SGLang 全栈学习路线：从数学与算子底座到双推理引擎](../ai/inference/flashinfer-vllm-sglang-learning-path.md) | [[flashinfer-vllm-sglang-learning-path]] | `#ai` `#llm` `#inference` `#flashinfer` `#vllm` `#sglang` `#cuda` `#learning-path` | 很多初学者一头扎进大模型推理系统时，常陷入两类极端困境： |
| [LLM 推理系统学习路线](../ai/inference/llm-inference-learning-path.md) | [[llm-inference-learning-path]] | `#ai` `#llm` `#inference` `#kv-cache` `#pd-disaggregation` `#router` `#学习计划` | LLM 推理系统的三件大事——Inference Router（请求路由层）、KV Cache（显存最大的金山）、PD 分离（Prefill/Decode 解耦）——是 2024 年以后大模型基... |
| [LLM 推理完整链路](../ai/inference/llm-inference-pipeline.md) | [[llm-inference-pipeline]] | `#ai` `#llm` `#inference` `#prefill` `#decode` `#kv-cache` | 本笔记从一次真实请求出发，解释 LLM 如何把用户文字变成 Token，经过 Prefill 和 Decode 逐步生成回答，再由推理服务返回。第一次阅读只需要掌握数据流；PagedAttent... |
| [LLM 推理学习进度](../ai/inference/llm-inference-progress.md) | [[llm-inference-progress]] | `#ai` `#llm` `#inference` `#学习计划` `#打卡` | 下面的「Week 1 ~ Week 8」是逻辑顺序，不是日历周。LLM 推理这条线的源码总量比 K8s 大、而且要看论文，节奏比 progress 更松一些： |
| [Go Map 序列化导致 Prefill Cache Miss 问题分析](../ai/inference/prefill-cache-miss.md) | [[prefill-cache-miss]] | `#prefill` `#ai` `#go` | LLM 推理引擎通过 Prefix Cache 缓存 Prefill 阶段的 KV 状态，避免重复计算。Cache 命中的前提是：Prompt 的 Token 序列必须精确匹配。 |
| [SGLang (SRT) 核心架构深度解析：RadixAttention、零开销重叠调度与结构化解码](../ai/inference/sglang-srt-radix-and-scheduler.md) | [[sglang-srt-radix-and-scheduler]] | `#ai` `#llm` `#inference` `#sglang` `#radix-attention` `#scheduler` `#structured-decoding` `#源码导读` | 早期大模型推理引擎的核心假设是“相互独立的单轮一问一答”：用户发来一段 Prompt，引擎做一次 Prefill 算出 KV Cache，逐 Token 自回归 Decode 吐完回答，随后立刻... |
| [vLLM 与 SGLang](../ai/inference/vllm-and-sglang.md) | [[vllm-and-sglang]] | `#ai` `#llm` `#inference` `#vllm` `#sglang` | vLLM 和 SGLang 都能把模型权重变成高吞吐、低延迟的推理服务。学习顺序应当是：先用同一个模型跑通两个服务，再用可控实验观察差异，最后才读 Scheduler、KV Cache 和 Ke... |
| [vLLM 架构演进与 V1 核心机制深度解析：统一调度、BlockPool 与 Piecewise CUDA Graph](../ai/inference/vllm-v1-architecture-and-internals.md) | [[vllm-v1-architecture-and-internals]] | `#ai` `#llm` `#inference` `#vllm` `#paged-attention` `#cuda-graph` `#flashinfer` `#源码导读` | 源码与版本边界：本文基于 vLLM V1 核心架构主线（源码根目录 vllm/v1/，自 vLLM v0.8+ 起成为新一代默认架构）与 FlashInfer v0.7.3（审查日期 2026-... |

### ai/kernels

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [CUDA 与 FlashInfer：先验证算对，再测算快](../ai/kernels/README.md) | — | — | 先做把 KV 拆开算再合回去：同一段历史序列分块前后，注意力输出应在约定误差内一致。CPU 就能检验公式；评估访存或执行时间则需要 GPU。 |
| [把 KV 切成四块，再合并一次注意力结果](../ai/kernels/online-softmax-and-split-kv/README.md) | — | — | 取长度为 19 的请求，每 6 个位置分一块。脚本把四块分别计算，再按正序、逆序和乱序合并，逐项对照一次读完 KV 的结果。 |
| [CUDA 全景生态与算子工程师进阶路线：从 PTX 到 CUTLASS、CuTe 与 Triton](../ai/kernels/cuda-ecosystem-and-career-roadmap.md) | [[cuda-ecosystem-and-career-roadmap]] | `#ai` `#llm` `#gpu` `#cuda` `#ptx` `#cutlass` `#cute` `#triton` `#学习路线` | 很多立志往 CUDA 算子开发 / AI Infra / 推理引擎 方向深耕的学习者，一打开开源仓库（如 FlashInfer、vLLM、SGLang、DeepGEMM、DeepEP）就会迎面撞... |
| [FlashInfer 三层架构与 JIT 编译体系源码导读](../ai/kernels/flashinfer-architecture-and-jit.md) | [[flashinfer-architecture-and-jit]] | `#ai` `#llm` `#flashinfer` `#jit` `#tvm-ffi` `#cutlass` `#cute-dsl` `#源码导读` | FlashInfer 能够在单个算子参数组合爆炸（dtype、head_dim、layout、attention variant、SM 架构）的前提下同时保持可热重载的研发体验与跨框架复用能力，... |
| [FlashInfer 零门槛导览：白话手算页表与 Plan-Run 契约](../ai/kernels/flashinfer-intuition-and-engine-bridge.md) | [[flashinfer-intuition-and-engine-bridge]] | `#ai` `#llm` `#flashinfer` `#cuda` `#paged-kv` `#入门导览` | 这是进入 FlashInfer 源码世界前的最后一篇零门槛导览：不需要 CUDA、不需要懂 C++ 模板、不需要 GPU，一支笔一张纸就能读完。读完后你将能回答三个问题： |
| [FlashInfer 系统学习路线与实验工作流](../ai/kernels/flashinfer-learning-roadmap.md) | [[flashinfer-learning-roadmap]] | `#ai` `#llm` `#flashinfer` `#cuda` `#learning-path` `#学习路线` | FlashInfer 是面向 LLM 服务的高性能 GPU 内核库，也是 vLLM、SGLang 等现代推理引擎的重要算子底座。它的工程复杂性横跨三层：Python 运行时与 Plan-Run... |
| [FlashInfer Norm 与 Sampling 源码导读](../ai/kernels/flashinfer-norm-and-sampling-source.md) | [[flashinfer-norm-and-sampling-source]] | `#ai` `#llm` `#flashinfer` `#cuda` `#rmsnorm` `#sampling` `#源码导读` | Norm 与 Sampling 是进入 FlashInfer 内核代码最理想的两条切入路径——它们没有复杂的 Paged KV 间接寻址，却完整覆盖了 FlashInfer 最常用的 GPU 工... |
| [FlashInfer Paged KV 与 Decode 内核源码导读](../ai/kernels/flashinfer-paged-kv-and-decode-source.md) | [[flashinfer-paged-kv-and-decode-source]] | `#ai` `#llm` `#gpu` `#flashinfer` `#cuda` `#paged-kv` `#decode` | 源码版本边界：本文所有文件路径、结构体与函数签名均对照 FlashInfer v0.7.3 (commit 82caa5668, 审查日期 2026-11-17) 核验。为避免随提交漂移，文中仅... |
| [FlashInfer Ragged Prefill、Cascade 与 MLA 源码导读](../ai/kernels/flashinfer-ragged-prefill-and-mla-source.md) | [[flashinfer-ragged-prefill-and-mla-source]] | `#ai` `#llm` `#gpu` `#flashinfer` `#cuda` `#prefill` `#cascade` `#mla` | 源码版本边界：本文所有文件路径、类与函数签名均对照 FlashInfer v0.7.3 (commit 82caa5668, 审查日期 2026-11-17) 核验。为避免随提交漂移，文中仅引用... |
| [GPU 执行、内存模型与 Online Softmax](../ai/kernels/gpu-execution-and-memory-model.md) | [[gpu-execution-and-memory-model]] | `#ai` `#llm` `#gpu` `#cuda` `#roofline` `#softmax` `#flashinfer` | 大模型推理服务的延迟与吞吐最终由 GPU 内核的执行效率决定。理解 FlashInfer 为什么这样切分任务（GQA 打包、Split-KV、Ragged 批处理）、为什么使用 128-bit... |
| [从 CPU SIMD 到 GPU SIMT：并行计算底层模型与向量化编程直觉](../ai/kernels/simd-and-gpu-simt-foundations.md) | [[simd-and-gpu-simt-foundations]] | `#ai` `#llm` `#gpu` `#cuda` `#simd` `#simt` `#tensor-core` | 几乎所有刚接触 CUDA 的开发者都会产生同一个困惑：为什么 GPU 上明明有几万个线程并行，写高性能算子时还要讲究连续内存 Stride、还要让每个线程用 float4 一次搬 4 个数、还要... |

### ai/post-training

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [后训练：改哪些参数，用什么信号改](../ai/post-training/README.md) | — | — | 先做LoRA 参数量实验：预测秩翻倍时，有多少参数要更新。再用双臂奖励实验看另一件事：即使参数个数不变，换一个奖励信号也会改变策略。 |
| [用两台老虎机跑通一次策略梯度](../ai/post-training/bandit-policy-gradient/README.md) | — | — | 两台机器给出奖励的概率分别为 0.20 和 0.80。策略起初各选一半，从头训练后，比较它选第二台机器的概率和最后 500 次的平均奖励。 |
| [给一个投影层算清 LoRA 参数量](../ai/post-training/lora-parameter-count/README.md) | — | — | 一个 1024 × 1024 的投影有 1,048,576 个权重。冻结它，训练 rank 16 的两块矩阵，需要 32,768 个参数。运行脚本，看看 rank 翻倍以后哪项开销也翻倍。 |
| [LLM Post-training](../ai/post-training/llm-post-training.md) | [[llm-post-training]] | `#ai` `#llm` `#post-training` `#sft` `#dpo` `#rlhf` | Post-training 指 Pre-training 之后，为了让模型更好地遵循指令、符合偏好、使用工具或完成特定任务而进行的一组训练过程。它不是单一算法，常见方法包括 SFT、Prefer... |
| [LLM 强化学习基础](../ai/post-training/llm-rl-fundamentals.md) | [[llm-rl-fundamentals]] | `#ai` `#llm` `#reinforcement-learning` `#ppo` `#grpo` `#rlhf` | 强化学习（Reinforcement Learning，RL）研究 RL Agent 如何通过与 Environment 交互，根据 Reward 改进 Policy。这里的 RL Agent... |
| [LoRA 与 QLoRA](../ai/post-training/lora-and-qlora.md) | [[lora-and-qlora]] | `#ai` `#llm` `#lora` `#qlora` `#peft` `#fine-tuning` | LoRA（Low-Rank Adaptation）是一种 Parameter-Efficient Fine-Tuning（PEFT）方法。它不改变训练目标，而是改变“哪些参数需要更新”：冻结 B... |

### ai/training

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [从一次权重更新到多机训练](../ai/training/README.md) | — | — | 做训练平台，先用小模型回答一个具体问题：多开一个进程之后，还是不是在训练同一个目标 |
| [中断后续训：检查样本顺序和权重](../ai/training/checkpoint-replay/README.md) | — | — | 训练重启后，loss 仍会继续变化。用一次从头到尾的训练作对照，才能看出恢复时有没有重复读样本，或者丢掉优化器的动量。 |
| [用两个 DDP 进程复核同一次权重更新](../ai/training/ddp-step/README.md) | — | — | 先做单进程梯度对照。这里换成真正的进程和通信，每个 rank 都保留一个整批参考模型，以便逐权重比较。 |
| [把同一批数据分成 1∶3 和 2∶2](../ai/training/sharded-gradient/README.md) | — | — | 四个有效位置共同决定一次权重更新。各分片先算自己的平均梯度，再用两种办法聚合。把它们与整批反向传播对照，可以发现一个小分片被放大了多少。 |
| [分布式训练框架：从 DDP 基线到模型并行](../ai/training/distributed-training-frameworks.md) | [[distributed-training-frameworks]] | `#ai` `#分布式训练` `#pytorch` `#fsdp` `#deepspeed` `#megatron` `#nccl` | 先判断瓶颈发生在权重、梯度、优化器状态、激活还是通信等待。DDP 和 FSDP2 是 PyTorch 的并行实现；ZeRO 是 DeepSpeed 中的一组分片策略；Megatron Core... |

## Cloud Native & Kubernetes

### cloud-native

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [让训练作业拿到 GPU](../cloud-native/README.md) | — | — | 先给一个 64 卡作业找位置。跑完静态布局，再跑队列回放：短作业能否用掉暂时空闲的节点，又不推迟队首的大作业？只需 CPU，不会访问集群。 |

### cloud-native/containers

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [容器：从一个训练镜像开始](../cloud-native/containers/README.md) | — | — | 先用 Docker 常用命令对照一个已有镜像和由它启动的容器，分别找出镜像 ID、进程和挂载点。没有本地环境时先读 Docker 基础，不必拉训练模型练手。 |
| [Buildah 大镜像处理](../cloud-native/containers/buildah-large-image.md) | [[buildah-large-image]] | `#docker` `#buildah` | 在项目中，客户训练完大模型后打包成镜像（约 20G），需要通过平台上传镜像。原有版本上传功能受制于入口控制器及上层负载均衡的配置，上传大镜像会因超时失败，严重阻塞客户的正常开发流程。 |
| [Cgroup 资源限制机制](../cloud-native/containers/cgroup.md) | [[cgroup]] | `#docker` `#cgroup` | Cgroup (Control Group) 是 Linux 下用于对一个或一组进程进行资源控制和监控的机制： |
| [Docker 概述](../cloud-native/containers/docker-basics.md) | [[docker-basics]] | `#docker` | Docker 是基于 Linux 内核的 cgroup、namespace 以及 union-fs 等技术，对进程进行封装隔离的操作系统层面虚拟化技术。由于隔离的进程独立于宿主和其它隔离进程，因... |
| [Docker 常用命令速查](../cloud-native/containers/docker-commands.md) | [[docker-commands]] | `#docker` `#docker常用` | Q: docker run 和 docker start 有什么区别？docker create 又是做什么的 |
| [Dockerfile 最佳实践](../cloud-native/containers/dockerfile.md) | [[dockerfile]] | `#docker` `#dockerfile` | 越靠近底层变更频率越低，cache 命中率越高。 |
| [Namespace 视图隔离机制](../cloud-native/containers/namespace.md) | [[namespace]] | `#docker` `#namespace` | Linux Namespace 是内核提供的资源隔离机制，使得不同进程组看到不同的系统资源视图。Docker 利用 Namespace 实现容器间的隔离。 |
| [Docker Bridge 网络模式](../cloud-native/containers/network-bridge.md) | [[network-bridge]] | `#docker` `#docker网络` | Bridge 是 Docker 默认的网络模式。Docker 在每台主机上创建一个名为 docker0 的网桥，通过 veth pair 连接每个容器。 |
| [Docker Null 网络模式](../cloud-native/containers/network-null.md) | [[network-null]] | `#docker` `#docker网络` | Null 模式 (--net=none) 是一个空实现 -- 容器拥有独立的 Network Namespace，但 Docker 不做任何网络配置。 |
| [Docker Underlay 网络模式](../cloud-native/containers/network-underlay.md) | [[network-underlay]] | `#docker` `#docker网络` | Underlay 网络模式直接使用宿主机的物理网络，为每个容器分配可路由的网络 IP，容器与宿主机处于同一网络平面。 |
| [UnionFS 联合文件系统](../cloud-native/containers/union-fs.md) | [[union-fs]] | `#docker` `#unionfs` | Union FS（联合文件系统）是一种将不同目录挂载到同一个虚拟文件系统下的文件系统 (unite several directories into a single virtual files... |

### cloud-native/kubernetes

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [Kubernetes：训练作业停在哪一步](../cloud-native/kubernetes/README.md) | — | — | 有一条作业不动了。先找到最后一个已经发生的事件，从它的下一步开始查： |

### cloud-native/kubernetes/core

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [从一次 ConfigMap 变更跟到控制器](../cloud-native/kubernetes/core/README.md) | — | — | 先做最小控制器：改一个 ConfigMap，找出日志里同名对象被处理的记录。再做 Operator 对照：这一次，控制器还要真的更新镜像 ConfigMap，而不是只打印日志。 |
| [让 Operator 把 ConfigMap 的改动同步到副本](../cloud-native/kubernetes/core/kubebuilder-operator/README.md) | — | — | 给源 ConfigMap 加一条 annotation，控制器会创建带 ownerReference 的 mirror。再手动改 mirror，看控制器是否把值改回源对象的值。 |
| [从三次写入看懂 etcd revision](../cloud-native/kubernetes/core/raftexample-walkthrough/etcd-client-demo/README.md) | — | — | 给同一个键依次写入 v1、v2、v3，然后读取第一次写入的 revision。另一个键绑定 5 秒租约，用最后一次 Get 检查是否过期。 |
| [用 client-go 跟踪一次 ConfigMap 变更](../cloud-native/kubernetes/core/sample-controller/README.md) | — | — | 在一个终端观察控制器日志，在另一个终端新建、修改、删除同一个 ConfigMap。沿着同一个 namespace/name，看事件如何经过缓存和工作队列。 |
| [Admission Webhook](../cloud-native/kubernetes/core/admission-webhook-component.md) | [[admission-webhook-component]] | `#kubernetes` `#component` `#extension` `#admission` | Admission Webhook 是 apiserver 写入路径上的扩展点，用于在对象持久化前做默认值注入、对象改写和不变量校验。它分为 MutatingAdmissionWebhook 和... |
| [Kubernetes API Resource 设计文档](../cloud-native/kubernetes/core/api-resource.md) | [[api-resource]] | `#kubernetes` `#api-resource` | 目前版本迭代需要对不同集群下的api-resource资源进行分类 |
| [client-go 源码导读](../cloud-native/kubernetes/core/client-go-source.md) | [[client-go-source]] | `#kubernetes` `#client-go` `#源码导读` | client-go 是 Kubernetes 官方提供的 Go 语言客户端库，是编写 Controller、Operator 的基石。它向上提供 RESTClient/Clientset/Dyn... |
| [cloud-controller-manager](../cloud-native/kubernetes/core/cloud-controller-manager-component.md) | [[cloud-controller-manager-component]] | `#kubernetes` `#component` `#control-plane` `#cloud` | cloud-controller-manager 把 Kubernetes 核心控制面和云厂商 API 解耦。它运行云相关控制器，让 Kubernetes 主仓不再内置每个云厂商的实现细节。 |
| [controller-runtime 源码导读](../cloud-native/kubernetes/core/controller-runtime-source.md) | [[controller-runtime-source]] | `#kubernetes` `#controller-runtime` `#operator` `#源码导读` | controller-runtime（sigs.k8s.io/controller-runtime）是 Kubernetes SIG 维护的 Operator 开发底层运行时库，kubebuil... |
| [etcd](../cloud-native/kubernetes/core/etcd-component.md) | [[etcd-component]] | `#kubernetes` `#component` `#control-plane` `#etcd` | etcd 是 Kubernetes 的强一致键值存储，保存所有持久化 API 对象。Kubernetes 不把 Pod 进程状态直接放进 etcd，而是保存对象的期望状态、观测状态和元数据。 |
| [etcd 源码导读](../cloud-native/kubernetes/core/etcd-source.md) | [[etcd-source]] | `#etcd` `#raft` `#源码导读` | 本篇是对 etcd（etcd-io/etcd，以 v3.5 / v3.6 主干为参照）的源码导读，目标是从「能读懂」的角度梳理 etcd 的内部结构。etcd 在架构上是一个分层系统：最上层是... |
| [etcd 分布式键值存储](../cloud-native/kubernetes/core/etcd.md) | [[etcd]] | `#etcd` | Etcd 是 CoreOS 基于 Raft 开发的分布式 key-value 存储，可用于服务发现、共享配置以及一致性保障（如数据库选主、分布式锁等）。 |
| [Google Borg](../cloud-native/kubernetes/core/google-borg.md) | [[google-borg]] | `#borg` | Borg 是 Google 内部的大规模集群管理系统，Kubernetes 的前身。它管理着 Google 数万台服务器上的数十万个 Job，是 Google 基础设施的核心。 |
| [Kubernetes Informer List-Watch 机制](../cloud-native/kubernetes/core/informer.md) | [[informer]] | `#kubernetes` `#informer` | Informer 是 Kubernetes client-go 提供的 List+Watch 资源监听与本地缓存机制，用于实时感知集群中对象的增删改事件并触发回调。它内部包含 Reflector... |
| [kube-apiserver](../cloud-native/kubernetes/core/kube-apiserver-component.md) | [[kube-apiserver-component]] | `#kubernetes` `#component` `#control-plane` `#apiserver` | kube-apiserver 是 Kubernetes 的唯一 API 入口。所有组件和用户都通过它读写对象；它负责认证、鉴权、准入控制、schema 校验、对象持久化、watch 分发和聚合... |
| [kube-controller-manager](../cloud-native/kubernetes/core/kube-controller-manager-component.md) | [[kube-controller-manager-component]] | `#kubernetes` `#component` `#control-plane` `#controller` | kube-controller-manager 运行 Kubernetes 内置控制器。控制器的共同模式是 watch 对象变化，然后 reconcile 期望状态和实际状态。 |
| [Kubebuilder Operator 最小骨架 Demo](../cloud-native/kubernetes/core/kubebuilder-operator/demo-kubebuilder-operator.md) | [[demo-kubebuilder-operator]] | `#kubernetes` `#controller-runtime` `#operator` `#demo` | cloud-native/kubernetes/core/kubebuilder-operator/ 下的最小 Operator 骨架，用于配合 controller-runtime-sourc... |
| [Kubebuilder 自定义控制器开发](../cloud-native/kubernetes/core/kubebuilder.md) | [[kubebuilder]] | `#kubernetes` `#kubebuilder` | 在 Kubernetes 中，部署一个 Deployment 时，通常还需要创建一个 Service 以便外部可以访问这个 Deployment。为了简化这个过程，可以创建一个自定义的 Cont... |
| [Kubernetes 基础架构](../cloud-native/kubernetes/core/kubernetes-basics.md) | [[kubernetes-basics]] | `#kubernetes` | Kubernetes 是谷歌开源的容器集群管理系统，是 Google 多年大规模容器管理技术 Borg 的开源版本，主要功能包括: |
| [Kubernetes Dashboard](../cloud-native/kubernetes/core/kubernetes-dashboard-component.md) | [[kubernetes-dashboard-component]] | `#kubernetes` `#component` `#addon` `#dashboard` | Kubernetes Dashboard 是历史上的官方 Web UI addon，用于通过浏览器查看和管理集群资源。它本质上是运行在集群内的普通应用，通过 kube-apiserver 读写资源。 |
| [Logging Agent](../cloud-native/kubernetes/core/logging-agent-component.md) | [[logging-agent-component]] | `#kubernetes` `#component` `#addon` `#logging` | Logging Agent 是集群日志采集组件，常以 DaemonSet 运行在每个节点上，把容器 stdout/stderr、节点日志或系统日志采集到后端日志系统。常见实现包括 Fluent... |
| [Metrics Server](../cloud-native/kubernetes/core/metrics-server-component.md) | [[metrics-server-component]] | `#kubernetes` `#component` `#addon` `#metrics` | metrics-server 是 Kubernetes 常见资源指标 addon。它从 kubelet Summary API 或 metrics endpoint 拉取 Pod/Node CP... |
| [Kubernetes Operator 模式](../cloud-native/kubernetes/core/operator-pattern.md) | [[operator-pattern]] | `#kubernetes` `#operator` `#crd` | Operator 是 Kubernetes 的一种扩展模式，核心思想是 CRD + Custom Controller：用 CRD 声明"期望状态"，用 Controller 驱动"实际状态"向... |
| [etcd Raftexample 共识机制走读 Demo](../cloud-native/kubernetes/core/raftexample-walkthrough/demo-raftexample-walkthrough.md) | [[demo-raftexample-walkthrough]] | `#etcd` `#raft` `#demo` | 围绕 etcd-source 里的概念做的两个配套教学件，不是一个完整 raft 集群实现——etcd 的 raft 是一个独立库，自己拼一个完整集群代码量太大；这里换两条路把核心概念落到「能跑... |
| [etcd Raftexample 状态机与核心源码笔记](../cloud-native/kubernetes/core/raftexample-walkthrough/raftexample-notes.md) | [[raftexample-notes]] | `#etcd` `#raft` `#demo` | contrib/raftexample 是 etcd 仓库里官方维护的「raft 库最小可运行示例」，几百行代码完整演示了上层应用如何接线 raft：propose、WAL、snapshot、H... |
| [Kubernetes RBAC 权限控制](../cloud-native/kubernetes/core/rbac.md) | [[rbac]] | `#kubernetes` `#rbac` | 基于角色（Role）的访问控制（RBAC）是一种基于组织中用户的角色来调节控制对计算机或网络资源的访问的方法。 |
| [Kubernetes RESTful API 设计](../cloud-native/kubernetes/core/restful-api-design.md) | [[restful-api-design]] | `#kubernetes` `#restful` | Kubernetes 的 RESTful API 是与集群交互的核心接口，通过不同的路由操作资源（如 Pod、Service、Deployment 等）。所有 kubectl 命令最终都会转化为... |
| [Sample Controller 源码走读与可运行 Demo](../cloud-native/kubernetes/core/sample-controller/demo-sample-controller.md) | [[demo-sample-controller]] | `#kubernetes` `#client-go` `#demo` | 这是一个最小化、可直接运行的 client-go 控制器示例，监听集群中所有 ConfigMap 资源，每次变更只在日志里打印一条 reconciled <ns>/<name>。代码不到 200... |

### cloud-native/kubernetes/interview

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [用一次真实排障复习 Kubernetes](../cloud-native/kubernetes/interview/README.md) | — | — | 挑一个做过的实验，合上笔记，只说三件事：看见了什么、凭哪条证据排除了什么、下一步该查谁。说不清楚时，再展开题目里的参考答案。 |
| [概述](../cloud-native/kubernetes/interview/k8s-gpu-ai-infra-interview-modules.md) | [[k8s-gpu-ai-infra-interview-modules]] | `#kubernetes` `#gpu` `#ai-infra` `#面试` `#系统设计` | 这是一份面向 K8s / GPU / AI Infra 岗位的模块化面试题库。它不按单场面试组织，而是按长期稳定的知识域组织：每次新的面试材料都拆成问题，追加到对应模块。 |
| [Kubernetes 面试题汇总](../cloud-native/kubernetes/interview/k8s-interview.md) | [[k8s-interview]] | `#kubernetes` `#interview` | K8s是kubernetes的简称，其本质是一个开源的容器编排系统，主要用于管理容器化的应用， |

### cloud-native/kubernetes/networking

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [定位 Pod 网络断在哪一跳](../cloud-native/kubernetes/networking/README.md) | — | — | 先做 bridge 实验。对同一网络命名空间执行 ADD、DEL，分别看接口、地址和路由；不要用一次 ping 推断整个集群网络正常。 |
| [用 Linux 网桥连通两个模拟 Pod](../cloud-native/kubernetes/networking/cni-bridge/README.md) | — | — | 在两个网络命名空间内各放一端 veth，让它们通过同一座 cni0 网桥通信。读 ADD 返回的 IP，核对第二个命名空间实际接收了第一个发来的包。 |
| [Calico 网络插件原理](../cloud-native/kubernetes/networking/calico.md) | [[calico]] | `#kubernetes` `#cni` `#calico` | Calico 是一个强大的 CNI 插件，不仅给 Pod 分配 IP，还支持跨节点通信、网络策略、安全隔离。 |
| [Cilium eBPF 深度解析](../cloud-native/kubernetes/networking/cilium-deep-dive.md) | [[cilium-deep-dive]] | `#kubernetes` `#cni` `#cilium` `#ebpf` | Cilium 是基于 eBPF 的 Kubernetes 网络、Service 转发、安全策略与可观测性方案。它仍然是 CNI 插件，但和传统 bridge/iptables 插件的差异在于：它... |
| [Cilium eBPF 容器网络](../cloud-native/kubernetes/networking/cilium.md) | [[cilium]] | `#kubernetes` `#cni` `#cilium` `#ebpf` | Cilium 是新一代 CNI 插件，基于 eBPF（extended Berkeley Packet Filter）技术，在 Linux 内核中直接处理网络逻辑，避免了 iptables 的性... |
| [CNI Bridge 插件可运行 Demo](../cloud-native/kubernetes/networking/cni-bridge/demo-cni-bridge.md) | [[demo-cni-bridge]] | `#kubernetes` `#cni` `#demo` | 本 demo 用 100 行 bash 实现一个符合 CNI 0.4.0 规范的 bridge 插件 —— 跟 containernetworking-plugins 仓库的真实 bridge... |
| [CNI Plugin](../cloud-native/kubernetes/networking/cni-plugin-component.md) | [[cni-plugin-component]] | `#kubernetes` `#component` `#networking` `#cni` | CNI Plugin 负责把 Pod sandbox 接入网络，包括创建 veth、分配 Pod IP、配置路由和执行网络策略等。Kubernetes 通过 CRI runtime 间接触发 C... |
| [CNI 插件调用链源码导读](../cloud-native/kubernetes/networking/cni-source.md) | [[cni-source]] | `#kubernetes` `#cni` `#源码导读` | 本篇是 Container Network Interface（CNI） 的源码导读笔记，覆盖三个层面：(1) 上游 CNI 规范本身（github.com/containernetworkin... |
| [CNI 网络排障指南](../cloud-native/kubernetes/networking/cni-troubleshooting.md) | [[cni-troubleshooting]] | `#kubernetes` `#cni` `#troubleshooting` | CNI 排障不要从“插件坏了”开始猜。正确做法是先把链路拆成四段： |
| [CNI 容器网络接口](../cloud-native/kubernetes/networking/cni.md) | [[cni]] | `#kubernetes` `#cni` | CNI（Container Network Interface）是一种标准接口，用于为容器（Pod）分配 IP、配置路由、连接网络。 |
| [CoreDNS](../cloud-native/kubernetes/networking/coredns-component.md) | [[coredns-component]] | `#kubernetes` `#component` `#addon` `#dns` | CoreDNS 是 Kubernetes 默认 DNS addon。它 watch Service 和 EndpointSlice，为 Pod 提供集群内服务发现能力，例如解析 kubernet... |
| [Flannel 容器网络插件](../cloud-native/kubernetes/networking/flannel.md) | [[flannel]] | `#kubernetes` `#cni` `#flannel` | Flannel 是最简单的 CNI 插件之一，由 CoreOS 开发，专注于为 Pod 提供 overlay 网络。 |
| [Headless Service 无头服务](../cloud-native/kubernetes/networking/headless-service.md) | [[headless-service]] | `#kubernetes` `#service` | 在 Kubernetes 中，Headless Service 是一种特殊的服务类型，它的 clusterIP 设置为 None，没有虚拟的 IP 地址。它不会通过一个统一的入口来访问应用程序，... |
| [Ingress Controller](../cloud-native/kubernetes/networking/ingress-controller-component.md) | [[ingress-controller-component]] | `#kubernetes` `#component` `#networking` `#ingress` | Ingress Controller 负责把 Kubernetes Ingress 资源转成真实的七层流量入口配置。常见实现包括 ingress-nginx、Traefik、HAProxy、云厂... |
| [kube-proxy](../cloud-native/kubernetes/networking/kube-proxy-component.md) | [[kube-proxy-component]] | `#kubernetes` `#component` `#node` `#networking` `#service` | kube-proxy 把 Kubernetes Service 抽象落到每个节点的数据面规则里。它 watch Service 和 EndpointSlice，然后维护 iptables/IPV... |
| [kube-proxy 服务转发与代理模式](../cloud-native/kubernetes/networking/kube-proxy.md) | [[kube-proxy]] | `#kubernetes` `#networking` `#service` | kube-proxy 负责把 Kubernetes Service 这个逻辑抽象落到每个 Node 的数据面规则里。它不创建 Pod 网卡，也不分配 Pod IP；这些属于 CNI。它做的是监听... |
| [Multus 多网卡 CNI 插件](../cloud-native/kubernetes/networking/multus.md) | [[multus]] | `#kubernetes` `#cni` `#multus` | Multus 不是传统意义的 CNI 插件，而是一个 meta-plugin（元插件），它允许一个 Pod 同时拥有多个网络接口（多网卡）。 |
| [Kubernetes 网络模型](../cloud-native/kubernetes/networking/network-model.md) | [[network-model]] | `#kubernetes` `#networking` | Kubernetes 对集群网络有一套清晰的设计约束，称为 Kubernetes Network Model。所有 CNI 插件都必须满足以下三大原则： |
| [Kubernetes Service 服务暴露](../cloud-native/kubernetes/networking/service.md) | [[service]] | `#service` | Service 是将多个 Pod 划分到同一个逻辑组中，并统一向外提供服务的抽象。Pod 通过 Label Selector 加入到指定的 Service 中。 |
| [Weave Net 容器网络](../cloud-native/kubernetes/networking/weave.md) | [[weave]] | `#kubernetes` `#cni` `#weave` | Weave Net 由 Weaveworks 开发，特点是部署极简且内置加密通信能力。 |

### cloud-native/kubernetes/node-runtime

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [Pod 已经绑定，容器还没起来](../cloud-native/kubernetes/node-runtime/README.md) | — | — | 先在本地用 fake CRI 查询 Version，再创建一个模拟 sandbox。能连通 socket、能登记一个 sandbox、能运行真实容器，是三件不同的事。 |
| [用 crictl 跟踪一个模拟 Pod sandbox](../cloud-native/kubernetes/node-runtime/fake-cri/README.md) | — | — | 启动只在本地 Unix socket 上监听的 CRI 服务，创建一个 sandbox，再观察它从 Ready 到移除。整个过程只读写进程内存，无需 kubelet 或 Kubernetes 集群。 |
| [Container Runtime](../cloud-native/kubernetes/node-runtime/container-runtime-component.md) | [[container-runtime-component]] | `#kubernetes` `#component` `#node` `#cri` `#runtime` | Container runtime 是节点上真正管理容器生命周期的组件，常见实现是 containerd 和 CRI-O。kubelet 不直接 fork 容器进程，而是通过 CRI 调用 ru... |
| [CRI 容器运行时接口源码导读](../cloud-native/kubernetes/node-runtime/cri-source.md) | [[cri-source]] | `#kubernetes` `#cri` `#containerd` `#源码导读` | CRI（Container Runtime Interface） 是 kubelet 与底层容器运行时（containerd / CRI-O / 历史上还有 dockershim）之间的 gRP... |
| [Fake CRI Runtime Server 可运行 Demo](../cloud-native/kubernetes/node-runtime/fake-cri/demo-fake-cri.md) | [[demo-fake-cri]] | `#kubernetes` `#cri` `#kubelet` `#demo` | 本 demo 在 cri-source 第九节落地为一个最小可运行的 fake CRI runtime server。它在 /tmp/fake-cri.sock 上监听 gRPC，只实现 CRI... |
| [kubelet](../cloud-native/kubernetes/node-runtime/kubelet-component.md) | [[kubelet-component]] | `#kubernetes` `#component` `#node` `#kubelet` | kubelet 是每个 Node 上的核心 agent。它 watch 绑定到本节点的 Pod，然后通过 CRI、CNI、CSI、Device Plugin 等接口把 Pod 真正运行起来，并持... |
| [Kubelet 与 CRI 源码导读](../cloud-native/kubernetes/node-runtime/kubelet-cri-source.md) | [[kubelet-cri-source]] | `#kubernetes` `#kubelet` `#cri` `#device-plugin` `#源码导读` | kubelet 是运行在每个 Node 上的核心 agent，负责把 API Server 中 spec.nodeName 指向本机的 Pod 真正拉起并维持在期望状态。它本身不直接操作容器，而... |
| [OCI 容器运行时规范（runc）](../cloud-native/kubernetes/node-runtime/oci-runtime.md) | [[oci-runtime]] | `#kubernetes` `#oci` `#container-runtime` | OCI 是 Linux Foundation 下的一个开放治理项目，旨在制定容器行业标准。2015 年由 Docker、CoreOS 等公司共同发起，主要包含两个核心规范： |
| [Pod 健康检查探针（Probes）](../cloud-native/kubernetes/node-runtime/probes.md) | [[probes]] | `#kubernetes` `#probe` | 探针（Probe）是由 kubelet 对容器执行的定期诊断，用来判断容器的运行状态、是否可对外服务、是否完成启动。kubelet 通过探针的返回结果决定是否重启容器、是否将 Pod 从 Ser... |

### cloud-native/kubernetes/roadmaps

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [选一条 Kubernetes 实践路线](../cloud-native/kubernetes/roadmaps/README.md) | — | — | 目标是训练平台：先做调度与恢复，交付一份作业从排队到第一步训练的记录。只有当一张物理卡要给不同工作负载共享时，再走 HAMi 路线。 |
| [Kubernetes 组件覆盖矩阵](../cloud-native/kubernetes/roadmaps/component-coverage-matrix.md) | [[component-coverage-matrix]] | `#kubernetes` `#components` `#matrix` | 这张表用于确认 各子系统的组件笔记 是否覆盖完整。标准分两层： |
| [Kubernetes 组件笔记标准](../cloud-native/kubernetes/roadmaps/component-note-standard.md) | [[component-note-standard]] | `#kubernetes` `#components` `#standard` | 本标准用于 各子系统的组件笔记 下所有 Kubernetes 组件笔记。目标不是“列概念”，而是让每篇组件笔记都能回答四件事： |
| [Kubernetes 组件拆解](../cloud-native/kubernetes/roadmaps/k8s-component-map.md) | [[k8s-component-map]] | `#kubernetes` `#components` `#learning-notes` | 这张地图把分属各子系统的 Kubernetes 常见组件按「每个组件一个文件」拆开。范围分三层： |
| [Kubernetes 开发学习路线](../cloud-native/kubernetes/roadmaps/k8s-development-roadmap.md) | [[k8s-development-roadmap]] | `#kubernetes` `#学习计划` `#源码导读` `#导航` | 这是 Kubernetes 开发方向的总路线。先通过下面的决策图选择路径，再进入对应的主题、源码和 Demo，不需要从头到尾机械阅读。 |
| [Kubernetes 实验环境与历史验证](../cloud-native/kubernetes/roadmaps/k8s-lab-environments.md) | [[k8s-lab-environments]] | `#kubernetes` `#实验` | 2026-05 历史验证环境：Mac (Darwin 25.2.0) + Go 1.26 |
| [Kubernetes 源码导读](../cloud-native/kubernetes/roadmaps/k8s-source-map.md) | [[k8s-source-map]] | `#kubernetes` `#源码导读` | 源码导读负责固定版本下的调用链、关键结构与验证方式；概念和选型返回对应主题目录。 |
| [Kubernetes 深入学习进度](../cloud-native/kubernetes/roadmaps/progress.md) | [[progress]] | `#kubernetes` `#学习计划` `#打卡` | 下面的「Week 1 ~ Week 6」是逻辑顺序，不是日历周。「6 周」对应的是全职脱产、每天 4-6 小时的下限投入。按你的实际情况换算： |

### cloud-native/kubernetes/scheduling

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [GPU 调度：卡在哪里，作业在等谁](../cloud-native/kubernetes/scheduling/README.md) | — | — | 先跑放置与回填实验。先预测 64 张空闲卡能放几个整组作业，再换掉排队策略，分别看大、小作业各等了多久。 |
| [让 kubelet 上报 8 个设备，再给 Pod 分配 2 个](../cloud-native/kubernetes/scheduling/device-plugin/README.md) | — | — | 从 Node 的 Capacity、Allocatable 一直查到容器里的 FAKE_VISIBLE_DEVICES 和挂载目录。资源已经上报、Pod 已经拿到设备，是两次不同的观察。 |
| [跟踪一张模拟 GPU 的 UUID 从节点进入容器](../cloud-native/kubernetes/scheduling/fake-gpu/README.md) | — | — | 插件上报 4 个 GPU 风格的设备 ID。消费 Pod 声明一个资源，查看它收到的 UUID，并和插件 Allocate 日志核对。 |
| [把 4 张模拟卡上报为 40 个 vGPU 切片](../cloud-native/kubernetes/scheduling/hami-mac/README.md) | — | — | 向 kubelet 上报每卡 10 个切片，启动两个各申请一份的 Pod。对照物理卡标识、切片 ID 和显存配额，区分资源计数、设备选择和容器实际能使用的显存。 |
| [用进程内存看懂配额如何生效](../cloud-native/kubernetes/scheduling/hami-mac/libvgpu-hook-demo/README.md) | — | — | 示例进程反复申请 1 MiB。Linux 动态库从环境变量读取配额并累计分配量，超限时返回 NULL。这个小程序让上层插件写进容器的配额有了可观察的结果。 |
| [64 张空闲 GPU，能启动几个训练作业](../cloud-native/kubernetes/scheduling/placement-lab/README.md) | — | — | 每个作业有 8 个 worker，每个 worker 必须在同一台机器拿齐 8 张卡。下面两个实验分别计算能放多少个完整作业，以及排队时谁先开跑。只需 Python 3.9+，从仓库根目录执行。 |
| [用两个节点验证调度器的打分插件](../cloud-native/kubernetes/scheduling/scheduler-plugin/README.md) | — | — | 给两个 worker 打相反的标签，让第二个调度器安排 6 个 Pod。交换标签后创建新 Pod，比较两轮落点。已经运行的 Pod 不会因为标签变化自行迁移。 |
| [Device Plugin](../cloud-native/kubernetes/scheduling/device-plugin-component.md) | [[device-plugin-component]] | `#kubernetes` `#component` `#node` `#device-plugin` `#gpu` | Device Plugin 让 GPU、RDMA、FPGA、SmartNIC 等硬件以 Kubernetes extended resource 的形式暴露给 Pod。插件在节点上向 kubel... |
| [Device Plugin 设备插件可运行 Demo](../cloud-native/kubernetes/scheduling/device-plugin/demo-device-plugin.md) | [[demo-device-plugin]] | `#kubernetes` `#device-plugin` `#demo` | 本 demo 在 kubelet-cri-source 的「Device Plugin 机制」节落地为一个可运行项目。它实现一个虚拟的设备插件，向 kubelet 宣告 8 个名为 learni... |
| [Fake GPU Device Plugin 可运行 Demo](../cloud-native/kubernetes/scheduling/fake-gpu/demo-fake-gpu.md) | [[demo-fake-gpu]] | `#kubernetes` `#gpu` `#device-plugin` `#demo` | 本 demo 在 gpu-scheduling-source 的「手写简化复现」节落地为一个可运行项目。它实现一个 GPU 风格的虚拟设备插件，向 kubelet 宣告 4 个名为 fake-g... |
| [Kubernetes GPU 调度源码导读](../cloud-native/kubernetes/scheduling/gpu-scheduling-source.md) | [[gpu-scheduling-source]] | `#kubernetes` `#gpu` `#scheduler` `#device-plugin` `#ai-infra` `#源码导读` | 本篇是 GPU 调度的端到端源码导读，串联 scheduler-framework-source 的扩展点流水线和 kubelet-cri-source 的 Device Plugin 框架，专... |
| [Kubernetes GPU Scheduling（GPU 调度与资源管理）](../cloud-native/kubernetes/scheduling/gpu-scheduling.md) | [[gpu-scheduling]] | `#kubernetes` `#gpu` `#ai-infra` | Kubernetes 本身不直接感知 GPU 硬件，而是通过 Extended Resource 机制将 GPU 抽象为可调度的资源。整体架构涉及三个层次： |
| [HAMi 学习路线](../cloud-native/kubernetes/scheduling/hami-learning-path.md) | [[hami-learning-path]] | `#kubernetes` `#gpu` `#hami` `#ai-infra` `#学习计划` | HAMi（Heterogeneous AI computing Virtualization middleware，CNCF Sandbox 项目）是 K8s 上的 GPU 虚拟化与共享中间件，... |
| [Mac 上复现 HAMi Device Plugin 可运行 Demo](../cloud-native/kubernetes/scheduling/hami-mac/demo-hami-mac.md) | [[demo-hami-mac]] | `#kubernetes` `#gpu` `#hami` `#device-plugin` `#demo` | 本 demo 在 demo-fake-gpu 的基础上升级，专门用来在 Mac（无 NVIDIA GPU）+ kind 上复现 HAMi device-plugin 的两件关键事： |
| [以 HAMi 为主线串通 Kubernetes 核心机制](../cloud-native/kubernetes/scheduling/hami-mac/learn-k8s-via-hami.md) | [[learn-k8s-via-hami]] | `#kubernetes` `#gpu` `#hami` `#学习计划` `#总结` | 学 Kubernetes 最容易踩的坑：对着官方文档背概念，看完一脸"原来如此"，过两周全忘。原因是 K8s 的每个机制（APIServer / Informer / Scheduler / k... |
| [HAMi GPU 虚拟化源码导读](../cloud-native/kubernetes/scheduling/hami-source.md) | [[hami-source]] | `#kubernetes` `#gpu` `#hami` `#ai-infra` `#源码导读` | 本篇是 HAMi（Heterogeneous AI computing Virtualization middleware，CNCF Sandbox） 的源码导读笔记，主线是回答一个问题：一块物... |
| [kube-scheduler](../cloud-native/kubernetes/scheduling/kube-scheduler-component.md) | [[kube-scheduler-component]] | `#kubernetes` `#component` `#control-plane` `#scheduler` | kube-scheduler 负责给未绑定的 Pod 选择一个合适的 Node。它不创建容器、不分配 Pod IP，也不挂载卷；它的核心输出是把 Pod 绑定到某个 Node。 |
| [Kube-Scheduler 深度解析](../cloud-native/kubernetes/scheduling/scheduler-deep-dive.md) | [[scheduler-deep-dive]] | `#kubernetes` `#scheduler` | kube-scheduler 是 Kubernetes control plane 的核心组件之一，负责将未调度的 Pod（即 spec.nodeName 为空的 Pod）分配到合适的 Node 上。 |
| [Kube-Scheduler Framework 源码导读](../cloud-native/kubernetes/scheduling/scheduler-framework-source.md) | [[scheduler-framework-source]] | `#kubernetes` `#scheduler` `#源码导读` | 源码基线：本地 kubernetes/kubernetes commit c59911870d6（v1.37.0-beta.0 之后的开发版本），复核日期 2026-08-13。路径、签名和默认... |
| [Kube-Scheduler Score Plugin 扩展 Demo](../cloud-native/kubernetes/scheduling/scheduler-plugin/demo-scheduler-plugin.md) | [[demo-scheduler-plugin]] | `#kubernetes` `#scheduler` `#demo` | 本笔记是 cloud-native/kubernetes/scheduling/scheduler-plugin/ 这个 demo 的走读说明：用 out-of-tree 的方式给 kube-s... |
| [Kube-Scheduler PodGroup 与 Gang Scheduling 源码导读](../cloud-native/kubernetes/scheduling/scheduler-podgroup-source.md) | [[scheduler-podgroup-source]] | `#kubernetes` `#scheduler` `#podgroup` `#gang-scheduling` `#源码导读` | 本篇是 kube-scheduler PodGroup 调度的学习路径，基于 Kubernetes v1.36.1 源码。它适合接在 scheduler-framework-source 之后阅... |
| [分布式训练作业：从成组准入到故障恢复](../cloud-native/kubernetes/scheduling/training-job-placement.md) | [[training-job-placement]] | `#kubernetes` `#gpu` `#分布式训练` `#调度` `#ai-infra` | 平台交付的是一组可用且互联的设备，训练作业交付的是可持久恢复的优化步。两者之间至少有四个不同事件：队列准入、Pod 绑定、所有 rank 会合（rendezvous）、第一次优化器更新。拿到前一... |
| [Volcano 批调度器源码导读](../cloud-native/kubernetes/scheduling/volcano-source.md) | [[volcano-source]] | `#kubernetes` `#scheduler` `#volcano` `#ai-infra` `#源码导读` | 本篇是 Volcano 调度器的源码导读笔记，聚焦 github.com/volcano-sh/volcano 仓库下的 pkg/scheduler 和 pkg/controllers 两个目录... |

### cloud-native/kubernetes/storage

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [沿训练文件找到它挂载的卷](../cloud-native/kubernetes/storage/README.md) | — | — | 从 hostPath CSI 实验开始。在测试节点上，把一份 PVC 跟到 Pod 内的挂载目录。PVC 显示 Bound，只能说明申请已绑定，不能证明训练进程读到了完整 checkpoint。 |
| [用一个 PVC 追踪 CSI 从建卷到挂载](../cloud-native/kubernetes/storage/csi-hostpath/README.md) | — | — | 一个 Pod 申请 1 GiB，并向 /data 写入 marker。先看 PVC 是否绑定，再进入 Pod 读回 marker，分别核对控制器建卷和节点挂载。 |
| [Ceph CSI 存储驱动](../cloud-native/kubernetes/storage/ceph-csi.md) | [[ceph-csi]] | `#kubernetes` `#csi` `#ceph` | Ceph 是目前 Kubernetes 生态中最主流的分布式存储系统之一，提供块存储（RBD）、文件存储（CephFS）和对象存储（RGW）三种统一接口。ceph-csi 是 Ceph 官方维护... |
| [云厂商 CSI 存储驱动](../cloud-native/kubernetes/storage/cloud-provider-csi.md) | [[cloud-provider-csi]] | `#kubernetes` `#csi` `#cloud` | 各大云厂商都提供了原生的 CSI 驱动，与自家的云存储服务深度集成。使用云厂商 CSI 的优势是零运维（存储由云厂商管理）、与云平台深度集成（IAM、加密、快照等），缺点是厂商锁定。 |
| [CSI Driver](../cloud-native/kubernetes/storage/csi-driver-component.md) | [[csi-driver-component]] | `#kubernetes` `#component` `#storage` `#csi` | CSI Driver 负责把 Kubernetes PVC/PV 抽象对接到真实存储系统。它通过 CSI gRPC 接口实现卷创建、删除、attach、mount、扩容、快照等能力。 |
| [CSI HostPath 驱动可运行 Demo](../cloud-native/kubernetes/storage/csi-hostpath/demo-csi-hostpath.md) | [[demo-csi-hostpath]] | `#kubernetes` `#csi` `#demo` | 本笔记是 cloud-native/kubernetes/storage/csi-hostpath/ 这个 demo 的走读说明：用最少代码实现一个 hostPath 风格的 CSI drive... |
| [CSI Sidecar 容器协作机制](../cloud-native/kubernetes/storage/csi-sidecars.md) | [[csi-sidecars]] | `#kubernetes` `#csi` `#storage` | CSI 规范只定义插件要实现的 gRPC service：Identity、Controller、Node。Kubernetes 里的 sidecar 是 SIG Storage 提供的“翻译层... |
| [CSI 存储调用链源码导读](../cloud-native/kubernetes/storage/csi-source.md) | [[csi-source]] | `#kubernetes` `#csi` `#源码导读` | 本篇是 Container Storage Interface（CSI） 的源码导读笔记，覆盖两个层面：(1) 上游 CSI 规范本身（github.com/container-storage-... |
| [CSI 存储排障指南](../cloud-native/kubernetes/storage/csi-troubleshooting.md) | [[csi-troubleshooting]] | `#kubernetes` `#csi` `#storage` `#troubleshooting` | CSI 排障的关键是按生命周期定位：PVC 没绑定、VolumeAttachment 没 attached、Pod 没 mount、扩容没完成、快照没 ready，分别对应不同控制器和不同 CS... |
| [CSI 容器存储接口](../cloud-native/kubernetes/storage/csi.md) | [[csi]] | `#kubernetes` `#csi` | CSI（Container Storage Interface）是 CNCF 制定的一套标准存储插件接口规范，用于让 Kubernetes 和各种存储系统解耦。 |
| [Longhorn 分布式块存储](../cloud-native/kubernetes/storage/longhorn.md) | [[longhorn]] | `#kubernetes` `#csi` `#longhorn` | Longhorn 是 Rancher Labs 开源的轻量级云原生分布式块存储，专为 Kubernetes 设计，CNCF Incubating 项目。适合中小规模集群（节点 < 50），部署简... |
| [NFS CSI 存储驱动](../cloud-native/kubernetes/storage/nfs-csi.md) | [[nfs-csi]] | `#kubernetes` `#csi` `#nfs` | NFS 是最经典的网络文件共享协议，NFS CSI Driver 让 Kubernetes 可以通过 CSI 标准接口使用 NFS 存储。最大的优势是支持 ReadWriteMany（RWX），... |
| [OpenEBS 云原生存储](../cloud-native/kubernetes/storage/openebs.md) | [[openebs]] | `#kubernetes` `#csi` `#openebs` | OpenEBS 是 CNCF Graduated 项目，提供多种存储引擎以适配不同场景，是「云原生存储」理念的代表。其核心设计哲学是 "Storage is just another workl... |
| [Velero 集群备份与恢复](../cloud-native/kubernetes/storage/velero.md) | [[velero]] | `#kubernetes` `#velero` | Velero 是一个开源的 Kubernetes 集群备份和恢复工具，支持按需备份、定时备份以及集群迁移。 |
| [Kubernetes Volume 生命周期](../cloud-native/kubernetes/storage/volume-lifecycle.md) | [[volume-lifecycle]] | `#kubernetes` `#storage` `#csi` | Kubernetes volume 生命周期不是“PVC 创建后马上挂到 Pod”。完整路径跨越 apiserver、PV controller、scheduler、Attach/Detach... |

## Go & Runtime

### go

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [Go：控制器退出时，后台任务也退出了吗](../go/README.md) | — | — | 先在自己安装的 Go 1.26 标准库里运行两个测试。不需要集群，也不需要建模块： |

### go/fundamentals

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [Go 语言机制：从程序的行为入手](../go/fundamentals/README.md) | — | — | 先选一段自己写过的并发代码，在纸上标出：谁创建 goroutine，谁负责通知它退出，它可能阻塞在哪里。用 Context 和 Channel 核对答案，再上源码看具体唤醒路径。 |
| [Channel 底层实现](../go/fundamentals/channel.md) | [[channel]] | `#go` `#channel` `#concurrency` | Channel 在 runtime 中对应 hchan 结构体，位于 runtime/chan.go： |
| [Context 取消与传播](../go/fundamentals/context.md) | [[context]] | `#go` `#context` | context.Context 是 Go 1.7 引入标准库的核心同步工具，广泛用于控制 goroutine 生命周期、传递请求作用域数据和超时信号。标准库中 os/exec、net、datab... |
| [并发三色标记 GC](../go/fundamentals/gc.md) | [[gc]] | `#gc` | Go 语言采用 并发三色标记清除（Concurrent Tri-color Mark and Sweep） 算法进行垃圾回收。GC 与用户代码（赋值器 mutator）并发执行，通过写屏障（wr... |
| [GMP 调度模型](../go/fundamentals/gmp-model.md) | [[gmp-model]] | `#go底层实现` `#go` `#gmp调度模型` | Go 的 goroutine 调度基于 GMP 模型，是一种 M:N 调度方案——将 M 个 goroutine 映射到 N 个 OS 线程上执行。三个核心实体： |
| [Go 版本新特性演进](../go/fundamentals/go-versions.md) | [[go-versions]] | `#go` `#版本` | Go 1.18 引入了类型参数（Type Parameters），是 Go 语言最大的一次语法变化。 |
| [Interface 与 reflect](../go/fundamentals/interface.md) | [[interface]] | `#go` `#interface` `#reflect` | Go 中接口分为两种底层表示，位于 runtime/runtime2.go： |
| [Map 哈希表底层实现](../go/fundamentals/map-internals.md) | [[map-internals]] | `#map` | Go 的 map 是基于 hash table 实现的无序键值对集合。底层使用 拉链法（桶 + 溢出桶）解决哈希冲突，支持渐进式扩容以避免一次性搬迁带来的性能抖动。 |
| [P.runnext 调度优化](../go/fundamentals/p-runnext.md) | [[p-runnext]] | `#go` `#gmp调度模型` | P.runnext 是 Go 调度器中的一个重要优化：每个 P 上有一个特殊的 runnext 字段，存储下一个优先执行的 goroutine。它的优先级高于 P 的本地队列和全局队列。 |
| [Slice 底层实现与扩容](../go/fundamentals/slice.md) | [[slice]] | `#go` `#slice` | Slice 在 runtime 中对应 slice 结构体（对外暴露为 reflect.SliceHeader）： |

### go/internals

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [Go Runtime：从一条阻塞栈找源码](../go/internals/README.md) | — | — | 先从两个取消测试取一条明确行为，再沿 context/context.go 的取消路径回看：信号送到了哪个子节点，等待者又是在哪里被唤醒的？遇到生产问题，先拿栈或 profile，不要按源码目录... |
| [Channel 源码导读](../go/internals/go-channel-source.md) | [[go-channel-source]] | `#go` `#runtime` `#concurrency` | channel 是 Go 在语言层提供的同步与通信原语。源码里它不是“神秘队列”，而是一个带锁的 hchan： |
| [Context 源码导读](../go/internals/go-context-source.md) | [[go-context-source]] | `#go` `#concurrency` `#stdlib` | context.Context 不是 runtime 包的一部分，但它是 Go 服务端并发控制的事实标准：用一棵取消树把 request scope、deadline、cancel cause... |
| [Go GC 新旧实现对比](../go/internals/go-gc-old-vs-new.md) | [[go-gc-old-vs-new]] | `#go` `#runtime` `#gc` | 这里的“旧 GC / 新 GC”特指 Go 当前源码中的传统标记扫描路径和 Green Tea GC 标记扫描路径，不是说 Go 从“完全 STW GC”变成“并发 GC”的那段早期历史。 |
| [Go GC 源码导读](../go/internals/go-gc-source.md) | [[go-gc-source]] | `#go` `#runtime` `#gc` | Go 的 GC 主线是并发、非分代、非移动的 mark-sweep collector。它的目标不是让 STW 彻底消失，而是在可控 CPU 开销下，把 STW 控制在很短的阶段，并通过 pac... |
| [GMP 调度器源码导读](../go/internals/go-gmp-source.md) | [[go-gmp-source]] | `#go` `#runtime` `#scheduler` | GMP 是 Go runtime 把大量 goroutine 映射到少量 OS thread 的调度模型： |
| [Go Map 新旧实现对比](../go/internals/go-map-old-vs-new.md) | [[go-map-old-vs-new]] | `#go` `#runtime` `#map` | Go map 的语言语义没有因为底层实现切换而改变，但源码结构和性能模型已经发生明显变化： |
| [Map 源码导读](../go/internals/go-map-source.md) | [[go-map-source]] | `#go` `#runtime` `#map` | Go 1.24+ 的 map 实现已经不是传统 hmap/bmap/overflow bucket 主线，而是迁移到了 internal/runtime/maps：整体设计基于 Swiss Ta... |
| [Slice 源码导读](../go/internals/go-slice-source.md) | [[go-slice-source]] | `#go` `#runtime` `#memory` | slice 是 Go 中最常见的动态序列结构。它本身不是数组，而是一个三字段 header：指向底层数组的指针、长度、容量。理解 slice 的关键不是背 API，而是看清楚： |

## CS & Engineering Foundations

### foundations

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [工程基础：任务没启动，不一定是 GPU 不够](../foundations/README.md) | — | — | 一个训练流水线有三个步骤：准备数据、训练、写 checkpoint。假如有人把最后一步的完成事件误接为第一步的前置条件，所有步骤都会等。用 Python 标准库看正常和异常两条路径： |

### foundations/algorithm

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [算法：按手头的问题选工具](../foundations/algorithm/README.md) | — | — | 先做工程基础首页的依赖环实验，再读拓扑排序。把一条边反过来，画出再也不能降到零的入度。其余算法不用按课表通读。 |
| [Backtracking 回溯算法](../foundations/algorithm/backtracking.md) | [[backtracking]] | `#回溯` | 回溯算法是一种通过穷举所有可能来寻找解的算法，本质上是 DFS 的一种应用。核心思想：在每一步做选择，如果发现当前选择不能导向有效解，就撤销选择（回溯），尝试其他选择。 |
| [BFS 广度优先搜索](../foundations/algorithm/bfs.md) | [[bfs]] | `#广度优先` | 广度优先搜索（Breadth-First Search, BFS）是一种用于图或树的遍历算法，它从一个起始点开始，逐层遍历所有节点，即先访问离起始点最近的节点，再逐步向外扩展到更远的节点。BFS... |
| [Binary Search 二分查找](../foundations/algorithm/binary-search.md) | [[binary-search]] | `#二分查找` | 给一个有序数组和目标值，找第一次/最后一次/任何一次出现的索引，如果没有出现返回 -1。 |
| [位运算 AND 技巧 Bitwise AND](../foundations/algorithm/bitwise-and.md) | [[bitwise-and]] | `#位运算` | 利用 n & (n-1) 可以消除二进制中最低位的 1，这是很多位运算题目的基础操作。 |
| [DFS 深度优先搜索](../foundations/algorithm/dfs.md) | [[dfs]] | `#深度优先` | 深度优先搜索（DFS, Depth-First Search）是一种用于遍历或搜索 tree 或 graph 的算法。核心思想是尽可能深地向前探索，直到到达终点，然后通过回溯来探索其他分支。DF... |
| [差分数组 Difference Array](../foundations/algorithm/diff-array.md) | [[diff-array]] | `#差分数组` `#前缀和` | 差分数组是前缀和的逆运算。对差分数组求前缀和可以还原出原数组。常用于区间批量修改场景，将 O(N) 的区间加操作优化为 O(1)。 |
| [Dijkstra 最短路径](../foundations/algorithm/dijkstra.md) | [[dijkstra]] | `#算法` `#graph` `#shortest-path` | Dijkstra 算法是解决单源最短路径问题的经典算法，适用于边权非负的加权图。核心思想是贪心：每次从未确定最短路径的节点中选取距离最小的节点，并用它去更新邻居的距离。 |
| [图论基础与 BFS/DFS 应用](../foundations/algorithm/graph-bfs-dfs.md) | [[graph-bfs-dfs]] | `#算法` `#graph` | 图（Graph）是由顶点（Vertex）和边（Edge）组成的数据结构，用于表示对象之间的关系。图的遍历是许多图论算法的基础，BFS 和 DFS 是两种最核心的遍历方式。 |
| [Heap Sort 堆排序](../foundations/algorithm/heap-sort.md) | [[heap-sort]] | `#堆` | 堆（Heap）是一种特殊的完全二叉树（Complete Binary Tree），主要有两种类型： |
| [打家劫舍 House Robber](../foundations/algorithm/house-robber.md) | [[house-robber]] | `#动态规划` | 经典 DP 系列题目。核心约束：不能偷相邻的房屋。 |
| [KMP 字符串匹配](../foundations/algorithm/kmp.md) | [[kmp]] | `#算法` `#string` `#kmp` | KMP（Knuth-Morris-Pratt）算法是一种高效的字符串匹配算法。与暴力匹配 O(n*m) 不同，KMP 利用已匹配的信息避免回退文本指针，实现 O(n+m) 的时间复杂度。其核心是... |
| [LCA 最低公共祖先](../foundations/algorithm/lca.md) | [[lca]] | `#LCA` | LCA 问题是在 tree（特别是二叉树）中寻找两个节点最近的共同祖先。这类问题在生物信息学、网络路由、社交网络分析等领域有广泛应用。 |
| [LFU Cache 最不经常使用缓存](../foundations/algorithm/lfu.md) | [[lfu]] | `#LFU` `#缓存策略` | LFU 缓存淘汰策略根据数据被访问的频率来决定淘汰哪些数据，优先淘汰访问次数最少的。与 LRU 不同，LFU 关注的是访问频率而不是访问时间顺序。 |
| [链表 Linked List](../foundations/algorithm/linked-list.md) | [[linked-list]] | `#算法` `#链表` | 链表是一种线性数据结构，每个节点包含数据和指向下一个节点的指针。常用技巧包括 dummy head、快慢指针、递归等。 |
| [LRU Cache 最近最少使用缓存](../foundations/algorithm/lru.md) | [[lru]] | `#LRU` `#缓存策略` | LRU 缓存是一种常见的缓存替换策略，基于最近访问的时间来淘汰最近最少使用的条目。每次访问一个条目时，将其移动到最近使用的位置（链表头部）；当缓存满时，淘汰链表尾部（最久未使用）的条目。 |
| [Merge Sort 归并排序](../foundations/algorithm/merge-sort.md) | [[merge-sort]] | `#master算法` | 适用于所有子问题规模相同的递归：T(n) = a * T(n/b) + O(n^c) |
| [单调栈 Monotonic Stack](../foundations/algorithm/monotonic-stack.md) | [[monotonic-stack]] | `#单调栈` | 单调栈是一种特殊的 stack 数据结构，栈内元素始终保持单调递增或单调递减。常用于解决 Next Greater Element（下一个更大元素） 类问题。 |
| [前缀和 Prefix Sum](../foundations/algorithm/prefix-sum.md) | [[prefix-sum]] | `#前缀和` | 前缀和是一种预处理技术，通过预先计算数组前缀的累加和，可以在 O(1) 时间内求出任意区间的和。常与 HashMap 结合使用。 |
| [Quick Sort 快速排序](../foundations/algorithm/quick-sort.md) | [[quick-sort]] | `#荷兰旗问题` | 快速排序是一种 divide-and-conquer 算法，包含三个步骤： |
| [Radix Sort 基数排序](../foundations/algorithm/radix-sort.md) | [[radix-sort]] | `#基数排序` | 基数排序（Radix Sort）是一种非比较型整数排序算法，原理是将整数按位数切割成不同的数字，然后按每个位数分别排序。由于整数也可以表示字符串或浮点数，所以基数排序并不仅限于整数。 |
| [滑动窗口 Sliding Window](../foundations/algorithm/sliding-window.md) | [[sliding-window]] | `#滑动窗口` | 滑动窗口是一种利用双指针维护一个动态区间的技巧，常用于解决子串/子数组的最优化问题。 |
| [排序算法总结 Sorting Overview](../foundations/algorithm/sorting-overview.md) | [[sorting-overview]] | `#算法` `#总结` | 排序算法的稳定性是指：同样大小的样本在排序之后不会改变原始的相对次序。 |
| [Topological Sort 拓扑排序](../foundations/algorithm/topological-sort.md) | [[topological-sort]] | `#算法` `#graph` `#topological-sort` | 拓扑排序是对有向无环图（DAG）的所有顶点进行线性排序，使得对于每条边 (u, v)，u 在排序中出现在 v 之前。拓扑排序不唯一，一个 DAG 可能有多种合法的拓扑序。如果图中存在环，则不存在... |
| [Trie 前缀树 Prefix Tree](../foundations/algorithm/trie.md) | [[trie]] | `#算法` `#string` `#trie` | Trie（发音 "try"）是一种树形数据结构，用于高效存储和检索字符串集合中的键。每个节点代表一个字符，从根到某个节点的路径组成一个前缀。Trie 的核心优势是前缀查询 O(m)，m 为查询字... |
| [完全背包 Unbounded Knapsack](../foundations/algorithm/unbounded-knapsack.md) | [[unbounded-knapsack]] | `#动态规划` `#背包` | 完全背包问题中，每种物品可以选取无限次。与 0-1 背包的区别在于内层循环的遍历方向：完全背包从小到大遍历容量，0-1 背包从大到小。 |
| [Union-Find 并查集](../foundations/algorithm/union-find.md) | [[union-find]] | `#算法` `#union-find` | 并查集是一种用于管理元素分组的数据结构，支持两个核心操作：合并（Union） 两个集合和 查找（Find） 元素所属集合。通过路径压缩和按秩合并两种优化，可以将操作复杂度降到近似 O(1)。 |
| [XOR 异或运算](../foundations/algorithm/xor.md) | [[xor]] | `#位运算` | 其中第 1 条最重要，所有其他结论都可以由此推导。第 4 条相关的题目最多，利用区间上异或和的性质。 |

### foundations/database

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [数据库：一次查询到底慢在哪里](../foundations/database/README.md) | — | — | 从一个真实的过滤条件开始，比如“按作业 ID 查状态”。先看执行计划用了什么索引，再判断是读了太多行，还是在等另一个事务。不要一看到慢查询就先加缓存。 |
| [Elasticsearch 基础与架构](../foundations/database/elasticsearch-basics.md) | [[elasticsearch-basics]] | `#elasticsearch` | Elasticsearch 是一款基于 Lucene 的开源分布式搜索与分析引擎，提供近实时的全文检索、结构化搜索和数据分析能力。 |
| [Elasticsearch 字段类型与映射](../foundations/database/es-field-types.md) | [[es-field-types]] | `#elasticsearch` | Elasticsearch 支持丰富的字段类型，合理选择字段类型对查询性能和存储效率至关重要。 |
| [ES + Kibana Docker 部署](../foundations/database/es-kibana-docker.md) | [[es-kibana-docker]] | `#elasticsearch` `#docker` | 本文介绍使用 Docker Compose 快速搭建 Elasticsearch + Kibana 开发环境。 |
| [MySQL 存储引擎 (InnoDB vs MyISAM)](../foundations/database/mysql-engine.md) | [[mysql-engine]] | `#mysql` `#存储引擎` | MySQL 支持多种存储引擎，最常用的有 InnoDB、MyISAM 和 Memory。InnoDB 自 MySQL 5.5 起成为默认存储引擎。 |
| [MySQL 索引 (B+Tree/聚簇索引)](../foundations/database/mysql-index.md) | [[mysql-index]] | `#mysql索引` | MySQL 索引可以从四个维度进行分类： |
| [MySQL 锁机制 (Next-Key Lock)](../foundations/database/mysql-lock.md) | [[mysql-lock]] | `#mysql` `#lock` | 典型使用场景：全库逻辑备份。但更推荐使用 mysqldump --single-transaction，利用 MVCC 实现一致性备份而无需加全局锁（仅适用于 InnoDB）。 |
| [MySQL 事务与 MVCC](../foundations/database/mysql-transaction.md) | [[mysql-transaction]] | `#mysql` `#transaction` `#mvcc` | *InnoDB 在 RR 级别通过 Next-Key Lock 在很大程度上解决了幻读问题。 |
| [PostgreSQL 高级特性](../foundations/database/postgresql-advanced.md) | [[postgresql-advanced]] | `#postgresql` `#advanced` | Window Function（窗口函数）在不折叠行的情况下对分组/排序后的结果集进行计算，是复杂分析查询的核心工具。 |
| [PostgreSQL 基础与核心原理](../foundations/database/postgresql-basics.md) | [[postgresql-basics]] | `#postgresql` `#database` | PostgreSQL 使用多进程（multi-process）模型，每个客户端连接对应一个独立的 backend 进程；MySQL 使用多线程（multi-thread）模型。 |
| [Redis 缓存三大问题](../foundations/database/redis-cache.md) | [[redis-cache]] | `#redis` `#cache` | 请求的数据在缓存和数据库中都不存在，每次请求都穿过缓存直达数据库。 |
| [Redis 高可用 (主从/Sentinel/Cluster)](../foundations/database/redis-cluster.md) | [[redis-cluster]] | `#redis` `#cluster` `#sentinel` | Sentinel 是一个独立的进程，用于监控 Redis 主从集群，实现自动故障转移。 |
| [Redis 数据类型与底层结构](../foundations/database/redis-data-types.md) | [[redis-data-types]] | `#redis` | Redis 提供五种基本数据类型，每种类型底层有不同的数据结构实现： |
| [Redis 持久化 (RDB/AOF)](../foundations/database/redis-persistence.md) | [[redis-persistence]] | `#redis` `#persistence` | RDB 将某一时刻的全量内存数据以二进制快照的方式保存到磁盘（默认 dump.rdb）。 |

### foundations/middleware

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [消息系统：跟踪一条任务完成事件](../foundations/middleware/README.md) | — | — | 给事件一个 ID，在纸上标出生产者发出、服务端确认、消费者处理、业务结果写入四个时刻。只知道“消息发过了”，还不能断定任务状态已经更新。 |
| [Canal 数据同步中间件](../foundations/middleware/canal.md) | [[canal]] | `#canal` | 早期阿里巴巴因为杭州和美国双机房部署，存在跨机房同步的业务需求，实现方式主要是基于业务 trigger 获取增量变更。从 2010 年开始，业务逐步尝试数据库日志解析获取增量变更进行同步，由此衍... |
| [Kafka 集群参数配置](../foundations/middleware/cluster-config.md) | [[cluster-config]] | `#kafka` | Kafka 集群的稳定运行依赖于合理的参数配置。以下是 Broker、Producer、Consumer 三端的关键参数说明。 |
| [分布式事务方案](../foundations/middleware/distributed-transaction.md) | [[distributed-transaction]] | `#分布式事务` | 传统分布式事务使用 2PC（Two-Phase Commit），依赖于底层数据库的 XA 协议支持。MySQL InnoDB 支持 XA 协议，可以将 XA 理解为一个强一致的中心化原子提交协议。 |
| [Kafka 基础概念与设计理念](../foundations/middleware/kafka-basics.md) | [[kafka-basics]] | `#kafka` | Kafka 的设计目标是成为一个可以帮助大型公司应对各种实时数据流处理的通用平台。关键词：大型公司、实时、通用。对应到系统设计上的核心指标： |
| [Kafka 面试知识点](../foundations/middleware/kafka-interview.md) | [[kafka-interview]] | `#kafka` | Kafka 将消息以 Topic 为单位归纳。发布消息的程序称为 Producer，消费消息的程序称为 Consumer。集群由一个或多个 Broker 组成，Producer 通过网络将消息发... |
| [NATS 基础与核心概念](../foundations/middleware/nats-basics.md) | [[nats-basics]] | `#nats` `#messaging` `#middleware` | NATS 是一个开源的、云原生的消息系统，由 Synadia 维护，使用 Go 编写。它的设计哲学是 简单、高性能、轻量。 |
| [NATS JetStream 持久化与流处理](../foundations/middleware/nats-jetstream.md) | [[nats-jetstream]] | `#nats` `#jetstream` `#streaming` | JetStream 是 NATS 的内置持久化消息层，从 NATS Server 2.2 开始内置。它在 Core NATS 的基础上增加了： |
| [Kafka 消息压缩机制](../foundations/middleware/producer-compression.md) | [[producer-compression]] | `#kafka` | Kafka 的消息压缩可以在 Producer 端和 Broker 端进行，核心目的是节省磁盘空间和网络带宽，提高传输效率。 |
| [生产者消息分区机制原理](../foundations/middleware/producer-partition.md) | [[producer-partition]] | `#kafka` | Kafka 的消息组织方式是三级结构：Topic -> Partition -> Message。分区设计的核心目的： |
| [Kafka 消息零丢失保证](../foundations/middleware/zero-message-loss.md) | [[zero-message-loss]] | `#kafka` | Kafka 只对已提交（committed）的消息提供有限度的持久化保证。这包含两层含义： |

### foundations/tools

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [终端：找回命令，改完再执行](../foundations/tools/README.md) | — | — | 在 Zsh 的常见 Emacs 键位下，按 Ctrl-R 输入上次命令的一个关键词；找到后先检查参数和目标，不要直接回车。这比重新拼一条很长的日志查询更不容易改错环境。 |
| [Zsh / iTerm2 快捷键](../foundations/tools/zsh.md) | [[zsh]] | `#terminal` `#zsh` `#快捷键` | Zsh / iTerm2 常用快捷键 |

## Repository Docs

### docs

| 文档 | Wikilink | 标签 | 摘要 |
| --- | --- | --- | --- |
| [Learning Notes 信息架构](architecture.md) | [[architecture]] | — | 仓库只使用一个主分类轴：知识领域，顶层收拢为 AI / Cloud Native / Go / Foundations 四大支柱。内容的学习形式和深度作为领域内的次级视图，不再与领域目录平级竞争。 |
| [AI Infra 与系统术语速查](infra-glossary.md) | [[infra-glossary]] | — | 从首页的实验遇到陌生词时再查，不必预习整张表。 |
| [从一次实验写一条用得上的笔记](learning-by-doing.md) | [[learning-by-doing]] | — | 没有研究证明某种文件夹布局最适合所有人。这里要解决的是更具体的问题：隔几天回到一个陌生训练故障时，能不能复现现象、解释原因，再用不同输入检验解释。目录按系统归档，学习按任务前进；两者不必长成同一棵树。 |
