#ai #分布式训练 #pytorch #fsdp #deepspeed #megatron #nccl

相关笔记：[[training-step-and-gradients]] | [[training-job-placement]] | [[gpu-execution-and-memory-model]]

# 分布式训练框架：从 DDP 基线到模型并行

先判断瓶颈发生在**权重、梯度、优化器状态、激活还是通信等待**。DDP 和 FSDP2 是 PyTorch 的并行实现；ZeRO 是 DeepSpeed 中的一组分片策略；Megatron Core 提供组合张量、流水线、数据等并行方式的训练组件。它们不负责给集群里的租户分配 GPU。

## 用同一模型比较

| 起点 | 每个 rank 保留什么 | 训练时交换什么 | 第一个验证点 |
| --- | --- | --- | --- |
| PyTorch DDP：单卡放得下，需要提高吞吐 | 完整模型、梯度和优化器状态；数据由训练程序分配 | 通常在反向时对梯度桶做 AllReduce，与剩余反向计算重叠 | 各 rank 输入不重复；固定全局 batch，看一步更新是否接近单卡结果 |
| PyTorch FSDP2：副本状态占满显存 | 在 DP 组内分片参数、梯度和优化器状态；计算某个分组时仍需临时完整参数 | AllGather 收集该组参数，ReduceScatter 聚合并分片梯度；是否在前向后立即重新分片影响下一次收集 | 同时记录常驻与峰值显存；分组太大时，临时完整参数仍可能 OOM |
| DeepSpeed ZeRO：按状态类别逐步分片 | Stage 1 分优化器状态；2 在此之上分梯度；3 再分参数 | Stage 3 计算时要收集所需参数；选择 offload 还要承担主机链路或存储开销 | 从最低够用的 stage 对照；记录峰值显存、传输等待和有效 tokens/s |
| Megatron Core：单个大层或深模型需要拆分计算 | TP 切层内张量；PP 把连续层分给不同 stage；仍可用 DP 复制这套模型 | TP 有层内 collective；PP 相邻 stage 传激活和反向梯度 | 对照 TP 组实际链路、PP 各段耗时及流水线气泡；不因卡多就把所有并行度调高 |

**显存账本例子**：仅作估算，假定每个参数常驻 BF16 权重 2 B、BF16 梯度 2 B、FP32 主权重 4 B、Adam 的两个 FP32 矩共 8 B。一份 10 亿参数副本即 `16 GB`（十进制），还没算激活、临时参数、通信缓冲和分配器。8 路完全分片时这几项**常驻状态**的理想均摊约 `2 GB/rank`，不是峰值显存。实现的梯度精度、主权重和 offload 配置不同时要重算。若峰值随序列长度陡增，先测激活；用 activation checkpointing 换重算，不等于把断点写入磁盘。

## 跟随一次通信

- **AllReduce**：每个 rank 手里是同形状的部分结果，结束后每个人拿到完整归约结果。DDP 典型用途是同步梯度。
- **ReduceScatter / AllGather**：前者归约后每人只保留不同切片，后者把各人的切片收成完整张量。辨认是哪种 collective，再按消息大小和拓扑分析时间。
- **从 trace 区分因果**：某个 rank 更晚读到数据，其他 rank 会在 collective 内等它；等待最长的 GPU 未必是慢源。要比较各 rank 进入同一步、同一个 collective 的时间，再看 NIC、链路或输入管线。

## 下一次真机实验

1. 用单卡可容纳的小模型留一份固定输入、初始权重和一步更新结果。先用 FP32，关闭 dropout 等随机源，记录允许的数值误差。按 [PyTorch DDP 入门示例](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)接入进程组，再扩为双卡，保持有效 token 和学习率不变。
2. 单节点跑通后，先在分配到的独占 GPU 上用 [NCCL Tests](https://github.com/NVIDIA/nccl-tests) 测同一种 collective 的小、大消息，再用相同消息范围测跨节点。区分 `algbw`、`busbw`、错误数；它们不是训练吞吐。
3. 对照完整 step 的稳态中位数和 P95、最大 rank 峰值显存、有效 tokens/s；只改变一个策略。保存框架版本、拓扑、精度、模型/数据 revision、序列长度以及 global batch。比较不同并行度前单独核验 checkpoint 是否支持该拓扑下恢复。

## 参考与版本边界

[PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) · [FSDP2 fully_shard](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) · [DeepSpeed ZeRO](https://deepspeed.readthedocs.io/en/stable/zero3.html) · [Megatron Core 并行指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)。本文对比机制；FSDP2 不沿用 FSDP1 包装器示例，各框架启动参数、可组合策略和 checkpoint 格式以所用版本为准。
