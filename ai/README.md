# 模型训练与推理

## 模型训练

刚开始补基础时，用[矩阵乘法](foundations/linear-algebra-for-llm.md)算出一个线性层的输出，再跟着[损失、梯度与参数更新](foundations/training-step-and-gradients.md)手算一个训练步。

已经会做一次更新，就做[分两份数据，算一次更新](training/sharded-gradient/README.md)。把 4 个样本位置分成 1 份和 3 份，看直接平均两边的梯度会错多少；再平分成 2 份和 2 份。只需要 CPU。

结果对上以后，做[双进程 DDP](training/ddp-step/README.md)，再[中断训练并恢复](training/checkpoint-replay/README.md)。在 GPU 上用 [CUDA 事件](kernels/gpu-execution-and-memory-model.md#训练侧练习区分提交时间与设备时间)量出单步耗时，再按[并行策略](training/distributed-training-frameworks.md)比较 DDP、FSDP2、ZeRO 和 Megatron Core。

[数学与模型基础](foundations/README.md) · [多机训练与 64 卡分组算例](training/README.md)

## 推理服务

先[测一次流式响应](inference/openai-compatible-benchmark/README.md)。首个数据包可能还没有文本，所以收到响应，不一定代表用户看到了第一个 token。没有 GPU 服务时，先做手册里的本地自测。

要解释服务表现，按问题分别查[请求调度与缓存](inference/README.md)、[CUDA 与 FlashInfer 算子](kernels/README.md)。要改变模型的回答方式，才进入[后训练](post-training/README.md)；要让模型调用工具，看 [Agent](agents/README.md)。

训练、调度和上线要一起做时，用[完整实战路线](ai-infra-fieldwork.md)串起来。

[返回首页](../README.md)
