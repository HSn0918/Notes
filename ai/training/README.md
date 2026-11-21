# 从一次权重更新到多机训练

做训练平台，先用小模型回答一个具体问题：多开一个进程之后，还是不是在训练同一个目标？

1. **只有 CPU**：运行[不等长分片](sharded-gradient/README.md)。保持四个有效位置不变，对照直接平均与加权平均。
2. **已有 PyTorch**：做[双进程 DDP](ddp-step/README.md)。比较相同输入更新后的权重，而不只是两个进程都启动成功。
3. **已有 GPU**：先做[单卡计时](../kernels/gpu-execution-and-memory-model.md#训练侧练习区分提交时间与设备时间)，再按[显存与通信瓶颈选框架](distributed-training-frameworks.md)。不要同时换精度、全局 batch 和并行策略。
4. **准备上集群**：做[64 卡放置回放](../../cloud-native/kubernetes/scheduling/placement-lab/README.md)和[中断续训](checkpoint-replay/README.md)。核对实际 rank 布局、恢复后的样本和权重，再按[作业验收](../../cloud-native/kubernetes/scheduling/training-job-placement.md)量排队与重算。万卡是容量目标，不是起步门槛。

## 算例：64 张卡如何分组

假设一个稠密模型固定使用 8 个节点，每节点 8 GPU，一进程一卡；`TP=8, PP=2, DP=4`，不启用其他并行维。让同一层的 8 个张量并行 rank 留在同一节点，两台节点组成一个两段流水线副本，复制四份。`8 × 2 × 4 = 64` 是 **rank 数**；每个数据副本每次读 2 条序列、累积 8 次才更新，则全局 batch 是 `2 × 8 × 4 = 64` 条序列，而不是再乘 64 张卡。如果以 token 计进度，还要扣除 padding 和被 mask 的目标。

- **在训练侧**画出一个 TP 组、一对相邻 PP stage 和一个 DP 组；说明各组在传本层张量、激活/梯度，还是跨副本聚合梯度。具体通信随框架和并行策略变化。
- **在平台侧**给出两张表：哪些节点可被同一作业同时取得；启动后每个 rank 实际落在哪里。调度器绑定 Pod 不等于训练框架依照预期建立了进程组。
- **在测量侧**区分两只钟：提交到第一步的时间（含排队和启动），以及开始训练后的每步耗时。强扩展固定模型、有效全局 batch 和同一批 token；若从 `N₀` 卡的吞吐 `T₀` 扩到 `N` 卡的 `T`，相对效率为 `(T/T₀)/(N/N₀)`。另报从提交到完成的有效 tokens/s，剔除失败后重算的 token。

**一组可核对的 rank 编号**：教学用 `rank = (dp × 2 + pp) × 8 + tp`，三个坐标均从 0 开始，连续 8 个 rank 在同一节点。于是 rank 0 的 TP 组为 `0…7`，它向下一流水段的 rank `8` 发送激活；负责相同切片的 DP 组是 `{0,16,32,48}`。这套编号是本例的约定，不是框架默认值。真机先导出各进程组成员和节点名，再判断哪次通信走了跨节点链路。

## 训练完成后再做推理

做预训练时不需要先学推理引擎。要服务训练产物，或在线后训练需要生成 rollout 时，再进入 [FlashInfer × vLLM × SGLang 路线](../inference/flashinfer-vllm-sglang-learning-path.md)：先读 [FlashInfer 的页表与 Plan/Run](../kernels/flashinfer-intuition-and-engine-bridge.md)，再比较 [vLLM 与 SGLang](../inference/vllm-and-sglang.md)。交付一份[同权重、同请求集的基准](../inference/openai-compatible-benchmark/README.md)，分别报告首 token 延迟、输出吞吐和实际选中的 attention 后端。仓库客户端的首个流事件只是 TTFT 近似，需与服务侧指标区分。

[返回 AI](../README.md)
