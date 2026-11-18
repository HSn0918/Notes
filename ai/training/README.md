# 万卡集群调度与模型训练路线图

> 返回 [AI 学习索引](../README.md)

适合会 Python、能读 Linux 进程和 Kubernetes 作业状态，但还不能从一次参数更新推导出 GPU、网络和存储需求的工程师。目标是给出一份**可验证的训练作业设计**：申请多少卡、哪些 rank 要放在一起、训练进度如何计量、一台机器掉线后从哪里继续。万卡是目标集群的容量，不是入门实验的资源要求，也不等于每个作业都用一万卡。

## 阶段与完成标准

| 阶段 | 先读什么 | 动手产出与过关标准 |
| --- | --- | --- |
| 1 · 线代与机器学习（CPU） | [矩阵乘与张量布局](../foundations/linear-algebra-for-llm.md) → [一次参数更新](../foundations/training-step-and-gradients.md) | 写出 `XW`、`XᵀG`、`GWᵀ` 的形状；运行[梯度实验](../experiments/sharded-gradient/README.md)，用有限差分检验梯度，并解释为什么两个 rank 的平均 loss 不能总是直接平均 |
| 2 · 单卡 CUDA 与训练基线 | [GPU 执行与内存模型](../kernels/gpu-execution-and-memory-model.md) → [CUDA 工具链](../kernels/cuda-ecosystem-and-career-roadmap.md) | 在单卡固定模型、batch、序列长和精度，只改变一次数据供给方式；留下一条包含 forward、backward、optimizer 的时间线，区分 host 空等、数据拷贝和 kernel 时间 |
| 3 · 单机多卡训练 | [并行与框架选型](distributed-training-frameworks.md) | 同一小批数据先单卡再双卡，比较一步更新后的权重误差；记录 DDP 每卡峰值显存，预测改为分片后哪项会下降，再实测。不要同时换精度、模型和框架 |
| 4 · 跨节点通信与放置 | [从 GPU 申请到训练进程组](../../cloud-native/kubernetes/core/training-job-placement.md) → [Volcano 调度源码](../../cloud-native/kubernetes/internals/volcano-source.md) | 先核算下面的 64 卡例子；有环境后采集 rank→GPU→节点→网络域映射，做单机和双机 collective 基线，再用训练 step 的尾延迟验证放置假设 |
| 5 · 面向万卡容量做验收 | 同一篇[作业放置笔记](../../cloud-native/kubernetes/core/training-job-placement.md)中的排队与恢复演练 | 先用队列回放评估小作业、整机作业混跑时的等待与碎片；再用逐级真机试验测有效吞吐和恢复耗时。分别交付调度报告、训练扩展曲线和一次成功续训的记录 |

**只有 CPU 时**先做阶段 1、阶段 3 的分片数值对照和阶段 4 的纸面放置题。它们不能验证 CUDA 性能、NCCL 带宽、真实集群调度时延或万卡稳定性。**已有训练经验时**可从阶段 3 开始，但保留单卡结果作对照。

## 用同一个作业串起两条线

假设一个稠密模型固定使用 8 个节点，每节点 8 GPU，一进程一卡；`TP=8, PP=2, DP=4`，不启用其他并行维。让同一层的 8 个张量并行 rank 留在同一节点，两台节点组成一个两段流水线副本，复制四份。`8 × 2 × 4 = 64` 是 **rank 数**；每个数据副本每次读 2 条序列、累积 8 次才更新，则全局 batch 是 `2 × 8 × 4 = 64` 条序列，而不是再乘 64 张卡。如果以 token 计进度，还要扣除 padding 和被 mask 的目标。

- **在训练侧**画出一个 TP 组、一对相邻 PP stage 和一个 DP 组；说明各组在传本层张量、激活/梯度，还是跨副本聚合梯度。具体通信随框架和并行策略变化。
- **在平台侧**给出两张表：哪些节点可被同一作业同时取得；启动后每个 rank 实际落在哪里。调度器绑定 Pod 不等于训练框架依照预期建立了进程组。
- **在测量侧**区分两只钟：提交到第一步的时间（含排队和启动），以及开始训练后的每步耗时。强扩展固定模型、有效全局 batch 和同一批 token；若从 `N₀` 卡的吞吐 `T₀` 扩到 `N` 卡的 `T`，相对效率为 `(T/T₀)/(N/N₀)`。另报从提交到完成的有效 tokens/s，剔除失败后重算的 token。

**一组可核对的 rank 编号**：教学用 `rank = (dp × 2 + pp) × 8 + tp`，三个坐标均从 0 开始，连续 8 个 rank 在同一节点。于是 rank 0 的 TP 组为 `0…7`，它向下一流水段的 rank `8` 发送激活；负责相同切片的 DP 组是 `{0,16,32,48}`。这套编号是本例的约定，不是框架默认值。真机先导出各进程组成员和节点名，再判断哪次通信走了跨节点链路。

## 推理支线：什么时候学 FlashInfer、vLLM 和 SGLang

做预训练时不需要先学推理引擎。要服务训练产物，或在线后训练需要生成 rollout 时，再进入 [FlashInfer × vLLM × SGLang 路线](../inference/flashinfer-vllm-sglang-learning-path.md)：先读 [FlashInfer 的页表与 Plan/Run](../kernels/flashinfer-intuition-and-engine-bridge.md)，再比较 [vLLM 与 SGLang](../inference/vllm-and-sglang.md)。交付一份[同权重、同请求集的基准](../experiments/openai-compatible-benchmark/README.md)，分别报告首 token 延迟、输出吞吐和实际选中的 attention 后端。仓库客户端的首个流事件只是 TTFT 近似，需与服务侧指标区分。
