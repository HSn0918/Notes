# AI Infra 学习笔记

从单卡训练到万卡集群，要同时看模型怎样更新、GPU 怎样分配，以及跨机通信花了多少时间。

- **[万卡集群调度](cloud-native/README.md)**　从 64 卡作业排队入手，查清队列何时放行、卡分在哪些节点、每个 worker 最后拿到了什么。
- **[模型训练](ai/README.md#模型训练)**　从矩阵乘、损失和梯度开始，用一次权重更新检验 DDP，再按显存和通信瓶颈读 FSDP2、DeepSpeed ZeRO 和 Megatron。
- **[推理优化](ai/README.md#推理服务)**　先用一批请求测 vLLM 和 SGLang，再到 CUDA 时间线和 FlashInfer 的注意力算子里解释差异。

## 第一个实验：64 张空闲卡能放下几个作业

作业需要 8 个 worker，每个 worker 独占一台机器上的 8 张 GPU。比较两种布局：8 台各余 8 张，或 16 台各余 4 张。从仓库根目录运行，只需 Python 3.9+。

```bash
python3 cloud-native/kubernetes/scheduling/placement-lab/placement.py
```

看 `node_fit` 这一列：`packed` 能启动 1 个完整作业，`fragmented` 是 0。第二种布局的总卡数够了，却没有一台机器能容纳一个 worker。

## 写平台时查

[Go 语言与运行时](go/README.md) · [算法、数据库与中间件](foundations/README.md) · [全部笔记](docs/catalog.md) · [标签](docs/tags.md) · [术语](docs/infra-glossary.md) · [仓库结构](docs/architecture.md)
