# 用两个 DDP 进程复核同一次权重更新

先做[单进程梯度对照](../sharded-gradient/README.md)。这里换成真正的进程和通信，每个 rank 都保留一个整批参考模型，以便逐权重比较。

## 先在 CPU 上用 Gloo

需要已有的 PyTorch distributed 环境，从仓库根目录执行。第一条建立单进程基线，第二条把四个有效位置分为 1 个和 3 个。

```bash
torchrun --standalone --nproc-per-node=1 ai/training/ddp-step/compare_step.py --device cpu
torchrun --standalone --nproc-per-node=2 ai/training/ddp-step/compare_step.py --device cpu
```

## 判断两个进程是否等价

检查 `ranks` 从 1 变成 2，两次都是 `tokens=4`，且 `max_weight_error < 1e-6`。脚本先求全局有效位置数，再缩放各自的 loss 之和，抵消 DDP 对各 rank 梯度取平均的影响。超出容差会报错。

## 在同一节点上换用 NCCL

需要已分配的两张 NVIDIA GPU 和支持 NCCL 的 PyTorch。保留相同输入与验收条件。

```bash
torchrun --standalone --nproc-per-node=2 ai/training/ddp-step/compare_step.py --device cuda
```

## 验证记录

2026-11-21 本仓库记录为 Python 语法检查。以上是运行时验收条件，尚无 Gloo 或 NCCL 实测记录。本例只执行一次 SGD 更新。

[并行训练框架](../distributed-training-frameworks.md) · [PyTorch DDP 教程](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
