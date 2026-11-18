# 不等长分片的训练梯度实验

> 返回 [AI Experiments](../README.md)

相关笔记：[[training-step-and-gradients]] | [[distributed-training-frameworks]]

用四个有效位置训练一个两类线性预测头。把同一批位置分成 1:3 两份，各自反向，再与不分片的参考梯度比较。目标不是跑快，而是判断两种训练方式是否仍在优化同一个目标。

## 运行

在仓库根目录执行，只需 Python 3.9+ 标准库：

```bash
python3 ai/experiments/sharded-gradient/sharded_gradient.py
```

## 结果与边界

2026-11-18 在 macOS arm64、Python 3.9.6 的 CPU 上实测：

```text
finite_diff_error=8.139e-12 weighted_error=1.110e-16 naive_error=2.316e-01
train_loss=0.407862 -> 0.344479
SELF-CHECK: PASS
```

`finite_diff_error` 是逐权重比较解析梯度和中心差分的最大绝对误差；`weighted_error` 与 `naive_error` 分别比较按位置数加权和不加权的聚合。把本地均值简单平均会过度强调只有一个位置的分片。最后一行 loss 对比使用同一训练批，只验证这一步 SGD 的方向，不衡量泛化。

脚本在单个 CPU 进程内模拟归约，没有启动 PyTorch、CUDA、DDP 或 NCCL，不测试通信顺序、混合精度和故障恢复。
