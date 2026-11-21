# 把同一批数据分成 1∶3 和 2∶2

四个有效位置共同决定一次权重更新。各分片先算自己的平均梯度，再用两种办法聚合。把它们与整批反向传播对照，可以发现一个小分片被放大了多少。

## 先分成 1∶3，再平分

从仓库根目录运行，需要 Python 3.9+ 标准库。

```bash
python3 ai/training/sharded-gradient/sharded_gradient.py
python3 ai/training/sharded-gradient/sharded_gradient.py --left-tokens 2
```

## 比较三列误差

2026-11-21 在 macOS arm64、Python 3.9.6 的 CPU 上实测：

```text
finite_diff_error=8.139e-12 weighted_error=1.110e-16 naive_error=2.316e-01
train_loss=0.407862 -> 0.344479
tokens=1:3
SELF-CHECK: PASS
```

`finite_diff_error` 是逐权重比较解析梯度和中心差分的最大绝对误差；`weighted_error` 与 `naive_error` 分别比较按位置数加权和不加权的聚合。把本地均值简单平均会过度强调只有一个位置的分片。最后一行 loss 对比使用同一训练批，只验证这一步 SGD 的方向，不衡量泛化。

换成 2∶2 后，两个分片对总损失的贡献相同，`naive_error` 也应接近零。

## 下一步用进程替代分片模拟

这里用单个 Python 进程检验公式。多进程是否得到相同的权重更新，用[双进程 DDP](../ddp-step/README.md)复核。

[梯度分母](../../foundations/training-step-and-gradients.md) · [训练路线](../../ai-infra-fieldwork.md)
