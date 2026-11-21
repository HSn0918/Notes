# 中断后续训：检查样本顺序和权重

训练重启后，loss 仍会继续变化。用一次从头到尾的训练作对照，才能看出恢复时有没有重复读样本，或者丢掉优化器的动量。

## 在第 5 步保存

从仓库根目录执行。Python 3.9+ 标准库即可，脚本会自动清理临时 checkpoint。

```bash
python3 ai/training/checkpoint-replay/replay.py
```

四条样本循环使用。完整恢复时，下一条样本编号为 1，最终权重与对照一致。漏掉游标会从样本 0 开始；只漏动量时样本仍正确，但权重误差约为 0.224。

## 把中断点移到轮次边界

```bash
python3 ai/training/checkpoint-replay/replay.py --cut 4
```

这次第 4 步刚好用完一轮样本。漏存游标也会碰巧读到正确样本。用两次输出对照 `next_sample`、`same_order` 和 `weight_error`，单测边界容易漏掉这个错误。

## 验证范围

2026-11-21 在 macOS arm64 CPU 验证了两种中断位置。脚本读写标量模型的 JSON 状态。真实分布式续训还需检验分片是否全部提交，以及 RNG、学习率和数据加载器状态是否一致。

[集群恢复验收](../../../cloud-native/kubernetes/scheduling/training-job-placement.md) · [返回训练索引](../README.md)
