# 用五个候选词观察温度和 top-p

同一组 logits 可以产生不同的下一词分布。先降低温度，再缩小 top-p，观察改变的是概率还是候选集合。

## 在 CPU 上运行

从仓库根目录执行。需要 Python 3.9+ 标准库。

```bash
python3 ai/inference/sampling/sampling_lab.py
```

## 读概率和候选集合

- `temperature=0.35` 的最大概率高于 `temperature=1.0`，分布更集中。
- `top_p=0.70` 的候选集合少于或等于完整分布，且累计概率达到阈值后不再保留低概率 token。
- 同一 `seed` 两次采样得到完全相同的 token 序列。
- 末尾出现 `SELF-CHECK: PASS`。

## 固定 seed 可以复现什么

实验中的 logits 是人为构造，不是模型输出。softmax 先将 logits 变为概率；温度越低，较大的 logit 被放大的相对优势越明显。nucleus (`top_p`) sampling 先按概率排序，仅在累计概率覆盖阈值的最小前缀内重新归一化后抽样。seed 只控制随机数流，不会让不同 logits 或不同采样参数得到相同结果。

## 复核结果

- [ ] 我能说明 `temperature` 不会修改模型权重。
- [ ] 我能解释 `top_p` 与 `top_k` 都是采样阶段的候选裁剪，但裁剪规则不同。
- [ ] 我知道固定 seed 保障的是同一实现、同一输入和同一参数下的可复现性。

## 验证记录

2026-08-13：已在本仓库 Python CPU 环境运行。该实验验证算法不变量，不是对任何模型服务随机数实现的兼容性测试。

[返回推理索引](../README.md)
