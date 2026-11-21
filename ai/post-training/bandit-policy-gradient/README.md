# 用两台老虎机跑通一次策略梯度

两台机器给出奖励的概率分别为 0.20 和 0.80。策略起初各选一半，从头训练后，比较它选第二台机器的概率和最后 500 次的平均奖励。

## 在 CPU 上运行

从仓库根目录执行。需要 Python 3.9+ 标准库。

```bash
python3 ai/post-training/bandit-policy-gradient/bandit_policy_gradient.py
```

## 读训练前后的概率

- 初始 policy 近似均匀。
- 训练后，reward 概率为 0.80 的 arm 概率明显高于 reward 概率为 0.20 的 arm。
- 输出训练前后概率与最近窗口平均 reward；末尾出现 `SELF-CHECK: PASS`。

## 从奖励到下一次选择

这里的 action 是选 arm，reward 是模拟的 0/1 回报。更新使用 `reward - baseline` 作为 advantage，令取得高于基线奖励的 action 概率上升。它对应 RL for LLM 的最小骨架，但并不包含语言模型、rollout 长序列、KL 约束、价值网络、PPO/GRPO、多样本 group advantage 或真实任务验证器。

## 复核结果

- [ ] 我能将 `arm → action`、Bernoulli 结果 → reward、logits → policy 对应到 LLM RL 术语。
- [ ] 我能说明 baseline 主要是降低方差，不是环境的真实奖励。
- [ ] 我不会把此收敛曲线当作 PPO/GRPO 或真实模型训练结论。

## 验证记录

2026-08-13：已在本仓库 Python CPU 环境运行。随机环境使用固定 seed；不同超参数和随机 seed 的收敛路径会不同。

[强化学习基础](../llm-rl-fundamentals.md) · [返回上一级](../README.md)
