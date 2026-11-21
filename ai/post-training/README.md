# 后训练：改哪些参数，用什么信号改

先做[LoRA 参数量实验](lora-parameter-count/README.md)：预测秩翻倍时，有多少参数要更新。再用[双臂奖励实验](bandit-policy-gradient/README.md)看另一件事：即使参数个数不变，换一个奖励信号也会改变策略。

- 显存有限，希望少训练一些参数：读 [LoRA 与 QLoRA](lora-and-qlora.md)。
- 手头有示范回答、偏好对或可打分的环境：按数据类型看 [SFT、DPO 与在线训练](llm-post-training.md)。
- 已经能收集奖励，但不明白参数怎么更新：从[策略梯度](llm-rl-fundamentals.md)起步，再读 PPO 或 GRPO。

[返回 AI](../README.md)
