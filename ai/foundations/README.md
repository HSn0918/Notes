# 模型基础：看懂一次矩阵乘法

从[分块矩阵乘实验](llm-math-and-simd-lab/README.md)开始。矩阵边长不能整除分块宽度时，最后一小块也要参与计算。先对比整块乘和分块乘的结果，再手算一个元素的位置。

**想看懂前向计算**：先看 [token 如何变成预测](llm-fundamentals.md)，再查[矩阵形状与步长](linear-algebra-for-llm.md)。转置视图改变了取数顺序，不一定搬动数据。

**想看懂训练**：沿[一次参数更新](training-step-and-gradients.md)看误差如何传回权重，用[不等长分片](../training/sharded-gradient/README.md)检查平均时的分母。

遇到 RoPE 或低秩分解时，再读[旋转、秩与奇异值](advanced-algebra-for-llm.md)。还分不清训练和生成，可从[大模型零基础路线](llm-learning-path.md)补起。

[返回 AI](../README.md)
