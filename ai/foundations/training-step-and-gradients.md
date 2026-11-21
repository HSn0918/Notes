#ai #机器学习 #线性代数 #分布式训练

相关笔记：[[linear-algebra-for-llm]] | [[llm-fundamentals]] | [[distributed-training-frameworks]]

# 一次参数更新：从矩阵梯度到有效 token

把语言模型先缩小成一个预测头：每个有效位置给出一行特征，模型要为下一 token 的真实类别打高分。训练系统无论使用一张卡还是多张卡，都应先说明自己在最小化哪一个标量。下面假设无类别权重、无标签平滑，每个有效位置权重相同。

## 从前向算到反向

设 `X` 为 `N × H`，`W` 为 `H × V`，`Z = XW` 为 `N × V`。`N` 是参与 loss 的位置数，`H` 是隐藏维度，`V` 是词表大小。对每一行 logits 做 softmax 得到概率 `P`。one-hot 标签矩阵 `Y` 用于推导；实现时只需真实类别的整数索引。

| 量 | 计算 | 理解与形状 |
| --- | --- | --- |
| loss | `L = -Σᵢ log P[i, yᵢ] / N` | 一个标量；对 logits 先减行最大值再算 log-sum-exp，可避免直接指数溢出 |
| logits 梯度 | `G = (P − Y) / N` | `N × V`；正确类的梯度非正，梯度下降会提高其相对分数 |
| 权重梯度 | `dW = XᵀG` | `H × V`；每个位置贡献一个外积，沿样本维相加，正是数据并行能切分的原因 |
| 输入梯度 | `dX = GWᵀ` | `N × H`；传回生成这些特征的上一层，不是优化器要更新的权重 |

**手算一行**：取 `X=[2,1]`、两类、`W=0`、正确类为 0。概率是 `[0.5, 0.5]`，loss 是 `ln 2`；`G=[-0.5, 0.5]`，`dW` 两行分别为 `[-1,1]`、`[-0.5,0.5]`。做一次 `W ← W − η·dW`（`η>0`），相同输入的第 0 类 logit 会超过第 1 类。注意 `dX` 在这个特定起点为零，即使 `dW` 并不为零。

## 从一批数据到一个训练循环

1. 先切分训练、验证和测试数据；文本按文档或来源分组后再切窗口，避免同一段文字同时出现在训练和验证里。训练集更新权重，验证集选择超参数，测试集留作最后评估。
2. 从训练集取 mini-batch。自回归训练将输入和目标错开一个 token；把 padding 和其他不参与监督的位置从**分子和分母**同时去掉。小批梯度是完整数据梯度的估计，两次更新的 loss 不必单调下降。
3. 前向算 loss、反向累积梯度、到更新边界再 `optimizer.step()` 并清梯度。SGD 直接沿负梯度走；AdamW 还需保存一阶、二阶矩，并单独做权重衰减。因此保存权重可以用于推理，却不足以等价续训。
4. 固定一个小批次检验能否拟合，再看未参与更新的验证 loss。训练 loss 下降但验证 loss 持续上升时，应检查过拟合、数据分布和切分；增加 GPU 不会修正错误标签或数据泄漏。

## 分片后究竟除以什么

若 rank `r` 有 `nᵣ` 个有效 token，本地平均梯度为 `gᵣ`，目标全局梯度是 `Σ(nᵣ·gᵣ) / Σnᵣ`。例如两个 rank 各有 1 和 3 个位置，平均梯度的某一分量分别为 2 和 6，正确答案是 5，不是 4。梯度累积窗口里各 micro-batch 长度不同时，同样要以整个窗口的有效 token 总数为分母。

在**默认按 DP 组大小 `D` 平均梯度的 DDP** 中，若 `N` 是全组整个更新窗口的有效 token 数，可令每个 rank 对自己的 **loss 之和**乘 `D/N` 后反向，最终平均恰好得到全局 token 均值梯度。不要让框架再重复除以累积步数。若本 rank 为零但全局非零，要提供可反传的零贡献并参与同一组 collective；若全局为零，所有 rank 一致跳过更新，不计算 `0/0`。

## 验证与参考

运行[不等长分片梯度实验](../training/sharded-gradient/README.md)：先用中心差分验证每个权重，再用 1:3 的分片对照同一完整 batch。它验证的是数值目标，而不是 DDP 或 NCCL 实现。

实现对照：[PyTorch 训练循环](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)、[CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)、[DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。换用类别加权、其他归约或自定义通信 hook 时需要重新推导分母。
