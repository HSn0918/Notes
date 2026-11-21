# 给一个投影层算清 LoRA 参数量

一个 1024 × 1024 的投影有 1,048,576 个权重。冻结它，训练 rank 16 的两块矩阵，需要 32,768 个参数。运行脚本，看看 rank 翻倍以后哪项开销也翻倍。

## 在 CPU 上运行

从仓库根目录执行。需要 Python 3.9+ 标准库。

```bash
python3 ai/post-training/lora-parameter-count/lora_parameter_count.py
```

## 核对 rank 与参数量

- 对 `1024 × 1024` 投影，rank 16 的 LoRA 参数量为 `2 × 1024 × 16`，远小于全量矩阵。
- rank 上升时 LoRA 参数量线性上升。只有满足 `r(d_in + d_out) < d_in × d_out` 时，LoRA 的可训练参数才少于全量矩阵；对于 `d × d` 方阵，条件是 `r < d / 2`。
- 末尾出现 `SELF-CHECK: PASS`。

## 把计算推广到其他投影

LoRA 用 `W + BA` 表示更新：冻结原始 `W`，训练形状为 `d_out × r` 和 `r × d_in` 的两个矩阵。LoRA 参数量为 `r(d_in + d_out)`，全量矩阵参数量为 `d_in × d_out`；“低秩”描述矩阵分解结构，不自动保证参数更少。这个实验只计算**投影权重**的可训练参数，不含 embedding、LayerNorm、bias、optimizer state、activation、量化存储或通信开销；因此不能用作显存或训练时间的精确预测。

## 复核结果

- [ ] 我能写出全量投影与 LoRA adapter 的参数量公式。
- [ ] 我能解释为什么 rank 是效果、可训练参数和成本之间的旋钮。
- [ ] 我知道 QLoRA 的 4-bit 权重存储不等于所有训练状态都是 4-bit。

## 验证记录

2026-08-13：已在本仓库 Python CPU 环境运行。这是数学规模估算，未加载或训练真实模型。

[LoRA 与 QLoRA](../lora-and-qlora.md) · [返回上一级](../README.md)
