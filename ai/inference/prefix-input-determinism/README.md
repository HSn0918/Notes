# 同一份 JSON 为什么会有两个前缀哈希

把相同字段按不同顺序写入两个 JSON。先比较序列化后的字节和 SHA-256，再比较按键名排序后的结果。

## 在 CPU 上运行

从仓库根目录执行。需要 Python 3.9+ 标准库。

```bash
python3 ai/inference/prefix-input-determinism/prefix_determinism.py
```

## 对照两组哈希

- 两个键值相同、插入顺序不同的普通 JSON 字节串不同，hash 也不同。
- `sort_keys=True` 后 canonical JSON 字节串和 hash 相同。
- 末尾出现 `SELF-CHECK: PASS`。

## 从字节到缓存命中

真实推理引擎按 tokenizer 产生的 Token IDs 与引擎 cache key 决定是否复用。本实验的 bytes/hash 仅是更低成本的前置代理：byte 不同必然不能称为“相同 Prompt”；byte 相同也不能单独证明某个服务会命中，因为 chat template、tokenizer、模型 revision、缓存设置和路由仍可不同。

## 复核结果

- [ ] 我能区分“对象语义相同”“JSON bytes 相同”和“Token IDs 相同”。
- [ ] 我能说明为什么把动态时间戳、随机 ID 放进公共前缀会破坏复用。
- [ ] 我不会把本实验的 hash 结果报告为 vLLM/SGLang 的 cache-hit 证据。

## 验证记录

2026-08-13：已在本仓库 Python CPU 环境运行。tokenizer 与 GPU 推理服务集成未在本实验中执行。

[前缀缓存排查](../prefill-cache-miss.md) · [返回上一级](../README.md)
