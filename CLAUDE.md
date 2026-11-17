# Learning Notes 项目说明

本仓库按四大知识支柱组织个人技术学习笔记，兼容 Obsidian 与 GitHub，全部采用纯文本 Markdown + Python 标准库维护，不包含任何前端依赖或 `node_modules`。

## 稳定目录职责

```text
ai/
  foundations/      LLM 基础与零基础路线
  kernels/          GPU 执行模型、CUDA 算子与 FlashInfer 源码导读
  inference/        推理链路、vLLM、SGLang 与进阶路线
  post-training/    LoRA、SFT、DPO、RLHF、PPO、GRPO
  agents/           Agent 路线、源码与生产化
  experiments/      可运行的采样、Prefix、LoRA、RL、Split-KV 与推理基准实验

cloud-native/
  containers/       Docker、OCI、Cgroup、Namespace、UnionFS 与容器网络
  kubernetes/
    core/           集群基础、etcd、API/RBAC/Informer、Probe/OCI、调度与 Operator
    networking/     CNI、Service、网络实现与排障
    storage/        CSI、Volume、驱动与排障
    components/     组件职责与故障边界
    internals/      源码调用链
    roadmaps/       学习路线与进度
    demos/          可运行代码
    interview/      复习题库

go/
  fundamentals/     Slice、Map、Interface、Context、Channel、GMP、GC 核心机制
  internals/        Go 1.26 Runtime 与标准库源码导读

foundations/
  algorithm/        搜索与图论、排序、常用技巧、DP、数据结构与字符串
  database/         MySQL、Redis、PostgreSQL、Elasticsearch
  middleware/       Kafka、NATS、Canal、分布式事务
  tools/            Zsh / iTerm2 等常用效率工具

docs/
  architecture.md   信息架构、迁移记录与文档类型边界
  catalog.md        脚本生成的全库 1-hop 知识速查索引
  tags.md           脚本生成的全库标签聚合索引

scripts/
  validate_notes.py       结构、H1 标题与问答格式自动校验
  search_notes.py         SQLite FTS5 全文检索（.traces/notes_fts.sqlite3）与索引生成工具
  test_validate_notes.py  校验脚本单元测试
```

## 快速检索工作流（1-hop）

- 查找现有笔记或去重前，优先读取 `docs/catalog.md` / `docs/tags.md`，或运行秒级 FTS5 检索：
  `python3 scripts/search_notes.py <keyword> [--domain ai] [--tag flashinfer] [--limit 15]`
- 新增、重命名或修改笔记元数据后，运行 `python3 scripts/search_notes.py --rebuild-catalog` 确定性重建 `docs/catalog.md`、`docs/tags.md` 与 FTS5 索引，再运行 `python3 scripts/validate_notes.py --quality`。

## 维护规则

- 详细写作、链接、MOC 和验证规则以 `AGENTS.md` 为准。
- 架构决策与文档类型边界见 `docs/architecture.md`。
- 不在本文件维护精确篇数或逐文件清单；全局清单由 `scripts/search_notes.py --rebuild-catalog` 维护在 `docs/catalog.md`。
- 结构验证命令：`python3 scripts/validate_notes.py --quality`。
