# Learning Notes 项目说明

本仓库按四大知识支柱组织个人技术学习笔记，兼容 Obsidian 与 GitHub，全部采用纯文本 Markdown + Python 标准库维护，不包含任何前端依赖或 `node_modules`。

## 稳定目录职责

AI、云原生、Go 和工程基础是知识归属，不是必修课表。实验在所属子系统内与 README、代码和数据一起维护；跨领域课程从 [AI Infra 实战路线](ai/ai-infra-fieldwork.md)串联，不复制正文。Kubernetes 的控制面、调度、节点运行时、网络和存储分别收录自己的组件、源码和 Demo。目录说明与旧入口迁移见[架构文档](docs/architecture.md)。

## 快速检索工作流（1-hop）

- 查找现有笔记或去重前，优先读取 `docs/catalog.md` / `docs/tags.md`，或运行秒级 FTS5 检索：
  `python3 scripts/search_notes.py <keyword> [--domain ai] [--tag flashinfer] [--limit 15]`
- 新增、重命名或修改笔记元数据后，运行 `python3 scripts/search_notes.py --rebuild-catalog` 确定性重建 `docs/catalog.md`、`docs/tags.md` 与 FTS5 索引，再运行 `python3 scripts/validate_notes.py --quality`。

## 维护规则

- 详细写作、链接、MOC 和验证规则以 `AGENTS.md` 为准。
- 架构决策与文档类型边界见 `docs/architecture.md`。
- 不在本文件维护精确篇数或逐文件清单；全局清单由 `scripts/search_notes.py --rebuild-catalog` 维护在 `docs/catalog.md`。
- 结构验证命令：`python3 scripts/validate_notes.py --quality`。
