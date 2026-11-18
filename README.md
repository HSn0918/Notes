# Learning Notes

这是一个聚焦 **AI Infra & Systems Full-Stack** 的个人技术学习笔记库，兼容 Obsidian 与 GitHub 阅读。内容以理解底层机制、源码导读和动手实验验证为核心，面试题只作为辅助复习视图。

## 从目标选入口

| 目标 | 入口 | 阅读顺序 |
| --- | --- | --- |
| **万卡集群调度与模型训练** | [AI 学习索引](ai/README.md) | 线代与机器学习 → 单卡 CUDA 与训练基线 → 多卡通信与训练框架 → 拓扑放置、扩展效率和故障恢复 |
| 从平台侧做 GPU 调度 | [云原生学习索引](cloud-native/README.md) | GPU 资源如何上报 → 作业如何成组拿卡 → 放置结果能否满足训练通信需求 |
| 优化大模型推理 | [AI 学习索引](ai/README.md) | CUDA 执行与访存 → FlashInfer 算子 → vLLM / SGLang 请求调度；这不是做训练的前置课 |
| 学 Go 或查运行时实现 | [Go 学习索引](go/README.md) | 语言机制 → Go Runtime 源码与验证 |
| 补计算机与后端基础 | [工程基石索引](foundations/README.md) | 按问题进入算法、数据库、中间件或工具 |

找一篇笔记用 [全库知识速查索引](docs/catalog.md)，跨目录找同一主题用 [标签索引](docs/tags.md)。路线图负责安排先后，索引不代表每篇都要读。

## 名词解释

只解释地图和表格里出现的词，每个词一句话大白话。更系统的定义在对应领域的笔记里。

### ① AI / LLM Systems

| 名词 | 一句话解释 |
| --- | --- |
| 线代 / 高代 | 大学数学课：线性代数管矩阵乘法与向量投影，高等代数管秩、正交旋转、特征值——本库只挑大模型用得上的部分 |
| CUDA | NVIDIA GPU 的编程平台，用它写代码让 GPU 并行干活 |
| rank / world size | rank 是进程在组内的编号，world size 是组的进程数；按一进程一卡启动时，全局进程数等于本作业使用的卡数 |
| DP / TP / PP | 数据并行分样本，张量并行切一层内的计算，流水线并行把不同层放在不同 stage；三者可组合 |
| SIMD / SIMT | SIMD 对向量的多个元素执行同一操作；NVIDIA GPU 以 32 个线程组成的 warp 调度，分支可能让部分线程不活跃 |
| 算子（Kernel） | 一个具体的 GPU 计算函数，比如"做一次 RMSNorm""算一次注意力" |
| FlashInfer | GPU 推理算子库；是否被推理引擎选为某个操作的后端，要看版本、硬件和配置 |
| KV Cache | 生成文本时缓存的"历史记忆"，避免每生成一个字就重算全部历史 |
| Prefill / Decode | Prefill = 一次性读完整个输入；Decode = 一个字一个字往外蹦 |
| Paged KV（分页） | 把 KV Cache 像操作系统管内存一样切成小块页来分配 |
| Prefix Cache（前缀缓存） | 多个请求开头相同时，复用已经算好的那部分 |
| Continuous Batching | 请求随到随插队、不等整批结束，让 GPU 一直吃饱 |
| PD 分离 | Prefill 和 Decode 放到不同机器上跑，互不拖累 |
| vLLM / SGLang | 两个主流推理引擎（服务进程），负责调度请求、管理显存、调用算子 |
| Post-training | 模型预训练完成之后的所有训练：微调、对齐、强化学习 |
| SFT | 监督微调：用示范问答数据教模型按指令行事 |
| LoRA | 冻结原权重，用一对低秩矩阵学习权重增量；节省比例取决于秩和目标层 |
| DPO / PPO / GRPO | 三种"按人类偏好调整模型"的训练方法（偏好对齐 / 强化学习） |
| Rollout | RL 训练中"让模型自己生成一批回答"的步骤，速度依赖推理引擎 |
| Agent | 让 LLM 调用工具、多步规划完成任务的系统；RAG = 先检索资料再回答 |

### ② Cloud Native

| 名词 | 一句话解释 |
| --- | --- |
| OCI / 容器 | OCI 是容器镜像的标准格式；容器 = 轻量级、可移植的隔离运行环境 |
| Kubernetes（K8s） | 管理成千上万个容器的"集群操作系统" |
| CNI / CSI | K8s 的网络插件标准 / 存储插件标准 |
| GPU 调度 / HAMi / DRA | 让 K8s 知道每张 GPU 被谁用了多少的机制（HAMi 是社区方案，DRA 是官方新框架） |

### ③ Go & Runtime

| 名词 | 一句话解释 |
| --- | --- |
| Runtime | Go 程序背后自带的运行时系统：管调度、内存、垃圾回收 |
| GMP | Go 并发调度的三层模型：G（协程）、M（系统线程）、P（调度处理器） |
| GC | 垃圾回收：自动释放不再使用的内存 |
| SwissMap | Go 1.24+ 的 map 全新底层实现（基于 Swiss Table，查找更快） |

### ④ Foundations

| 名词 | 一句话解释 |
| --- | --- |
| 中间件 | 独立于业务的基础软件，比如 Kafka（消息队列）、Redis（缓存） |
| 算法与数据结构 / 数据库 / 工具 | 后端工程师的通用基本功 |

## 快速检索

[全库知识速查索引](docs/catalog.md) · [全库标签聚合索引](docs/tags.md) · SQLite FTS5 秒级 CLI 检索（纯 Python 3 标准库，索引维护在 `.traces/notes_fts.sqlite3`）：

```bash
python3 scripts/search_notes.py "权重吸收"          # 中文关键词
python3 scripts/search_notes.py paged_kv_t --domain ai
python3 scripts/search_notes.py --tag flashinfer    # 按标签检索
python3 scripts/search_notes.py --rebuild-catalog   # 新增/移动笔记后重建索引
```

## 仓库约定

- 顶层只有 `ai/`、`cloud-native/`、`go/`、`foundations/` 四大支柱，每个知识文件只有一个 canonical 归属；路线、源码、Demo、题库属于领域内子目录。
- 普通笔记使用全局唯一的英文小写 `kebab-case` 文件名，必须包含一级 `# H1`；所有正式学习题都带默认折叠的参考答案。
- 新增、移动或修改笔记元数据后运行：`python3 scripts/search_notes.py --rebuild-catalog` 与 `python3 scripts/validate_notes.py`。

完整目录职责与写作规范见 [仓库架构](docs/architecture.md) 与 [AGENTS.md](AGENTS.md)。
