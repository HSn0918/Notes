# Go 语言学习索引

> ⬆ 返回 [知识库首页](../README.md)

本目录分为语言核心机制（[Go Fundamentals](fundamentals/README.md)）与 Go 1.26 Runtime 源码深入（[Go Internals](internals/README.md)）两层：前者聚焦语言语义、并发原语与内存模型，后者直击当前版本真实实现、热路径与线上排障。

## 推荐学习顺序

**入门**：Slice 底层与扩容 → Map 哈希表实现 → Interface 与反射 → Go 版本新特性

**进阶**：Context 控制与取消传播 → Channel 底层实现 → GMP 调度模型

**深入**：P.runnext 调度优化 → 并发三色标记 GC → Runtime 源码导读

## 笔记本目录

| 方向 | 入口 | 覆盖内容 |
| --- | --- | --- |
| 语言与运行机制 | [Go Fundamentals](fundamentals/README.md) | Slice、Map、Interface、Go 版本演进、Context、Channel、GMP、P.runnext、三色标记 GC（9 篇） |
| Runtime 源码导读 | [Go Runtime Internals](internals/README.md) | Go 1.26 源码入口（GMP / Channel / Context / Slice / Swiss Table Map / Green Tea GC） |

### 核心机制速览

| 笔记 | 简介 | 难度 |
| --- | --- | --- |
| [Slice 底层实现与扩容](fundamentals/slice.md) | slice 三字段结构、共享底层数组与 append 扩容陷阱 | 入门 |
| [Map 哈希表底层实现](fundamentals/map-internals.md) | hmap/bmap 结构、拉链法与渐进式扩容机制 | 进阶 |
| [Interface 与 reflect](fundamentals/interface.md) | iface/eface、itab 动态分发、类型断言与反射原理 | 进阶 |
| [Go 版本新特性演进](fundamentals/go-versions.md) | Go 1.18~1.23+ 泛型、迭代器、GOMEMLIMIT 与 PGO | 入门 |
| [Context 取消与传播](fundamentals/context.md) | Context 树形取消信号与超时传播 | 进阶 |
| [Channel 底层实现](fundamentals/channel.md) | hchan 环形缓冲区、等待队列与阻塞唤醒 | 进阶 |
| [GMP 调度模型](fundamentals/gmp-model.md) | G/M/P 三实体 M:N 调度与 work-stealing | 深入 |
| [P.runnext 调度优化](fundamentals/p-runnext.md) | 本地优先槽位与调度局部性优化 | 深入 |
| [并发三色标记 GC](fundamentals/gc.md) | 三色标记清除、混合写屏障与 STW 边界 | 深入 |

## 源码深入

完成上述基础后进入 [Go Runtime 源码学习索引](internals/README.md)，按 GMP → Channel/Context → Slice/Map → GC → 新旧实现对比推进。
