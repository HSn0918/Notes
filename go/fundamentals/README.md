# Go 语言核心机制

> 返回 [Go 语言学习索引](../README.md)

本目录收录 Go 数据结构、语言演进、并发原语、调度模型与 GC 基础笔记；进入 Go 1.26 源码热路径与版本对照请前往 [Go Runtime 源码导读](../internals/README.md)。

## 阅读顺序

1. **数据结构与类型系统**：[Slice 底层实现与扩容](slice.md) · [Map 哈希表底层实现](map-internals.md) · [Interface 与 reflect](interface.md) · [Go 版本新特性演进](go-versions.md)
2. **并发控制原语**：[Context 取消与传播](context.md) · [Channel 底层实现](channel.md)
3. **调度与内存回收**：[GMP 调度模型](gmp-model.md) · [P.runnext 调度优化](p-runnext.md) · [并发三色标记 GC](gc.md)
