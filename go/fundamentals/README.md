# Go 语言机制：从程序的行为入手

先选一段自己写过的并发代码，在纸上标出：谁创建 goroutine，谁负责通知它退出，它可能阻塞在哪里。用 [Context](context.md) 和 [Channel](channel.md) 核对答案，再上[源码](../internals/README.md)看具体唤醒路径。

- **改了一个值，另一处也跟着变**：从 [Slice 共享底层数组](slice.md)、[Map](map-internals.md) 与 [Interface](interface.md) 分清变量、引用和动态类型。
- **有可运行任务却等不到 CPU**：读 [GMP](gmp-model.md)，再看 [runnext](p-runnext.md) 对本地执行顺序的影响。
- **吞吐稳定，内存却在涨**：先区分仍被引用的对象和等待回收的空间，再查 [GC](gc.md)。

读实现细节前记录 `go version`；跨版本差异查[版本演进](go-versions.md)。

[返回 Go](../README.md)
