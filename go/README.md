# Go：控制器退出时，后台任务也退出了吗

先在自己安装的 Go 1.26 标准库里运行两个测试。不需要集群，也不需要建模块：

```bash
GOTOOLCHAIN=local go test -count=1 -v -run '^TestAfterFunc(CalledAfterCancel|NotCalledAfterStop)$' context
```

留意两个测试名：一个先取消 context，另一个先停止回调再取消。两行都应显示 PASS；`[no tests to run]` 不算验证。版本不同时先用 `go test -list AfterFunc context` 核对名称。

**后台任务没退出**：从 [Context](fundamentals/context.md) 和 [Channel](fundamentals/channel.md) 区分取消信号与实际阻塞点，再追[取消传播](internals/go-context-source.md)和[唤醒路径](internals/go-channel-source.md)。

**进程在跑却少有进展**：看 [GMP](fundamentals/gmp-model.md)、[本地队列](fundamentals/p-runnext.md) 和[调度源码](internals/go-gmp-source.md)。若是内存迟迟不降，再查 [Slice](fundamentals/slice.md) 和 [GC](fundamentals/gc.md)。

[语言机制](fundamentals/README.md)讲程序能依赖什么；[Runtime 源码](internals/README.md)讲特定版本如何实现。

[返回首页](../README.md)
