# Go Runtime：从一条阻塞栈找源码

先从[两个取消测试](../README.md)取一条明确行为，再沿 `context/context.go` 的取消路径回看：信号送到了哪个子节点，等待者又是在哪里被唤醒的？遇到生产问题，先拿栈或 profile，不要按源码目录从头读。

## 按现象读

- goroutine 一直不退：查 [context](go-context-source.md) 和 [channel](go-channel-source.md)，区分“收到了取消”与“代码检查了取消”。用 goroutine dump、block profile 或 `go tool trace` 看等待点。
- runnable 多却吞吐低：读 [GMP 热路径](go-gmp-source.md)。把 CPU profile 与调度 trace 放在一起，区分等待运行和已经占用 CPU。
- 堆增长：对照 [slice 扩容](go-slice-source.md)、[map 分配](go-map-source.md)和[垃圾回收](go-gc-source.md)。分开看 `-alloc_space` 与 `-inuse_space`。
- 同一段解释与本机源码对不上：查 [map 新旧实现](go-map-old-vs-new.md)和 [GC 新旧实现](go-gc-old-vs-new.md)。

## 对齐源码版本

原笔记核验环境是 `go1.26.1 darwin/arm64`。在自己机器上用 `go version` 和 `go env GOROOT` 找对应的标准库；不要照抄作者的安装路径。Go 1.24 后的 Swiss Table map 与旧 bucket map 分开读。语言保证的语义、当前实现的优化和手头这次测量，是三种不同的证据。

## 合上源码后自检

### Q: Go runtime 源码应该从哪里读？

> [!question]- 参考答案（点击展开）
>
> 先读结构体和状态机，再读热路径。GMP 从 `runtime2.go` 的 `g/m/p` 和 `proc.go` 的 `newproc/schedule/findRunnable/execute` 开始；channel 从 `hchan/chansend/chanrecv/closechan` 开始；map 从 Go 1.24+ 的 `internal/runtime/maps` 开始；GC 从 `mgc.go/mgcmark.go/mbarrier.go/mgcpacer.go` 开始。

### Q: 为什么要单独区分新 map 和旧 map？

> [!question]- 参考答案（点击展开）
>
> 因为 Go 1.24+ 的实现从 `hmap/bmap/overflow bucket` 切到了 Swiss Table 风格的 `Map/Table/Group/Control word/Directory`。语言语义基本不变，但局部性、查找方式、扩容粒度、源码入口和 unsafe/linkname 依赖风险都变了。

### Q: 为什么要单独区分旧 GC 和 Green Tea GC？

> [!question]- 参考答案（点击展开）
>
> Green Tea 改的是标记扫描的局部性：传统实现偏向对象级 workbuf 扫描，Green Tea 会延迟扫描并按 span 聚合对象，使用 marks/scans 两套标记位和 FIFO span queue。它不等于把 Go 变成分代/移动 GC，也不改变 Go GC 的基本并发标记清扫模型。

### Q: 面试回答源码时最容易犯什么错？

> [!question]- 参考答案（点击展开）
>
> 把旧版本实现当成当前实现、把语义当成实现、把平均复杂度当成绝对复杂度、把调优参数当成根因。回答时要明确版本、源码文件、核心结构、热路径、慢路径和排障手段。

[返回 Go](../README.md)
