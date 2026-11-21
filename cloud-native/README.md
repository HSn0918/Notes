# 让训练作业拿到 GPU

[先给一个 64 卡作业找位置](kubernetes/scheduling/placement-lab/README.md)。跑完静态布局，再跑队列回放：短作业能否用掉暂时空闲的节点，又不推迟队首的大作业？只需 CPU，不会访问集群。

## 在真实作业里定位等待

- **还没分到节点**：查[队列、配额与 GPU 放置](kubernetes/scheduling/README.md)。空闲卡总数不等于可用整机数。
- **已经分到节点，进程还没起来**：查[节点运行时](kubernetes/node-runtime/README.md)；看不懂镜像、隔离或资源限制时再回[容器基础](containers/README.md)。
- **各进程都启动了，训练却没进展**：先区分[进程之间连不通](kubernetes/networking/README.md)和[训练数据或断点读不到](kubernetes/storage/README.md)。
- **对象改了，控制器没动作**：跟踪[一次事件从写入到调谐](kubernetes/core/README.md)。

看[完整 Kubernetes 目录](kubernetes/README.md)，或回[训练侧](../ai/training/README.md)核对每个进程实际用的卡。

[返回首页](../README.md)
