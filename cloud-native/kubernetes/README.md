# Kubernetes：训练作业停在哪一步

有一条作业不动了。先找到最后一个已经发生的事件，从它的**下一步**开始查：

1. API 里已有对象，但没有后续动作：[控制面与控制器](core/README.md)。
2. worker 还没分到节点：[GPU 与批作业调度](scheduling/README.md)。没有集群也能做这里的放置实验。
3. Pod 已绑定，但没有训练进程：[kubelet 与容器运行时](node-runtime/README.md)。
4. worker 已起来，但彼此连不上：[Pod 网络、DNS 与 Service](networking/README.md)。
5. 进程能通信，但读不到训练文件：[存储卷的绑定与挂载](storage/README.md)。

第一次学平台开发，选[一条实践路线](roadmaps/README.md)。做过实验后，再用[题库](interview/README.md)检查能否脱稿解释。

[返回云原生](../README.md)
