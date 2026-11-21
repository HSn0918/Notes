# GPU 调度：卡在哪里，作业在等谁

先跑[放置与回填实验](placement-lab/README.md)。先预测 64 张空闲卡能放几个整组作业，再换掉排队策略，分别看大、小作业各等了多久。

## 从排队追到容器

1. **队列接受了作业**，不等于所有 worker 已分到卡。先看[成组准入与放置](training-job-placement.md)，再对照 [PodGroup](scheduler-podgroup-source.md) 和 [Volcano](volcano-source.md)。
2. **调度器选了节点**，不等于这个节点是唯一可选项。用[节点打分实验](scheduler-plugin/README.md)改变一个标签，沿[调度流程](scheduler-deep-dive.md)、[组件职责](kube-scheduler-component.md) 和 [Framework](scheduler-framework-source.md) 看分数怎么影响结果。
3. **节点上报了设备**，不等于容器能运行 CUDA。做[通用设备插件](device-plugin/README.md)与[模拟 GPU 分配](fake-gpu/README.md)，核对容器实际拿到的 ID；实现见 [Device Plugin](device-plugin-component.md)、[GPU 调度机制](gpu-scheduling.md) 和[源码](gpu-scheduling-source.md)。

研究切卡时再进 [HAMi 实验](hami-mac/README.md)、[专项路线](hami-learning-path.md)和[实现](hami-source.md)。研究训练时回[多机训练](../../../ai/training/README.md)核对 rank 与设备的映射。

[返回 Kubernetes](../README.md)
