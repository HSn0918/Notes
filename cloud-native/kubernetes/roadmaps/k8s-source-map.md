#kubernetes #源码导读

相关笔记：[[k8s-development-roadmap]] · [[k8s-component-map]]

# Kubernetes 源码导读

> 返回 [Kubernetes 学习索引](../README.md)

源码导读负责固定版本下的调用链、关键结构与验证方式；概念和选型返回对应主题目录。

| 方向 | 笔记 |
| --- | --- |
| Client / Controller | [client-go](../core/client-go-source.md) · [controller-runtime](../core/controller-runtime-source.md) |
| Scheduling | [scheduler-framework](../scheduling/scheduler-framework-source.md) · [scheduler-podgroup](../scheduling/scheduler-podgroup-source.md) · [Volcano](../scheduling/volcano-source.md) |
| Node / Runtime | [kubelet-cri](../node-runtime/kubelet-cri-source.md) · [CRI](../node-runtime/cri-source.md) |
| Network / Storage | [CNI](../networking/cni-source.md) · [CSI](../storage/csi-source.md) |
| Infrastructure | [etcd](../core/etcd-source.md) |
| GPU | [GPU scheduling](../scheduling/gpu-scheduling-source.md) · [HAMi](../scheduling/hami-source.md) |
