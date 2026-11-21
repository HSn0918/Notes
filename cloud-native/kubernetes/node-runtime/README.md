# Pod 已经绑定，容器还没起来

先在本地用 [fake CRI](fake-cri/README.md) 查询 Version，再创建一个模拟 sandbox。能连通 socket、能登记一个 sandbox、能运行真实容器，是三件不同的事。

**沿调用链走一遍**：[kubelet](kubelet-component.md)观察 Pod 的期望状态，[容器运行时](container-runtime-component.md)接收请求。对照 [kubelet 调用链](kubelet-cri-source.md)和 [CRI 接口](cri-source.md)，记录出错的是哪一步。

**进程已经启动**：隔离和底层进程看 [OCI Runtime](oci-runtime.md)；容器不接流量则看[探针](probes.md)。就绪探针失败和进程启动失败不是同一故障。

[返回 Kubernetes](../README.md)
