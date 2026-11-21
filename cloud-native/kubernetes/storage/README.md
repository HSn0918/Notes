# 沿训练文件找到它挂载的卷

从 [hostPath CSI 实验](csi-hostpath/README.md)开始。在测试节点上，把一份 PVC 跟到 Pod 内的挂载目录。PVC 显示 Bound，只能说明申请已绑定，不能证明训练进程读到了完整 checkpoint。

- **申请未绑定**：沿[卷的生命周期](volume-lifecycle.md)检查 StorageClass、PVC 和 PV。
- **已绑定但挂载失败**：从 [CSI 的接口](csi.md)和 [sidecar 分工](csi-sidecars.md)定位步骤，再对照[驱动职责](csi-driver-component.md)、[源码](csi-source.md)和[故障现象](csi-troubleshooting.md)。
- **要换后端或做备份**：按实际后端查 [Ceph](ceph-csi.md)、[Longhorn](longhorn.md)、[OpenEBS](openebs.md)、[NFS](nfs-csi.md) 或[云盘](cloud-provider-csi.md)。[Velero 的备份](velero.md)不等价于优化器和数据进度都恢复正确。

[返回 Kubernetes](../README.md)
