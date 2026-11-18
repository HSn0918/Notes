# Kubernetes 核心架构、控制面、节点与调度

> 返回 [Kubernetes 学习索引](../README.md)

本目录收敛了 Kubernetes 核心概念与控制链路（原 `foundations/`、`infrastructure/`、`control-plane/`、`node/`、`scheduling/`、`extension/`），按集群数据流顺序组织。

## 集群架构与持久化底座

- [Kubernetes 基础架构](kubernetes-basics.md) —— Control Plane / Node 全局架构与核心机制
- [Google Borg](google-borg.md) —— Kubernetes 前身 Borg 的 Cell、Job/Task 与设计演进
- [etcd 分布式存储](etcd.md) —— Kubernetes 唯一持久化状态后端、Raft 一致性与运维最佳实践

## API 与控制面

- [API Resource 与 GVK](api-resource.md) —— Group、Version、Kind、Resource 与对象元数据模型
- [RESTful API 设计](restful-api-design.md) —— API Server 资源路径、Watch 与声明式语义
- [RBAC 权限控制](rbac.md) —— Role/ClusterRole、Binding 与最小权限实践
- [Informer List-Watch 机制](informer.md) —— Reflector、DeltaFIFO、Indexer 与控制器本地缓存

## 节点运行时

- [Pod Probe 健康检查](probes.md) —— Liveness、Readiness 与 Startup 探针行为边界
- [OCI Runtime 规范](oci-runtime.md) —— runc、OCI bundle 与容器生命周期标准

## 调度与扩展机制

- [Kube-Scheduler 深度解析](scheduler-deep-dive.md) —— 调度框架、Assume/Bind、抢占与队列
- [GPU 调度与 Device Plugin](gpu-scheduling.md) —— Extended Resource 与设备插件机制
- [分布式训练作业的成组准入与放置](training-job-placement.md) —— 从队列和跨节点拓扑追到 rank 会合、第一步更新与断点恢复
- [Kubebuilder 自定义控制器](kubebuilder.md) —— CRD + Controller 开发骨架与脚手架
- [Operator Pattern](operator-pattern.md) —— 声明式调谐闭环与复杂有状态应用自动化
- [Velero 集群备份恢复](velero.md) —— 资源与持久化卷的备份还原机制

源码层继续阅读 [[scheduler-framework-source]]、[[scheduler-podgroup-source]]、[[volcano-source]]、[[client-go-source]]、[[controller-runtime-source]] 与 [[gpu-scheduling-source]]。
