# 从一次 ConfigMap 变更跟到控制器

先做[最小控制器](sample-controller/README.md)：改一个 ConfigMap，找出日志里同名对象被处理的记录。再做 [Operator 对照](kubebuilder-operator/README.md)：这一次，控制器还要真的更新镜像 ConfigMap，而不是只打印日志。

## 事件没走到预期的位置

- **请求未写入**：查 [API 对象](api-resource.md)、[REST](restful-api-design.md)、[RBAC](rbac.md) 和 [Admission](admission-webhook-component.md)，再看 [API Server](kube-apiserver-component.md) 的职责。
- **已经写入，看不到变更**：跑 [etcd/Raft 实验](raftexample-walkthrough/demo-raftexample-walkthrough.md)，沿 [etcd 机制](etcd.md)、[组件](etcd-component.md) 和[源码](etcd-source.md)查 revision 与 watch。
- **有事件，没有调谐结果**：从 [Informer](informer.md) 跟进 [client-go](client-go-source.md)，或沿 [Kubebuilder](kubebuilder.md)、[Operator](operator-pattern.md) 和 [controller-runtime](controller-runtime-source.md) 查 Reconcile。

按架构查：[集群基础](kubernetes-basics.md)、[内置控制器](kube-controller-manager-component.md)、[云控制器](cloud-controller-manager-component.md)、[Borg](google-borg.md)。按观测手段查：[日志采集](logging-agent-component.md)、[Metrics Server](metrics-server-component.md)、[Dashboard](kubernetes-dashboard-component.md)。

[返回 Kubernetes](../README.md)
