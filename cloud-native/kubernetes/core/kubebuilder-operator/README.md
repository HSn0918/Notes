# 让 Operator 把 ConfigMap 的改动同步到副本

给源 ConfigMap 加一条 annotation，控制器会创建带 ownerReference 的 mirror。再手动改 mirror，看控制器是否把值改回源对象的值。

## 在测试集群创建源对象与副本

需要 Go 1.26、kubectl 和测试 kubeconfig。命令分两个终端执行，均使用 `~/.kube/config` 的同一 context，实验对象位于 `default`。

```bash
cd cloud-native/kubernetes/core/kubebuilder-operator

# 2) 本地跑（用 ~/.kube/config 连集群，跑在你电脑上）
go run . --kubeconfig ~/.kube/config --namespace default

# 另开一个终端验证：
kubectl create configmap demo-src -n default --from-literal=x=1
kubectl label  configmap demo-src demo.learning-notes/source=true -n default
kubectl annotate configmap demo-src demo.learning-notes/payload="hello" -n default

# 观察 mirror 被创建
kubectl get configmap demo-src-mirror -n default -o yaml
# 应该看到 annotation demo.learning-notes/payload: hello
# 以及 metadata.ownerReferences 指向 demo-src

# 改 annotation：
kubectl annotate configmap demo-src demo.learning-notes/payload="world" --overwrite -n default
kubectl get configmap demo-src-mirror -n default -o jsonpath='{.metadata.annotations}'
# 应该变成 {"demo.learning-notes/payload":"world"}

# 演示 Owns 反向链路 —— 改 mirror 的 annotation，会被 reconcile 改回去：
kubectl annotate configmap demo-src-mirror demo.learning-notes/payload="tampered" --overwrite -n default
# 等待下一次 reconcile，然后重读 mirror
kubectl get configmap demo-src-mirror -n default -o jsonpath='{.metadata.annotations}'

# 清理：
kubectl delete configmap demo-src -n default
# mirror 也会被级联删除
```

## 从对象关系追到 Reconcile

监听一个 namespace（默认 `default`）下所有 `ConfigMap`，对每个带
`label demo.learning-notes/source=true` 的源 CM：

1. 如果它有 annotation `demo.learning-notes/payload`，
   就创建（或更新）一个名为 `<source-name>-mirror` 的 ConfigMap，
   把该 annotation 复制过去。
2. 源 CM 被删时，级联删除 mirror。
3. mirror CM 上写有 `ownerReferences` 指回源 CM，演示 `Owns(...)`
   的反向链路：若有人手动改 mirror 的 annotation，Controller 会被
   触发并把它改回来。

| 本 demo | 对应真实库代码 |
|---------|--------------|
| `ctrl.NewManager(...)` | `pkg/manager/manager.go: New` |
| `(&ConfigMapReconciler{}).SetupWithManager` | `pkg/builder/controller.go: Build` |
| `For(&ConfigMap{})` | `EnqueueRequestForObject` 注册到 source.Kind |
| `Owns(&ConfigMap{})` | `EnqueueRequestForOwner` 注册到 source.Kind |
| `r.Get / r.Update / r.Create` | `DelegatingClient`（读 Cache、写直连） |
| `ctrl.SetControllerReference` | `pkg/controller/controllerutil` |
| `ctrl.SetupSignalHandler` | `pkg/manager/signals` |

## 运行参数

| flag | 默认 | 说明 |
|------|------|------|
| `--namespace` | `default` | 监听哪个 namespace |
| `--metrics-bind-address` | `:8080` | Prometheus metrics |
| `--health-probe-bind-address` | `:8081` | `/healthz`、`/readyz` |
| `--leader-elect` | `false` | 多副本时启用 Leader Election（基于 Lease） |
| `--zap-log-level` | `info` | 日志级别（zap.Options） |

## 验证与权限

2026-05 的 macOS 记录使用 controller-runtime v0.19.3。运行方式与权限边界见[实验环境](../../roadmaps/k8s-lab-environments.md)。本地进程沿用 kubeconfig 的权限。

[controller-runtime 源码](../controller-runtime-source.md) · [返回控制面](../README.md)
