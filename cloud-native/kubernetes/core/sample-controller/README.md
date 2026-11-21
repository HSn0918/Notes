# 用 client-go 跟踪一次 ConfigMap 变更

在一个终端观察控制器日志，在另一个终端新建、修改、删除同一个 ConfigMap。沿着同一个 namespace/name，看事件如何经过缓存和工作队列。

## 启动控制器

需要 Go 1.26、kubectl 和可列举所有 namespace 的 ConfigMap 的测试集群。先确认 kubeconfig 当前选中该集群，再启动。

```bash
kubectl --kubeconfig "$HOME/.kube/config" config current-context
cd cloud-native/kubernetes/core/sample-controller
go run -mod=readonly . --kubeconfig "$HOME/.kube/config"
```

## 在第二个终端修改 ConfigMap

启动时：

```
Starting sample-controller (watching ConfigMaps)
Caches synced, starting workers
reconciled kube-system/kube-root-ca.crt (resourceVersion=..., keys=1)
reconciled kube-public/cluster-info (resourceVersion=..., keys=2)
...
```

之后在default namespace 创建/修改/删除 ConfigMap 都会触发对应日志：

```bash
kubectl -n default create configmap demo --from-literal=foo=bar
kubectl -n default patch configmap demo -p '{"data":{"foo":"baz"}}'
kubectl -n default delete configmap demo
```

输出：

```
reconciled default/demo (resourceVersion=..., keys=1)
reconciled default/demo (resourceVersion=..., keys=1)
reconciled default/demo (deleted)
```

Ctrl+C 触发 `SIGINT`，控制器优雅退出（关闭 workqueue → worker 自然结束）。

两次修改很接近时，队列可能合并同一个 key。对照最终资源状态和删除日志，日志条数取决于事件到达次序。

## 对照 main.go 读工作队列

| 行为 | 笔记 `client-go-source.md` 的对应章节 |
| --- | --- |
| `informers.NewSharedInformerFactory` + `factory.Start` | SharedInformerFactory 共享机制 |
| `cmInformer.Informer().AddEventHandler` | processorListener：事件分发 |
| `cache.WaitForCacheSync` | Resync 与 HasSynced |
| `workqueue.NewTypedRateLimitingQueue` + `DefaultTypedControllerRateLimiter` | Workqueue：去重、限速、重试 |
| `c.enqueue` 把对象转 key 后 `queue.Add` | 基础队列的去重与并发安全 |
| `processNextItem` 中的 `Get`/`Done`/`Forget`/`AddRateLimited` | RateLimiter 与指数退避 |
| `cmLister.ConfigMaps(ns).Get(name)` 读本地缓存 | Indexer 与 ThreadSafeStore |
| `syncHandler` 必须幂等且处理 NotFound | 自定义控制器骨架 |

## 验证记录

2026-05 记录使用 macOS、Go 1.26 与测试 kubeconfig。运行条件见[实验环境](../../roadmaps/k8s-lab-environments.md)。

[client-go 源码](../client-go-source.md) · [接着做镜像 ConfigMap](../kubebuilder-operator/README.md)
