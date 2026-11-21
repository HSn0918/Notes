# 跟踪一张模拟 GPU 的 UUID 从节点进入容器

插件上报 4 个 GPU 风格的设备 ID。消费 Pod 声明一个资源，查看它收到的 UUID，并和插件 Allocate 日志核对。

## 在本地构建

需要 Go 1.26、Docker、kind、kubectl。下面创建专属单节点测试集群，不需要 NVIDIA GPU。

```bash
cd cloud-native/kubernetes/scheduling/fake-gpu
go build -mod=readonly ./...
docker build -t learning-notes/fake-gpu:latest -f ../device-plugin/Dockerfile .
```

## 核对节点容量

```bash
kind create cluster --name fake-gpu
kind load docker-image learning-notes/fake-gpu:latest --name fake-gpu
kubectl --context kind-fake-gpu apply -f daemonset.yaml
kubectl --context kind-fake-gpu -n kube-system rollout status daemonset/fake-gpu-plugin --timeout=120s
kubectl --context kind-fake-gpu describe node | grep fake-gpu.k8s.io/gpu
```

在 Capacity 和 Allocatable 中分别找到 `fake-gpu.k8s.io/gpu: 4`。

## 用消费 Pod 核对 Allocate

```bash
kubectl --context kind-fake-gpu apply -f pod-consumer.yaml
kubectl --context kind-fake-gpu wait --for=condition=Ready pod/fake-gpu-consumer --timeout=120s
kubectl --context kind-fake-gpu logs fake-gpu-consumer
```

对照 `FAKE_GPU_DEVICES` 和 `NVIDIA_VISIBLE_DEVICES`，两者应包含同一个 UUID。这里检查的是 Allocate 返回的环境变量。在这个 kind 节点上，容器没有实际的 NVIDIA 设备。

## 资源未出现或 Pod 未启动

1. 节点还没有资源时，先查看插件日志中的注册和 `ListAndWatch`，再检查清单是否挂载 `/var/lib/kubelet/device-plugins`。
2. Pod 仍在 Pending 时，查看 Pod 的调度事件，核对资源名和请求数量。Pod 启动后再查看 `Allocate` 日志与容器输出。

```bash
kubectl --context kind-fake-gpu -n kube-system logs daemonset/fake-gpu-plugin
kubectl --context kind-fake-gpu describe pod fake-gpu-consumer
```

## 从清单找到插件实现

| 文件 | 作用 |
| :--- | :--- |
| `main.go` | 进程入口：启动 gRPC server、首次 Register、用 fsnotify 监听 kubelet.sock 重建并重新注册 |
| `plugin.go` | `FakeGPU` 结构体，实现 `pluginapi.DevicePluginServer` 的 4 个 RPC，Allocate 返回 `NVIDIA_VISIBLE_DEVICES` env |
| `daemonset.yaml` | 把 `/var/lib/kubelet/device-plugins/` 挂进容器的 DaemonSet |
| `pod-consumer.yaml` | 请 1 块 `fake-gpu.k8s.io/gpu` 的 busybox Pod，启动后打印注入的 env |
| `go.mod` | 依赖 `k8s.io/kubelet`（pluginapi）、`google.golang.org/grpc`、`fsnotify` |
| `demo-fake-gpu.md` | 笔记入口（含 walkthrough）|

## 清理实验

```bash
kubectl --context kind-fake-gpu delete pod fake-gpu-consumer
kubectl --context kind-fake-gpu delete -f daemonset.yaml
kind delete cluster --name fake-gpu
```

2026-05 的 macOS 记录确认本地编译，kind 部署与上述输出作为验收步骤。见[实验环境](../../roadmaps/k8s-lab-environments.md)。

[GPU 分配源码](../gpu-scheduling-source.md) · [返回 GPU 调度](../README.md)
