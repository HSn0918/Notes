# 让 kubelet 上报 8 个设备，再给 Pod 分配 2 个

从 Node 的 Capacity、Allocatable 一直查到容器里的 `FAKE_VISIBLE_DEVICES` 和挂载目录。资源已经上报、Pod 已经拿到设备，是两次不同的观察。

## 在本地构建

需要 Go 1.26、Docker、kind、kubectl。下面创建专属单节点测试集群，不需要 NVIDIA GPU。

```bash
cd cloud-native/kubernetes/scheduling/device-plugin
go build -mod=readonly ./...
docker build -t learning-notes/fake-device-plugin:latest -f Dockerfile .
```

## 核对节点容量

```bash
kind create cluster --name dp
kind load docker-image learning-notes/fake-device-plugin:latest --name dp
kubectl --context kind-dp apply -f daemonset.yaml
kubectl --context kind-dp -n kube-system rollout status daemonset/fake-device-plugin --timeout=120s
kubectl --context kind-dp describe node | grep learning-notes.io/fake-device
```

在 Capacity 和 Allocatable 中分别找到 `learning-notes.io/fake-device: 8`。

## 用消费 Pod 核对 Allocate

```bash
cat <<EOF | kubectl --context kind-dp apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: fake-device-consumer
spec:
  restartPolicy: Never
  containers:
    - name: app
      image: busybox:1.36
      command: ["sh", "-c", "env | grep FAKE_; ls -la /etc/fake-devices; sleep 3600"]
      resources:
        limits:
          # Extended Resource 必须 requests == limits, 这是 K8s 的硬性约束
          learning-notes.io/fake-device: 2
EOF

# 看 Allocate 返回的 envs / mounts 是否真的注入了
kubectl --context kind-dp wait --for=condition=Ready pod/fake-device-consumer --timeout=120s
kubectl --context kind-dp logs fake-device-consumer
# 期望看到:
#   FAKE_VISIBLE_DEVICES=fake-device-0,fake-device-1
#   FAKE_RESOURCE_NAME=learning-notes.io/fake-device
#   /etc/fake-devices/  (目录存在)
```

具体 ID 由 kubelet 选取。从插件 Pod 的 `Allocate` 日志核对两个 ID，再看容器输出里同一对 ID 和 `/etc/fake-devices`。

## 资源未出现或 Pod 未启动

1. 节点还没有资源时，先查看插件日志中的注册和 `ListAndWatch`，再检查清单是否挂载 `/var/lib/kubelet/device-plugins`。
2. Pod 仍在 Pending 时，查看 Pod 的调度事件，核对资源名和请求数量。Pod 启动后再查看 `Allocate` 日志与容器输出。

```bash
kubectl --context kind-dp -n kube-system logs daemonset/fake-device-plugin
kubectl --context kind-dp describe pod fake-device-consumer
```

## 从清单找到插件实现

| 文件 | 作用 |
| :--- | :--- |
| `main.go` | 进程入口：启动 gRPC server、首次 Register、用 fsnotify 监听 kubelet.sock 重建并重新注册 |
| `plugin.go` | `FakePlugin` 结构体，实现 `pluginapi.DevicePluginServer` 的 4 个 RPC |
| `Dockerfile` | 多阶段构建静态二进制 + distroless 运行镜像 |
| `daemonset.yaml` | 把 `/var/lib/kubelet/device-plugins/` 挂进容器的 DaemonSet |
| `go.mod` | 依赖 `k8s.io/kubelet`（pluginapi）、`google.golang.org/grpc`、`fsnotify` |
| `demo-device-plugin.md` | 笔记入口（含 walkthrough）|

## 清理实验

```bash
kubectl --context kind-dp delete pod fake-device-consumer
kubectl --context kind-dp delete -f daemonset.yaml
kind delete cluster --name dp
```

2026-05 的 macOS 记录确认本地编译，kind 部署与上述输出作为验收步骤。见[实验环境](../../roadmaps/k8s-lab-environments.md)。

[kubelet 与 CRI](../../node-runtime/kubelet-cri-source.md) · [返回 GPU 调度](../README.md)
