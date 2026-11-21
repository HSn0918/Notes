# 把 4 张模拟卡上报为 40 个 vGPU 切片

向 kubelet 上报每卡 10 个切片，启动两个各申请一份的 Pod。对照物理卡标识、切片 ID 和显存配额，区分资源计数、设备选择和容器实际能使用的显存。

## 在本地构建

需要 Go 1.26、Docker、kind、kubectl。下面创建专属单节点测试集群，不需要 NVIDIA GPU。

```bash
cd cloud-native/kubernetes/scheduling/hami-mac
go build -mod=readonly ./...
docker build -t learning-notes/hami-mac:latest -f Dockerfile .
```

## 核对节点容量

```bash
kind create cluster --name hami-mac
kind load docker-image learning-notes/hami-mac:latest --name hami-mac
kubectl --context kind-hami-mac apply -f daemonset.yaml
kubectl --context kind-hami-mac -n kube-system rollout status daemonset/hami-mac-plugin --timeout=120s
kubectl --context kind-hami-mac describe node | grep nvidia.com/gpu
```

在 Capacity 和 Allocatable 中分别找到 `nvidia.com/gpu: 40`。

## 用消费 Pod 核对 Allocate

```bash
kubectl --context kind-hami-mac apply -f pod-hami-consumer.yaml
kubectl --context kind-hami-mac wait --for=condition=Ready pod/hami-consumer pod/hami-consumer-b --timeout=120s
kubectl --context kind-hami-mac logs hami-consumer
kubectl --context kind-hami-mac logs hami-consumer-b
```

在两份日志中找 `HAMI_FAKE_VGPU_IDS`、`NVIDIA_VISIBLE_DEVICES`、`CUDA_DEVICE_MEMORY_LIMIT_0=3000m` 和 `CUDA_DEVICE_SM_LIMIT_0=30`。切片 ID 应不同；是否属于同一张物理卡要看实际 UUID。容器里只注入了变量，没有加载真正执行 CUDA 配额的库。

## 资源未出现或 Pod 未启动

1. 节点还没有资源时，先查看插件日志中的注册和 `ListAndWatch`，再检查清单是否挂载 `/var/lib/kubelet/device-plugins`。
2. Pod 仍在 Pending 时，查看 Pod 的调度事件，核对资源名和请求数量。Pod 启动后再查看 `Allocate` 日志与容器输出。

```bash
kubectl --context kind-hami-mac -n kube-system logs daemonset/hami-mac-plugin
kubectl --context kind-hami-mac describe pod hami-consumer
```

## 模拟切片与真实 HAMi 的差异

| 真实 HAMi | 本 demo |
| :--- | :--- |
| webhook 改写 Pod spec、注入 volumeMounts | 不做（用进程 env 当默认值） |
| scheduler-extender 选 vGPU + 写 annotation | 不做（DeviceManager 随便挑） |
| ListAndWatch 用 NVML 真实读卡 UUID | 全零硬编码 UUID |
| Topology / NUMA 上报 | 不做 |
| 健康巡检（XID/ECC） | 永远 Healthy |
| libvgpu.so 真 hook CUDA API | 仅注入 env，无真实 hook |
| 多容器共享一卡的共享内存配额协商 | 无 |
| leader election + annotation reconcile | 无 |

## 清理实验

```bash
kubectl --context kind-hami-mac delete -f pod-hami-consumer.yaml
kubectl --context kind-hami-mac delete -f daemonset.yaml
kind delete cluster --name hami-mac
```

2026-05 的 macOS 记录确认本地编译，kind 部署与上述输出作为验收步骤。见[实验环境](../../roadmaps/k8s-lab-environments.md)。

[HAMi 学习路线](../hami-learning-path.md) · [进程内配额实验](libvgpu-hook-demo/README.md) · [返回 GPU 调度](../README.md)
