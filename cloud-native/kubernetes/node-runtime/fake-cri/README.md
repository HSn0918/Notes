# 用 crictl 跟踪一个模拟 Pod sandbox

启动只在本地 Unix socket 上监听的 CRI 服务，创建一个 sandbox，再观察它从 Ready 到移除。整个过程只读写进程内存，无需 kubelet 或 Kubernetes 集群。

## 启动本地服务

需要 Go 和 crictl。默认地址为 `/tmp/fake-cri.sock`。

```bash
cd cloud-native/kubernetes/node-runtime/fake-cri
go build -mod=readonly -o fake-cri .
./fake-cri
```

看到 `fake-cri listening on unix:///tmp/fake-cri.sock` 后，保留这个终端。

## 在第二个终端执行 crictl

```bash
crictl --runtime-endpoint unix:///tmp/fake-cri.sock version
```

预期输出：
```
Version:  0.1.0
RuntimeName:  fake-cri
RuntimeVersion:  v0.0.1
RuntimeApiVersion:  v1
```

### 列 sandbox（初始空列表）

```bash
crictl --runtime-endpoint unix:///tmp/fake-cri.sock pods
```

预期输出：
```
POD ID   CREATED   STATE   NAME   NAMESPACE   ATTEMPT   RUNTIME
```

### 创建一个 sandbox 并保存它的 ID

准备 `pod.json`：

```json
{
  "metadata": {
    "name": "demo-pod",
    "namespace": "default",
    "uid": "00000000-0000-0000-0000-000000000001",
    "attempt": 0
  },
  "log_directory": "/tmp",
  "linux": {}
}
```

执行：

```bash
sandbox_id=$(crictl --runtime-endpoint unix:///tmp/fake-cri.sock runp pod.json)
echo "$sandbox_id"
```

预期输出（sandbox id）：
```
sandbox-1747280400-1
```

再列：

```bash
crictl --runtime-endpoint unix:///tmp/fake-cri.sock pods
```

```
POD ID                 CREATED          STATE   NAME      NAMESPACE   ATTEMPT   RUNTIME
sandbox-1747280400-1   3 seconds ago    Ready   demo-pod  default     0
```

`crictl --runtime-endpoint unix:///tmp/fake-cri.sock inspectp "$sandbox_id"` 能看到伪 IP `10.244.0.42`。

### 停止 + 删除

```bash
crictl --runtime-endpoint unix:///tmp/fake-cri.sock stopp "$sandbox_id"
crictl --runtime-endpoint unix:///tmp/fake-cri.sock rmp "$sandbox_id"
```

## 逐项对照 RuntimeService

| RPC | 行为 |
|-----|------|
| `Version` | 返回固定字符串 `fake-cri v0.0.1` |
| `Status` | 返回 `RuntimeReady=true`、`NetworkReady=true` |
| `RunPodSandbox` | 在内存 map 里登记 sandbox，返回伪 id |
| `StopPodSandbox` | 标记 NOTREADY（必须 idempotent） |
| `RemovePodSandbox` | 从 map 删除（必须 idempotent） |
| `PodSandboxStatus` | 单个 sandbox 详情，含假 IP `10.244.0.42` |
| `ListPodSandbox` | 全量返回 |
| 其它 RPC | 走 `UnimplementedRuntimeServiceServer`，返回 Unimplemented |

| 章节 | 代码体现 |
|------|----------|
| § 二 gRPC 契约 | `RegisterRuntimeServiceServer` 在 `main()` 把 fakeRuntime 注册到 gRPC server |
| § 三 sandbox 模型 | `RunPodSandbox` 只创建 in-memory 状态——真实 runtime 这里要启 pause + 调 CNI |
| § 四 cri-client 连接 | server 用 `grpc.NewServer()` + unix socket，与客户端 `grpc.DialContext(unix://...)` 对接 |
| § 八 RPC idempotency | `StopPodSandbox` / `RemovePodSandbox` 找不到 sandbox 也返回成功（idempotent） |
| § 十一 面试 Q9 | 没实现的 RPC 走 Unimplemented，对应"CRI 是同步 unary"——客户端会立刻拿到 codes.Unimplemented |

## 运行范围

2026-05 有 macOS 本地运行记录。重启会丢失 sandbox 记录，服务没有镜像、容器进程或网络实现。详见[实验环境](../../roadmaps/k8s-lab-environments.md)。

[CRI 协议与源码](../cri-source.md) · [kubelet 调用路径](../kubelet-cri-source.md)
