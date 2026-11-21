# 用一个 PVC 追踪 CSI 从建卷到挂载

一个 Pod 申请 1 GiB，并向 /data 写入 marker。先看 PVC 是否绑定，再进入 Pod 读回 marker，分别核对控制器建卷和节点挂载。

## 为单节点 kind 编译 Linux 驱动

需要 Go 1.26、Docker、kubectl、kind。驱动使用 Linux bind mount；即使在 macOS 上构建，也要选择 kind 节点的 CPU 架构。

```bash
cd cloud-native/kubernetes/storage/csi-hostpath
kind create cluster --name csi-lab
NODE_ARCH=$(kubectl --context kind-csi-lab get node -o jsonpath='{.items[0].status.nodeInfo.architecture}')
CGO_ENABLED=0 GOOS=linux GOARCH="$NODE_ARCH" go build -mod=readonly -o csi-hostpath .
docker build --platform "linux/$NODE_ARCH" -t ghcr.io/learning-notes/csi-hostpath:dev -f - . <<'EOF'
FROM gcr.io/distroless/static-debian12
COPY csi-hostpath /csi-hostpath
ENTRYPOINT ["/csi-hostpath"]
EOF
kind load docker-image ghcr.io/learning-notes/csi-hostpath:dev --name csi-lab
```

## 部署后检查 PVC 与文件

```bash
kubectl --context kind-csi-lab apply -f csidriver.yaml
kubectl --context kind-csi-lab apply -f daemonset.yaml      # 含 ServiceAccount + RBAC + Controller + Node
kubectl --context kind-csi-lab apply -f storageclass.yaml
```

观察 Pod / CSIDriver / CSINode：

```bash
kubectl --context kind-csi-lab -n kube-system rollout status daemonset/csi-hostpath-node --timeout=120s
kubectl --context kind-csi-lab get csidriver
kubectl --context kind-csi-lab get csinode -o yaml   # 看 spec.drivers[] 里是不是有 learning-notes.csi.k8s.io
```

### 创建 PVC 和消费 Pod

```bash
kubectl --context kind-csi-lab apply -f pvc.yaml

kubectl --context kind-csi-lab get pvc csi-hostpath-test
# 期望：STATUS 从 Pending -> Bound（external-provisioner 调 CreateVolume 创建出 PV）

kubectl --context kind-csi-lab wait --for=condition=Ready pod/csi-hostpath-test --timeout=120s
kubectl --context kind-csi-lab exec csi-hostpath-test -- cat /data/marker
# 看到 "csi-hostpath demo works at ..." 说明 bind mount + 读写都正常
```

## 清理专用环境

```bash
kubectl --context kind-csi-lab delete -f pvc.yaml
kubectl --context kind-csi-lab delete -f storageclass.yaml
kubectl --context kind-csi-lab delete -f daemonset.yaml
kubectl --context kind-csi-lab delete -f csidriver.yaml
```

```bash
kind delete cluster --name csi-lab
```

## 从 RPC 找源码

[controller.go](controller.go) 按卷名生成稳定的 ID，使 CreateVolume 可以重试。[node.go](node.go) 将本地目录 bind mount 到 kubelet 的目标路径。驱动依赖节点本地文件系统，本例只检验一次读写。

- 没实现 `NodeStageVolume` / `NodeUnstageVolume`，所以 RWX 多 Pod 共享场景下没有共享 stage 层（每个 Pod 各自 publish）。
- 没实现 `ControllerExpandVolume` —— SC 里 `allowVolumeExpansion: false`。
- 没实现 `CreateSnapshot` —— 不接入 external-snapshotter。
- 没有真正的容量管理 —— `CreateVolume` 不检查 dataRoot 剩余空间。
- 错误处理简化 —— 生产驱动会区分 gRPC `Aborted` / `FailedPrecondition` / `Internal` 等更精细的状态码。

## 验证记录

2026-05 的 macOS 记录区分了 Linux 交叉编译和真实挂载，见[实验环境](../../roadmaps/k8s-lab-environments.md)。端到端验收需取得 Pod 内读回的文件内容。

[CSI 源码](../csi-source.md) · [完整走读](demo-csi-hostpath.md)
