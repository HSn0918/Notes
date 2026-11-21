#kubernetes #实验

# Kubernetes 实验环境与历史验证

相关笔记：[[k8s-development-roadmap]] · [[k8s-source-map]]

> **2026-05 历史验证环境**：Mac (Darwin 25.2.0) + Go 1.26
> 表中状态属于当时的实现，不代表已在当前机器部署。实验现与所属子系统放在一起；从仓库根目录进入下表链接对应的目录。

| demo | 编译 | Mac 运行 | 前置依赖 | 最后验证命令 | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [sample-controller](../core/sample-controller/) | ✅ | ✅ | `go mod tidy` + kubeconfig | `cd cloud-native/kubernetes/core/sample-controller && go build ./...` | 5 分钟反馈，最快入门 |
| [device-plugin](../scheduling/device-plugin/) | ✅ | ⚠️ | `go mod tidy`、需 kind/真集群 | `cd cloud-native/kubernetes/scheduling/device-plugin && go build ./...` | DaemonSet 部署，Mac kind 可跑 |
| [fake-cri](../node-runtime/fake-cri/) | ✅ | ✅ | `go mod tidy`、`crictl` | `cd cloud-native/kubernetes/node-runtime/fake-cri && go build ./...` | 最小 fake CRI server，`crictl` 直接探测 |
| [fake-gpu](../scheduling/fake-gpu/) | ✅ | ⚠️ | `go mod tidy`、kind | `cd cloud-native/kubernetes/scheduling/fake-gpu && go build ./...` | 同上 |
| [hami-mac](../scheduling/hami-mac/) | ✅ | ⚠️ | docker build + kind load | `cd cloud-native/kubernetes/scheduling/hami-mac && go build ./...` | 看 README，含 walkthrough |
| [kubebuilder-operator](../core/kubebuilder-operator/) | ✅ | ✅ | `go mod tidy`、kubeconfig | `cd cloud-native/kubernetes/core/kubebuilder-operator && go build ./...` | 升 controller-runtime v0.19.3 已修复 |
| [csi-hostpath](../storage/csi-hostpath/) | ⚠️ | ❌ | `GOOS=linux go build`，需 Linux 节点 | `cd cloud-native/kubernetes/storage/csi-hostpath && GOOS=linux go build ./...` | 用 Linux-only `unix.MS_BIND`，Mac 编不过；kind 节点里跑 |
| [cni-bridge](../networking/cni-bridge/) | N/A | ✅ | docker | `cd cloud-native/kubernetes/networking/cni-bridge && ./run-in-docker.sh` | bash 实现，Mac 通过 Docker 跑 |
| [scheduler-plugin](../scheduling/scheduler-plugin/) | ⚠️ | ⚠️ | 手填 `replace` 列表 | `cd cloud-native/kubernetes/scheduling/scheduler-plugin && go build ./...` | 详见 README「编译说明」；推荐 fork scheduler-plugins 仓用其 go.mod |
| [raftexample-walkthrough/etcd-client-demo](../core/raftexample-walkthrough/etcd-client-demo/) | ✅ | ✅ | `go mod tidy`、需 etcd | `cd cloud-native/kubernetes/core/raftexample-walkthrough/etcd-client-demo && go build ./...` | 跑前 `etcd` 起本地实例 |

## 通用前置

在实验自己的目录内检查 go.mod 的 Go 版本及 README 所需的工具。先按锁定依赖构建；缺模块时记录原始错误，不要为了让学习笔记显示绿色而修改依赖。需要 Docker、kind 或节点挂载权限的步骤只在相应测试环境执行。

## 图例

- ✅ = 在 Mac 直接通过
- ⚠️ = 需要额外步骤（已注明）
- ❌ = Mac 上不可行（需 Linux 环境）
- N/A = 非 Go demo

## 排错快查

| 报错 | 原因 | 解决 |
| :--- | :--- | :--- |
| `missing go.sum entry` | go.sum 不全 | `go mod tidy` |
| `undefined: unix.MS_BIND`（csi-hostpath） | Mac 没有 Linux syscall 常量 | `GOOS=linux go build`，或在 kind 节点容器里编 |
| `not enough arguments in call to fn`（kubebuilder-operator 旧版） | controller-runtime v0.18 与 k8s.io v0.31 API 不兼容 | 已升 v0.19.3，重新 `go mod tidy` |
| `package k8s.io/kubernetes/pkg/scheduler/framework: ... v0.0.0`（scheduler-plugin） | K8s 主仓 staging 占位版本 | 手填 go.mod 的 replace 列表，见该 demo README |
