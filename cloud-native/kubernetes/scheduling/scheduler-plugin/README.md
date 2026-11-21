# 用两个节点验证调度器的打分插件

给两个 worker 打相反的标签，让第二个调度器安排 6 个 Pod。交换标签后创建新 Pod，比较两轮落点。已经运行的 Pod 不会因为标签变化自行迁移。

## 准备两个 worker 和调度器镜像

需要 Go 1.26、Docker、kubectl 和 kind。模块锁定 Kubernetes v1.34.0，已包含相应的 staging `replace`。从仓库根目录执行。

```bash
cd cloud-native/kubernetes/scheduling/scheduler-plugin
kind create cluster --name score-lab --config=- <<'EOF'
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
- role: worker
- role: worker
EOF
NODE_ARCH=$(kubectl --context kind-score-lab get node score-lab-worker -o jsonpath='{.status.nodeInfo.architecture}')
CGO_ENABLED=0 GOOS=linux GOARCH="$NODE_ARCH" go build -mod=readonly -o scheduler-plugin .
docker build --platform "linux/$NODE_ARCH" -t learning-notes-scheduler:dev .
kind load docker-image learning-notes-scheduler:dev --name score-lab
```

执行下方[安装清单](#安装清单)，然后回到下一步。

## 观察正向与反向落点

```bash
# 1. 确认调度器拿到了独立的 leader lease（不和 default-scheduler 抢锁）
kubectl --context kind-score-lab -n kube-system get lease learning-notes-scheduler
kubectl --context kind-score-lab -n kube-system logs deploy/learning-notes-scheduler | grep "successfully acquired lease"

# 2. 给两个 worker 打相反的 label
kubectl --context kind-score-lab label node score-lab-worker learning-notes/preferred=true  --overwrite
kubectl --context kind-score-lab label node score-lab-worker2 learning-notes/preferred=false --overwrite

# 3. 跑一批测试 Pod（用 Deployment 拉 6 个副本，比单 Pod 更能看出倾向）
cat <<EOF | kubectl --context kind-score-lab apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nl-test
spec:
  replicas: 6
  selector:
    matchLabels: { app: nl-test }
  template:
    metadata:
      labels: { app: nl-test }
    spec:
      schedulerName: learning-notes-scheduler   # 不写则走 default-scheduler
      containers:
        - name: pause
          image: registry.k8s.io/pause:3.10
EOF

# 4. 看落点分布
kubectl --context kind-score-lab rollout status deployment/nl-test --timeout=120s
kubectl --context kind-score-lab get pods -l app=nl-test -o custom-columns=NODE:.spec.nodeName --no-headers | sort | uniq -c
# 实测结果：6 个副本全部落在 preferred=true 的节点。

# 5. 确认是自定义调度器调的，且打分插件命中
kubectl --context kind-score-lab get events --field-selector reason=Scheduled | grep nl-test
kubectl --context kind-score-lab -n kube-system logs deploy/learning-notes-scheduler | grep "NodeLabelScore matched"
```

交换标签后滚动重建测试副本。

```bash
kubectl --context kind-score-lab label node score-lab-worker learning-notes/preferred=false --overwrite
kubectl --context kind-score-lab label node score-lab-worker2 learning-notes/preferred=true --overwrite
kubectl --context kind-score-lab rollout restart deployment/nl-test
kubectl --context kind-score-lab rollout status deployment/nl-test --timeout=120s
kubectl --context kind-score-lab get pods -l app=nl-test -o custom-columns=NODE:.spec.nodeName --no-headers
```

已有记录使用 v1.35 的 kind 节点、v1.34 调度器，正反两轮各有 6/6 个 Pod 落到带标签节点。该分数是一项偏好；实际结果还受过滤条件和其他插件影响。实验结束后执行 `kind delete cluster --name score-lab`。

## 从分数追到配置

[插件实现](pkg/nodelabel/nodelabel.go) 的 `Score` 直接接收本轮 `fwk.NodeInfo`，读取 `learning-notes/preferred`，返回 100 或 0。`NormalizeScore` 按本轮最高分归一化。[main.go](main.go) 用 `app.WithPlugin` 注册工厂，[config.yaml](config.yaml) 把它加入 Score 并设置权重 5。

升级时同步调整 Kubernetes 主模块与 staging 模块。遇到 `v0.0.0` 依赖错误，先核对本模块的 `replace`。还可以对照 [scheduler-plugins](https://github.com/kubernetes-sigs/scheduler-plugins) 同系列版本的模块配置。

## 安装清单

独立的 ServiceAccount、调度和卷绑定权限、Lease 权限、profile 和 Deployment 都在这份清单中。业务 Pod 显式选择 `learning-notes-scheduler`。

```bash
kubectl --context kind-score-lab apply -f - <<'EOF'
apiVersion: v1
kind: ServiceAccount
metadata:
  name: learning-notes-scheduler
  namespace: kube-system
---
# 调度核心权限：直接复用 K8s 内置 ClusterRole system:kube-scheduler
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: learning-notes-scheduler
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: system:kube-scheduler
subjects:
  - kind: ServiceAccount
    name: learning-notes-scheduler
    namespace: kube-system
---
# 卷调度权限（VolumeBinding 插件需要）
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: learning-notes-scheduler-volume
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: system:volume-scheduler
subjects:
  - kind: ServiceAccount
    name: learning-notes-scheduler
    namespace: kube-system
---
# ★ 关键易漏点：system:kube-scheduler 只授权了名为 "kube-scheduler" 的那个 lease。
# 我们用了独立 resourceName=learning-notes-scheduler，必须单独补一个 Role 授权这个新 lease，
# 否则 Pod 起来后 leader election 会一直 forbidden、选不出 leader。
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: learning-notes-scheduler-leaderelection
  namespace: kube-system
rules:
  - apiGroups: ["coordination.k8s.io"]
    resources: ["leases"]
    verbs: ["create"]
  - apiGroups: ["coordination.k8s.io"]
    resources: ["leases"]
    resourceNames: ["learning-notes-scheduler"]
    verbs: ["get", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: learning-notes-scheduler-leaderelection
  namespace: kube-system
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: learning-notes-scheduler-leaderelection
subjects:
  - kind: ServiceAccount
    name: learning-notes-scheduler
    namespace: kube-system
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: learning-notes-scheduler-config
  namespace: kube-system
data:
  config.yaml: |
    apiVersion: kubescheduler.config.k8s.io/v1
    kind: KubeSchedulerConfiguration
    leaderElection:
      leaderElect: true
      resourceName: learning-notes-scheduler
      resourceNamespace: kube-system
    profiles:
      - schedulerName: learning-notes-scheduler
        plugins:
          score:
            enabled:
              - name: NodeLabelScore
                weight: 5
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: learning-notes-scheduler
  namespace: kube-system
  labels: { app: learning-notes-scheduler }
spec:
  replicas: 1
  selector:
    matchLabels: { app: learning-notes-scheduler }
  template:
    metadata:
      labels: { app: learning-notes-scheduler }
    spec:
      serviceAccountName: learning-notes-scheduler
      containers:
        - name: scheduler
          image: learning-notes-scheduler:dev
          imagePullPolicy: IfNotPresent   # kind load 进来的本地镜像，别让它去 pull
          command:
            - /scheduler-plugin
            - --config=/etc/kubernetes/scheduler-plugin/config.yaml
            - --leader-elect=true
            - -v=4
          volumeMounts:
            - name: config
              mountPath: /etc/kubernetes/scheduler-plugin
              readOnly: true
      volumes:
        - name: config
          configMap:
            name: learning-notes-scheduler-config
EOF
kubectl --context kind-score-lab -n kube-system rollout status deployment/learning-notes-scheduler --timeout=120s
```

## 按日志区分两种权限问题

如果日志显示无法取得 `learning-notes-scheduler` Lease，检查 `kube-system` 中对应 Role、RoleBinding 和 Lease 名称。另一类历史告警是无法读取 `extension-apiserver-authentication` ConfigMap，它涉及调度器 HTTPS 端点的认证配置。先核对是否取得 Lease，以及测试 Pod 的 Scheduled 事件。历史版本见[实验环境](../../roadmaps/k8s-lab-environments.md)。

[Scheduler Framework](../scheduler-framework-source.md) · [调度器工作流程](../scheduler-deep-dive.md)
