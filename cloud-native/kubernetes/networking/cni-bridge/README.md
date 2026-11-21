# 用 Linux 网桥连通两个模拟 Pod

在两个网络命名空间内各放一端 veth，让它们通过同一座 cni0 网桥通信。读 ADD 返回的 IP，核对第二个命名空间实际接收了第一个发来的包。

## 在 Docker 的 Linux 环境运行

从仓库根目录执行。需要 Docker、可拉取 Ubuntu 镜像，并允许脚本使用特权容器创建 netns 和网桥。

```bash
cd cloud-native/kubernetes/networking/cni-bridge
./run-in-docker.sh
```

先看各自的 `eth0` 和 ADD JSON，再看两端互 ping。具体 IP 取决于 IPAM 计数器。脚本结束时执行 DEL 并退出临时容器。

## 从 ADD 到 DEL

1. 调用方通过环境变量传命令、接口名和 netns，通过 stdin 传 JSON 配置。
2. 插件准备网桥并分配 IP，将 veth 一端接桥，另一端放入目标 netns。
3. 对端 ping 成功后执行 DEL，检查主机侧 veth 消失。单独的公网 ping 还受 Docker 网络影响。

## 在独立 Linux 测试机逐步看接口

进入本实验目录。需要 root、iproute2、iptables 和 jq，且测试机上的 `cni0`、`pod-1` 与实验子网没有其他用途。这个脚本调用 `ip -n` 时用的是 netns 名称。

```bash
NETCONF='{"cniVersion":"0.4.0","name":"learn","type":"learning-bridge","subnet":"10.244.0.0/24","gateway":"10.244.0.1","bridge":"cni0"}'

ip netns add pod-1
CNI_COMMAND=ADD CNI_CONTAINERID=abc1 CNI_NETNS=pod-1 CNI_IFNAME=eth0 \
  ./learning-bridge <<< "$NETCONF"

ip netns exec pod-1 ip a
```

结束后仅清理这次创建的命名空间、网桥和 NAT 规则。

```bash
CNI_COMMAND=DEL CNI_CONTAINERID=abc1 CNI_NETNS=pod-1 CNI_IFNAME=eth0 ./learning-bridge <<< "$NETCONF"
ip netns del pod-1
ip link del cni0
iptables -t nat -D POSTROUTING -s 10.244.0.0/24 ! -o cni0 -j MASQUERADE
```

## 实现边界

示例只处理单机 IPv4，IPAM 依赖本地计数文件，CHECK 固定返回成功。真实运行时通常传入 netns 路径；若脚本在 `ip -n /run/netns/...` 处失败，记录该错误，再用上面的命名空间名称逐步核对。本页没有新的容器连通性实测。历史环境见[实验记录](../../roadmaps/k8s-lab-environments.md)。

[CNI 走读](demo-cni-bridge.md) · [CNI 源码](../cni-source.md)
