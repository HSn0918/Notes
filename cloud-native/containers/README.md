# 容器：从一个训练镜像开始

先用 [Docker 常用命令](docker-commands.md)对照一个已有镜像和由它启动的容器，分别找出镜像 ID、进程和挂载点。没有本地环境时先读 [Docker 基础](docker-basics.md)，不必拉训练模型练手。

- **镜像越来越大**：从 [Dockerfile 的每一层](dockerfile.md)看哪些文件被保留，再读 [UnionFS](union-fs.md) 和 [Buildah 大镜像案例](buildah-large-image.md)。
- **容器看到的进程和宿主机不同**：看 [Namespace](namespace.md)；进程能看到 CPU 却用不满时，区分可见性和 [Cgroup 限制](cgroup.md)。
- **不知道包从哪张网卡出去**：对照 [Bridge](network-bridge.md)、[Underlay](network-underlay.md) 与 [Null 网络](network-null.md)。

[返回云原生](../README.md)
