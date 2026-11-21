# 用进程内存看懂配额如何生效

示例进程反复申请 1 MiB。Linux 动态库从环境变量读取配额并累计分配量，超限时返回 NULL。这个小程序让上层插件写进容器的配额有了可观察的结果。

## 在 macOS 上用 Linux 容器运行

需要 Docker，从仓库根目录执行。

```bash
cd cloud-native/kubernetes/scheduling/hami-mac/libvgpu-hook-demo
./run-in-docker.sh
```

脚本先设置 `MEM_LIMIT_MB=5`，再运行两秒无配额对照。看 `[hook] DENY malloc` 和 `victim: malloc returned NULL`。C 库自身的分配也可能占用配额，最后成功的轮数以输出为准。

## 在 Linux 上直接运行

在本实验目录内执行，需要 GCC 和 libdl。

```bash
gcc -shared -fPIC malloc-limit.c -o malloc-limit.so -ldl
gcc victim.c -o victim
MEM_LIMIT_MB=5 LD_PRELOAD=./malloc-limit.so ./victim
```

收到 NULL 后，示例程序以状态 1 退出。确认退出前打印了拒绝分配的记录。

## 对照代码读配额

[malloc-limit.c](malloc-limit.c) 在进程加载时读取 `MEM_LIMIT_MB`，包装 `malloc` 并累计成功分配的字节数。[victim.c](victim.c) 每次申请 1 MiB 并检查返回值。示例没有扣回已释放的内存，也没有实现并发分配的原子预留。

## 与 HAMi 的关系

外层插件示范如何通过 Allocate 传递配额。这里单独运行 CPU 进程，只检验分配失败的处理。复核于 2026-11-21，对照了 Linux 脚本和 C 源码；未收录此次容器运行结果。

[上一级 vGPU 切片实验](../README.md)
