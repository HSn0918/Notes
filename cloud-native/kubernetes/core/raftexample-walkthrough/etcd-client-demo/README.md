# 从三次写入看懂 etcd revision

给同一个键依次写入 v1、v2、v3，然后读取第一次写入的 revision。另一个键绑定 5 秒租约，用最后一次 Get 检查是否过期。

## 启动本机的实验实例

需要 Docker 和本目录 go.mod 对应的 Go。客户端固定连接 127.0.0.1:2379，应指向新建的独立实验实例。

```bash
docker run -d -p 127.0.0.1:2379:2379 --name etcd-demo \
  quay.io/coreos/etcd:v3.5.18 \
  etcd \
  --advertise-client-urls=http://127.0.0.1:2379 \
  --listen-client-urls=http://0.0.0.0:2379
```

确认能连：

```bash
docker exec etcd-demo etcdctl endpoint status --write-out=table
```

## 运行客户端

```bash
cd cloud-native/kubernetes/core/raftexample-walkthrough/etcd-client-demo
go run -mod=readonly .
```

Watch 的范围只覆盖 `/learning-notes/etcd-demo/foo`。租约键使用独立的 Get 检查。

## 核对历史值与租约

```
[start] current etcd revision = 2, watch from rev=3
[put]   #0 value=v1 -> revision=3          ← 每次 Put 分配一个全局递增的 revision
[put]   #1 value=v2 -> revision=4
[put]   #2 value=v3 -> revision=5
[watch] rev=3 type=PUT key=/learning-notes/etcd-demo/foo value=v1
[watch] rev=4 type=PUT key=/learning-notes/etcd-demo/foo value=v2
[watch] rev=5 type=PUT key=/learning-notes/etcd-demo/foo value=v3
[get]   key=/learning-notes/etcd-demo/foo value=v3 create=3 mod=5 version=3 (lease=0)
                                              ↑ CreateRevision 是首次写入的 rev
                                                     ↑ ModRevision 是最近一次写入的 rev
                                                            ↑ Version 是该 key 的修改次数
[get@rev=3] value=v1                        ← MVCC 让我们能读历史 revision 的值
[lease] id=<id> ttl=5s key=/learning-notes/etcd-demo/lease-bound (5s 后被自动 revoke)
[lease] after TTL: /learning-notes/etcd-demo/lease-bound 还存在? false
```

示例假定全新的实例且只有本程序写入。Watch 与 Put 日志可能交错；请对照 revision 和键名。

## 回查 MVCC 与 Watch

| 看到的现象 | 对应源码 |
| --- | --- |
| 每次 `Put` 返回一个递增的 `revision` | `server/storage/mvcc/kvstore_txn.go` 的 `storeTxnWrite.put`，事务 End 时 `currentRev++` |
| `CreateRevision != ModRevision` | `put` 沿用之前的 `created.Main` 作为 `CreateRevision` |
| `Watch(WithRev(N))` 从 N（含）开始读事件 | 同 K8s `staging/.../etcd3/watcher.go` 的 `clientv3.WithRev(initialRev+1)` |
| `Get(WithRev(oldRev))` 能读出历史值 | MVCC：boltdb 物理 key 是 revision，老 revision 的 KeyValue 仍在 |
| Lease 过期后 key 自动消失（一条 DELETE 事件） | `server/lease/lessor.go`，Leader 发 `LeaseRevoke` 提案 |

## 结束实验

```bash
docker rm -f etcd-demo
```

2026-05 有 macOS 本地运行记录，见[实验环境](../../../roadmaps/k8s-lab-environments.md)。

[etcd 源码](../../etcd-source.md) · [返回控制面](../../README.md)
