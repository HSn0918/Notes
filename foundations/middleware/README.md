# 消息系统：跟踪一条任务完成事件

给事件一个 ID，在纸上标出生产者发出、服务端确认、消费者处理、业务结果写入四个时刻。只知道“消息发过了”，还不能断定任务状态已经更新。

- **Kafka**：从[一条消息的路径](kafka-basics.md)入手，对照[分区选择](producer-partition.md)与[确认和丢失边界](zero-message-loss.md)。要解释带宽或调优才查[消息压缩](producer-compression.md)与[集群配置](cluster-config.md)。
- **NATS**：先区分[即时发布订阅](nats-basics.md)和 [JetStream 的持久化与重放](nats-jetstream.md)，再判断订阅者离线期间会错过什么。
- **数据库变更与业务结果不同步**：从 [Canal 读取 binlog](canal.md)追源事件。要解决多个系统的提交或补偿问题，再比较[分布式事务方案](distributed-transaction.md)。

能说清各段证据后，再用 [Kafka 复习题](kafka-interview.md)检查遗漏。

[返回工程基础](../README.md)
