# 数据库：一次查询到底慢在哪里

从一个真实的过滤条件开始，比如“按作业 ID 查状态”。先看执行计划用了什么索引，再判断是读了太多行，还是在等另一个事务。不要一看到慢查询就先加缓存。

**MySQL｜查询与写入。** 看[索引与回表](mysql-index.md)，再对照[存储引擎](mysql-engine.md)。两个会话读到不同的值查[事务与 MVCC](mysql-transaction.md)；写入互相等待查[行锁与间隙锁](mysql-lock.md)。

**Redis｜缓存与恢复。** 从[数据类型](redis-data-types.md)确认存的是什么。热点过期查[缓存击穿](redis-cache.md)，重启后数据变化查[RDB 与 AOF](redis-persistence.md)，故障切换则查[复制与集群](redis-cluster.md)。

**PostgreSQL｜会话与查询。** [进程与连接](postgresql-basics.md)解释谁在执行；[窗口函数、锁与复制](postgresql-advanced.md)解释一次具体查询或等待。

**Elasticsearch｜检索结果。** 在独立环境[运行 ES 与 Kibana](es-kibana-docker.md)，用同一个值比较 [text 与 keyword](es-field-types.md)，再到[分片与索引](elasticsearch-basics.md)查请求去了哪里。

[返回工程基础](../README.md)
