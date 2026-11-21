# 工程基础：任务没启动，不一定是 GPU 不够

一个训练流水线有三个步骤：准备数据、训练、写 checkpoint。假如有人把最后一步的完成事件误接为第一步的前置条件，所有步骤都会等。用 Python 标准库看正常和异常两条路径：

```bash
python3 - <<'PYTHON'
from graphlib import CycleError, TopologicalSorter
deps = {"prepare": (), "train": ("prepare",), "save": ("train",)}
print(tuple(TopologicalSorter(deps).static_order()))
deps["prepare"] = ("save",)
try:
    tuple(TopologicalSorter(deps).static_order())
except CycleError as error:
    print("cycle:", error.args[1])
PYTHON
```

第一行给出合法的执行顺序；第二行打印依赖环。还没有可运行的任务，调度器即使有空卡也帮不上忙。用[拓扑排序](algorithm/topological-sort.md)解释这两次结果。

接下来按症状选：

- [算法与数据结构](algorithm/README.md)：依赖排序、队列、缓存淘汰该选什么结构。
- [数据库](database/README.md)：状态查询慢，先分清扫描、锁等待与缓存。
- [消息与一致性](middleware/README.md)：同一完成事件被消费两次，从发送、确认和提交位置逐段查。
- [终端操作](tools/README.md)：找回上一次的诊断命令，不用重打一遍。

[返回首页](../README.md)
