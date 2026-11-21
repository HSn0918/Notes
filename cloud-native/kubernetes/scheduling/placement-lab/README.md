# 64 张空闲 GPU，能启动几个训练作业

每个作业有 8 个 worker，每个 worker 必须在同一台机器拿齐 8 张卡。下面两个实验分别计算能放多少个完整作业，以及排队时谁先开跑。只需 Python 3.9+，从仓库根目录执行。

## 先看空闲卡分布

```bash
python3 cloud-native/kubernetes/scheduling/placement-lab/placement.py
python3 cloud-native/kubernetes/scheduling/placement-lab/placement.py --gpus-per-worker 4
```

比较 `packed` 和 `fragmented`。8 台各余 8 卡能开一组，16 台各余 4 卡只能得到 `pooled_jobs=1`、`node_fit=0`。第二条命令把每个 worker 改为 4 卡，看看后一种布局现在能容纳多少组。

然后看 `split`。空机分属两个域时，要求 8 个 worker 同域会进一步收紧上界。`fleet` 的万卡快照中，总量估算为 23 组，按整机算至多 15 组。

## 再看哪个作业会多等

```bash
python3 cloud-native/kubernetes/scheduling/placement-lab/queue_replay.py
```

八台机器中，有四台占用到时刻 6。大作业在时刻 1 申请全部八台。时刻 2 又来了两个作业，一个只需两台、跑两个时间单位，另一个需要四台、跑八个时间单位。

| 策略 | 大作业等待 | 短作业等待 |
| --- | ---: | ---: |
| 按到达顺序 FIFO | 5 | 8 |
| 保护队首预约的回填 | 5 | 0 |
| 有空位就放的 greedy | 11 | 0 |

短作业可以在预约前结束。长作业一旦提前开跑，大作业就得等它释放资源。读输出时同时看每个作业的等待和整批结束时间。

## 验证范围

2026-11-21 在 macOS arm64 CPU 回放上述合成输入。容量计算使用静态快照，排队模型使用已知运行时长。真实训练还要检查队列准入、Pod 分配和 rank 会合。

[训练作业放置](../training-job-placement.md) · [断点续训](../../../../ai/training/checkpoint-replay/README.md) · [完整路线](../../../../ai/ai-infra-fieldwork.md)
