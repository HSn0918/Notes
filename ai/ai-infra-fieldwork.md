#ai #ai-infra #学习路线

相关笔记：[[training-step-and-gradients]] · [[training-job-placement]] · [[flashinfer-intuition-and-engine-bridge]]

# AI Infra 实战路线：交付一个能训练、能恢复、能服务的模型

适合会运行 Python 脚本、刚开始做训练平台的人。先拿一份可比的单步更新，再把它交给多进程和调度器；最后判断恢复后是否重复算了数据，以及服务延迟量的到底是什么。以下命令都从仓库根目录执行。没有 GPU 也可以从今天开始，真机部分另行验收。

## 第一站：把两份梯度合成一次更新

```bash
python3 ai/training/sharded-gradient/sharded_gradient.py
python3 ai/training/sharded-gradient/sharded_gradient.py --left-tokens 2
```

先遮住结果，预测把 4 个有效位置切成 1:3 和 2:2 两份时，哪一种可以直接平均两边的梯度。运行后比较 `naive_error` 与 `weighted_error`；只抄最后的 PASS 不算完成。**交付**：两行对照记录，以及“权重该按什么算”的一句解释。看不懂形状才回查[矩阵乘](foundations/linear-algebra-for-llm.md)，推不出分母才看[反向传播](foundations/training-step-and-gradients.md)。

## 从单卡扩到多机

1. **先把一次更新算对。** 有 PyTorch 时，把上面的两种切分搬到[双进程 DDP](training/ddp-step/README.md)。记录有效 token 数和更新后的最大权重误差。只有 CPU 就用 Gloo，拿到同一节点的两张卡后再测 NCCL。
2. **量出一步训练的时间。** 用[矩阵分块实验](foundations/llm-math-and-simd-lab/README.md)理解尾块和布局，再用 [CUDA 事件](kernels/gpu-execution-and-memory-model.md#训练侧练习区分提交时间与设备时间)量单卡执行时间。多机上保持消息大小不变，对比机内与跨机的 AllReduce，并查最后到达的 rank 是否卡在数据读取。[训练框架与通信](training/distributed-training-frameworks.md)
3. **让调度器交付框架需要的布局。** 用[64 卡放置与队列回放](../cloud-native/kubernetes/scheduling/placement-lab/README.md)比较整机空闲和碎片空闲，再比较大、小作业各等了多久。进入测试集群后，把准入事件、Pod 所在节点和每个 rank 的设备列表放在一起核对。[GPU 调度](../cloud-native/kubernetes/scheduling/README.md)
4. **中断一次，再检查进度。** 将[连续训练与恢复训练](training/checkpoint-replay/README.md)的下一批样本、优化步和权重对齐。若 checkpoint 存在共享卷上，还要验证[重新挂载与读回](../cloud-native/kubernetes/storage/README.md)。单独记录恢复时花在排队、加载和重算上的时间。
5. **逐级验收扩展效率。** 先用[64 卡 rank 分组算例](training/README.md)画出通信组，再扩大节点数和网络域。比较时固定模型、有效全局 batch 和序列分布，同时记录排队 P95、有效 tokens/s、扩展效率及失败后重算量。万卡容量是规划目标，实测结论只写实际运行的规模。

## 训练产物上线后再测推理

先[手算 Split-KV](kernels/online-softmax-and-split-kv/README.md)，用分块归约误差检验 FlashInfer 算子的数学原理。有服务时再用[同一批请求比较 vLLM 和 SGLang](inference/openai-compatible-benchmark/README.md)，保留请求记录、首个流事件延迟、输出吞吐和实际选中的注意力后端。追一次请求经过哪些模块，可接着读[推理引擎源码路线](inference/flashinfer-vllm-sglang-learning-path.md)。

## 给下一次排障留一条记录

每次只记：**输入/版本 → 我的预测 → 唯一改变的条件 → 原始输出 → 结论不适用于哪里**。比如“1:3 切分、等权平均；最大误差约 0.232；这证明目标加权错了，不说明网络慢”。下一次学习先合上答案换一种切分，重新预判；若判断错误，就把错因和辨别它的证据补回原笔记，不再新开一套知识树。完整的[笔记方法与研究依据](../docs/learning-by-doing.md)。
