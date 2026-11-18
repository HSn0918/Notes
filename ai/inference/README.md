# LLM Inference

> 返回 [AI 学习索引](../README.md)

研究请求调度、KV Cache 和算子选择，从 [FlashInfer × vLLM × SGLang 路线](flashinfer-vllm-sglang-learning-path.md) 进入。两条源码线分别在 [vLLM V1](vllm-v1-architecture-and-internals.md) 与 [SGLang SRT](sglang-srt-radix-and-scheduler.md)；具体 attention 后端以启动配置和实际调用为准。调度训练作业、切分模型参数则看[分布式训练路线](../training/README.md)。

## 阅读顺序

1. [LLM 推理全流程](llm-inference-pipeline.md) —— Messages 到 Response 的完整数据流。
2. [vLLM 与 SGLang](vllm-and-sglang.md) —— 两个推理引擎的最小使用、指标和设计重点。
3. [Prefill Cache Miss](prefill-cache-miss.md) —— 输入确定性与 Prefix Cache 的工程案例。
4. [LLM 推理进阶路线](llm-inference-learning-path.md) —— Scheduler、KV Cache、PD 分离和 Router。
5. [LLM 推理实践表](llm-inference-progress.md) —— 按实验和完成标准推进。

底层算子延伸：[GPU 算子与 FlashInfer](../kernels/README.md) —— PagedAttention、Split-KV Decode、Ragged/Chunked Prefill 与 Cascade/MLA 的 CUDA Kernel 源码导读与离线实验。

基础页负责稳定概念；进阶路线中的命令、指标与源码路径容易随版本变化，使用时应结合官方文档核对。
