# 推理服务：先量用户等了多久

做[流式请求实验](openai-compatible-benchmark/README.md)：固定模型和请求集，分别看首次响应、生成结束、实际输出 token 数。没有服务时可先跑客户端自测，它不给出引擎排名。

- **第一次部署**：先沿[一条请求的完整路径](llm-inference-pipeline.md)找出哪里在等，再用[同一模型比较 vLLM 和 SGLang](vllm-and-sglang.md)。
- **相似请求没变快**：用[输入字节对照](prefix-input-determinism/README.md)排除前缀差异，再查[缓存未命中案例](prefill-cache-miss.md)。回答不同则先用[采样实验](sampling/README.md)分开参数影响。
- **要改引擎**：带着一次请求去读 [vLLM V1](vllm-v1-architecture-and-internals.md) 或 [SGLang SRT](sglang-srt-radix-and-scheduler.md)。进入 GPU 算子前，确认实际选中的 attention 后端。

按阶段练习用[算子与双引擎路线](flashinfer-vllm-sglang-learning-path.md)；做跨节点服务时转到[分布式推理](llm-inference-learning-path.md)，用[实践表](llm-inference-progress.md)记录测量条件。

[返回 AI](../README.md)
