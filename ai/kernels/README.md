# CUDA 与 FlashInfer：先验证算对，再测算快

先做[把 KV 拆开算再合回去](online-softmax-and-split-kv/README.md)：同一段历史序列分块前后，注意力输出应在约定误差内一致。CPU 就能检验公式；评估访存或执行时间则需要 GPU。

## 在 GPU 上量什么

从[一次设备事件计时](gpu-execution-and-memory-model.md#训练侧练习区分提交时间与设备时间)分清 CPU 提交与 GPU 执行。看不懂线程如何协作时读 [SIMD 与 SIMT](simd-and-gpu-simt-foundations.md)；准备分析访存或选实现时再查 [CUDA 工具链](cuda-ecosystem-and-career-roadmap.md)。

## 带着一条请求读 FlashInfer

1. [手算页表，再看 Plan/Run](flashinfer-intuition-and-engine-bridge.md)：这次请求要读哪些 KV 页？
2. [跟踪 Decode 与 Split-KV](flashinfer-paged-kv-and-decode-source.md)；输入长度不一时转到 [Ragged Prefill、Cascade 与 MLA](flashinfer-ragged-prefill-and-mla-source.md)。
3. 需要知道 kernel 如何生成和加载，查 [JIT 与绑定层](flashinfer-architecture-and-jit.md)。想从更小的算子入手，读 [Norm 与 Sampling](flashinfer-norm-and-sampling-source.md)。

这些源码笔记以 FlashInfer v0.7.3（`82caa5668`，审查于 2026-11-17）为边界。[实验工作流](flashinfer-learning-roadmap.md)区分本地数值验证和真机测量；上游怎么使用算子见[推理服务](../inference/README.md)。

[返回 AI](../README.md)
