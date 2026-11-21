# 跟一次模型工具调用

打开[工具调用源码导读](agent-development-source.md)，选一个例子，只记三样东西：模型请求了什么参数、工具实际返回什么、模型下一轮看到了什么。这样才能分清是工具执行错了，还是模型拿到正确结果后解释错了。

- **刚开始写 Agent**：走[开发路线](agent-development-learning-path.md)，先做一次调用，再加检索和多步流程。
- **已经能调用工具**：用[生产案例](production-agent-development.md)检查重复执行、权限和失败后的重试。不要用一次回答正确代替整条调用链的验证。

[返回 AI](../README.md)
