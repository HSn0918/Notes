# 定位 Pod 网络断在哪一跳

先做 [bridge 实验](cni-bridge/README.md)。对同一网络命名空间执行 ADD、DEL，分别看接口、地址和路由；不要用一次 ping 推断整个集群网络正常。

1. **Pod 没有 IP**：从[网络模型](network-model.md)和 [CNI 协议](cni.md)入手；需要看实现再查 [CNI 组件](cni-plugin-component.md)和[源码](cni-source.md)。
2. **Pod IP 可达，服务名不可用**：区分 [CoreDNS](coredns-component.md) 解析、[普通 Service](service.md)、[Headless Service](headless-service.md) 和 [kube-proxy 转发](kube-proxy.md)。组件边界见 [kube-proxy 职责](kube-proxy-component.md)。
3. **集群内正常，从外面访问失败**：查[入口控制器](ingress-controller-component.md)。仍定位不了，照 [CNI 排障顺序](cni-troubleshooting.md)逐跳排除。

按集群实际插件选读：[Calico](calico.md)、[Flannel](flannel.md)、[Cilium](cilium.md)及其[数据面](cilium-deep-dive.md)、[Weave](weave.md)、[Multus](multus.md)。

[返回 Kubernetes](../README.md)
