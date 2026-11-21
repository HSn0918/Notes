# 算法：按手头的问题选工具

先做[工程基础首页的依赖环实验](../README.md)，再读[拓扑排序](topological-sort.md)。把一条边反过来，画出再也不能降到零的入度。其余算法不用按课表通读。

## 查图、路径与候选解

- 按层找最近节点用 [BFS](bfs.md)，沿一条路径找到底用 [DFS](dfs.md)；[图遍历案例](graph-bfs-dfs.md)对照两者。
- 边带非负权时查 [Dijkstra](dijkstra.md)；反复判断两个节点是否连通查[并查集](union-find.md)；树上找共同祖先查 [LCA](lca.md)。
- 在有序区间找边界，先写出[二分查找](binary-search.md)的不变量。要枚举选择并撤销，则是[回溯](backtracking.md)。

## 查序列、区间与状态

- 多次查区间和用[前缀和](prefix-sum.md)，多次改一段区间用[差分](diff-array.md)；窗口边界移动查[滑动窗口](sliding-window.md)，找下一个更大值查[单调栈](monotonic-stack.md)。
- 从[不能相邻选取](house-robber.md)练习状态转移，再对比[可以重复选取](unbounded-knapsack.md)。位运算按需要选[异或](xor.md)或[按位与](bitwise-and.md)。
- 需要排序先比较[稳定性、空间和输入分布](sorting-overview.md)，再选[快排](quick-sort.md)、[归并](merge-sort.md)、[堆排](heap-sort.md)或[基数排序](radix-sort.md)。

## 查缓存与字符串

- 对访问序列 `A B A C` 手推一次淘汰，区分[最近没用](lru.md)和[使用频率低](lfu.md)；维护顺序时回查[链表](linked-list.md)。
- 一批词共享前缀查 [Trie](trie.md)，单个模式的连续匹配查 [KMP](kmp.md)。

[返回工程基础](../README.md)
