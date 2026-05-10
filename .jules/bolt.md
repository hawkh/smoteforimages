## 2024-05-10 - PyTorch Pairwise Distance Memory Optimization
**Learning:** In PyTorch, computing pairwise distances by explicit dimension expansion (`unsqueeze(0) - unsqueeze(1)`) before calculating the norm allocates an intermediate tensor of size O(K^2 * D). This causes significant memory bloat and latency bottlenecks during operations like computing repulsion loss.
**Action:** Use `torch.cdist(x, x, p=2)` for optimized, memory-efficient pairwise distance computations, which evaluates in O(K^2) without materializing the large intermediate tensor.
