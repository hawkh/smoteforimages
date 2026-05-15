## 2024-05-15 - [PyTorch Memory Optimization]
**Learning:** Explicit dimension expansion using `unsqueeze(0) - unsqueeze(1)` for pairwise distances allocates O(K^2 * D) memory, which causes significant memory usage and latency in `_compute_repulsion_loss`.
**Action:** Use `torch.cdist(x, x, p=2)` for optimized, memory-efficient O(K^2) execution and avoid unnecessary large tensor allocations.
