## 2025-02-04 - Memory Bottleneck in PyTorch Pairwise Distance Computations
**Learning:** Using explicit dimension expansion (`unsqueeze(0) - unsqueeze(1)`) to compute pairwise distances in PyTorch allocates an intermediate tensor of size O(K^2 * D), which causes significant memory usage and can lead to out-of-memory errors on large batch sizes.
**Action:** Always use `torch.cdist(x, x, p=2)` for optimized, memory-efficient O(K^2) execution of pairwise distance calculations, making sure to handle data type casting (`.float()`) properly.
