## 2023-10-27 - [Avoid Explicit Dimension Expansion for Pairwise Distances]
**Learning:** Using explicit dimension expansion (`unsqueeze(0) - unsqueeze(1)`) to compute pairwise distances in PyTorch allocates an intermediate tensor of size `O(K^2 * D)`, leading to high memory usage and increased latency, particularly with large embeddings or batch sizes.
**Action:** Always use `torch.cdist(x, x, p=2)` for optimized, memory-efficient `O(K^2)` pairwise distance calculations in PyTorch.
