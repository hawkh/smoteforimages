## 2024-05-19 - [Optimize repulsion loss calculation]
**Learning:** In PyTorch, using explicit dimension expansion (`unsqueeze(0) - unsqueeze(1)`) for computing pairwise distances creates a massive intermediate tensor allocating O(K^2 * D) memory. This causes slow execution and high memory overhead, especially for large embedding sizes (D) or batches.
**Action:** When computing batched pairwise distances, always use `torch.cdist(x, x, p=2)` which performs the operation in O(K^2) memory complexity and uses optimized native backend computations.
