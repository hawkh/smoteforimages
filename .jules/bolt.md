## 2024-05-03 - [Optimize Pairwise Distances in PyTorch]
**Learning:** Computing pairwise distances via explicit broadcasting (`unsqueeze(0) - unsqueeze(1)`) creates an intermediate tensor of size O(K^2 * D), which causes massive memory spikes and OOM errors for large batch sizes.
**Action:** Always use `torch.cdist(x, x, p=2.0)` instead. It is mathematically equivalent, heavily optimized in C++/CUDA, and does not instantiate the O(K^2 * D) intermediate tensor.
