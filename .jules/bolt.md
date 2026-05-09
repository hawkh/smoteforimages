## 2026-05-09 - [Repulsion Loss Optimization]
**Learning:** Explicitly computing pairwise distances using `unsqueeze(0) - unsqueeze(1)` in PyTorch expands an intermediate tensor of shape `[K, K, D]`. When K or D is large, this O(K^2 * D) memory allocation can become a severe bottleneck.
**Action:** Replaced with `torch.cdist(x, x, p=2.0)`, which is highly optimized under the hood, computing pairwise Euclidean distance directly without materialising the fully expanded difference tensor.
