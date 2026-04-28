
## 2024-05-18 - [Optimized Diversity Metric Computation]
**Learning:** Computing diversity metrics natively with `torch.cdist` and `torch.triu_indices` avoids expensive PyTorch-to-NumPy (CPU/GPU syncs and data transfers).
**Action:** When working with image tensors already on GPU or managed by PyTorch, prefer PyTorch's native distance implementations (`cdist`) over converting to numpy and using `sklearn.metrics.pairwise_distances`. Be mindful of `torch.std(unbiased=False)` matching numpy's default population standard deviation, and explicitly specify `device` for tensor initializations like indices to prevent unexpected host/device synchronization overhead.
