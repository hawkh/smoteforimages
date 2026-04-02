## 2025-03-01 - [Avoid CPU/NumPy Transfers in QualityAssessor]
**Learning:** Computing diversity metrics natively using `torch.cdist` avoids CPU/NumPy transfers (~2.5x speedup).
**Action:** When replacing NumPy with PyTorch, ensure `torch.std` uses `unbiased=False` to match `np.std`'s default behavior, and use `device=tensor.device` in functions like `torch.triu_indices` to prevent CPU-GPU synchronization overhead.
