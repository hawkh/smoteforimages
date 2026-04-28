## 2024-05-24 - [Avoid Device Sync Overheads in Distances]
**Learning:** Computing distance metrics natively using PyTorch `torch.cdist` on the CPU or GPU is dramatically faster (almost 2x speedup on CPU and 2.5x speedup generally) than converting tensors to numpy arrays and using `sklearn.metrics.pairwise.pairwise_distances`, because it avoids CPU-GPU synchronization and data transfers.
**Action:** Always prefer native PyTorch operations for matrix and distance computations instead of dropping down to numpy or scikit-learn.
