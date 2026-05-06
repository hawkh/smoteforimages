## 2025-05-06 - [Optimize diversity metrics computation]
**Learning:** Replacing CPU-bound `sklearn.metrics.pairwise.pairwise_distances` with GPU-native `torch.cdist` along with `torch.triu_indices` improves performance for diversity metrics computation. PyTorch's `std` is unbiased by default (`unbiased=True`), so pass `unbiased=False` to match numpy's biased default (`ddof=0`).
**Action:** Always prefer native PyTorch GPU/CPU functions over scikit-learn metrics when computing distances on tensors to avoid transferring data to the CPU.
