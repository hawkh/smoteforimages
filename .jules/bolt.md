
## 2025-05-04 - [Quality Assessor Diversity Computation Optimization]
**Learning:** In PyTorch codebases where data is already batched into tensors and located on GPUs, relying on scikit-learn (`sklearn.metrics.pairwise.pairwise_distances`) forces an expensive `cpu().numpy()` transfer which can slow down evaluation significantly.
**Action:** Replace scikit-learn's generic euclidean distance computation with native PyTorch functionality (`torch.cdist` with `p=2`) and extract the upper triangle with `torch.triu_indices` to keep calculations batched and on-device. Ensure `unbiased=False` is passed to `torch.std` to maintain exact backward compatibility with numpy's `ddof=0` default.
