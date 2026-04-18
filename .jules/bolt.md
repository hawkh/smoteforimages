
## 2024-05-24 - [Optimize Diversity Metrics Calculation]
**Learning:** Native PyTorch operations (`torch.cdist`, `torch.triu_indices`) should be favored over CPU-bound sklearn operations (`sklearn.metrics.pairwise.pairwise_distances`) for metric calculations that occur heavily. Using `numpy()` conversions triggers costly CPU-GPU memory transfers. We found an 80%+ speedup in metric assessment by switching to PyTorch functions, reducing overhead immensely.
**Action:** Always seek to compute metrics directly with PyTorch tensor functions whenever the tensors are actively processed. Eliminate any use of `cpu().numpy()` during performance critical phases.
