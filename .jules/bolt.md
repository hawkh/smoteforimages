## 2024-05-19 - Fast Diversity Computation with torch.cdist
**Learning:** In `QualityAssessor`, computing diversity metrics natively on tensors using `torch.cdist(flattened, flattened)` avoids CPU and NumPy data transfers, providing a significant (~3x) speedup compared to using `sklearn.metrics.pairwise_distances`.
**Action:** Use PyTorch native pairwise distance computations (`torch.cdist`, `torch.pdist`) when calculating diversity or similarities over embedding/image batches to keep execution on the GPU or avoid costly `.cpu().numpy()` conversions.
