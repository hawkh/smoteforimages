## 2026-04-30 - PyTorch Standard Deviation Defaults vs NumPy
**Learning:** When migrating mathematical calculations from NumPy to PyTorch, such as in quality assessment metrics, be aware that `numpy.std()` defaults to a biased estimator (Delta Degrees of Freedom = 0), whereas `torch.std()` defaults to Bessel's correction (unbiased=True). This subtle difference can cause exact test assertions to fail when swapping CPU NumPy calculations for GPU PyTorch calculations.
**Action:** Always explicitly pass `unbiased=False` to `torch.std()` when refactoring NumPy standard deviation logic to ensure exact mathematical equivalence.

## 2026-04-30 - GPU-Bound Indexing
**Learning:** When using PyTorch index generators like `torch.triu_indices` to extract specific matrix elements (e.g., upper triangle values of a pairwise distance matrix), you must explicitly pass the `device` argument of the source tensor. By default, index generation happens on the CPU, causing device mismatch errors when used to slice GPU tensors.
**Action:** Always supply `device=tensor.device` when generating indices in PyTorch (e.g., `row_idx, col_idx = torch.triu_indices(size, size, offset=1, device=tensor.device)`).
