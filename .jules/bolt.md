
## 2023-10-25 - [Vectorized Distance Filtering Optimization]
**Learning:** When replacing O(N x M) nested loops with vectorized `scipy.spatial.distance.cdist` or `torch.cdist` in Python, processing the entire dataset at once for large classes can easily cause Out-Of-Memory (OOM) errors.
**Action:** Always batch vectorized distance computations (e.g., in chunks of 1000) when working with high-dimensional embeddings to balance CPU/GPU memory constraints and performance.
