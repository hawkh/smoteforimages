
## 2024-10-24 - Optimize `_filter_by_distance` with Vectorization
**Learning:** O(N x M) nested Python loops for distance computation in large datasets are a critical performance bottleneck and can cause Out-Of-Memory (OOM) errors.
**Action:** Replace nested loops with batched, vectorized `scipy.spatial.distance.cdist` calculations processed in chunks to drastically improve performance and manage memory efficiently.
