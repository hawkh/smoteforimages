
## 2024-05-24 - [Distance Calculations in High-Dimensional Spaces]
**Learning:** O(N x M) nested Python loops for distance calculations between sets of high-dimensional embeddings in `ConstrainedSMOTE` (like in `_filter_by_distance`) cause extreme performance bottlenecks and memory issues. The original loop approach computed `np.linalg.norm` sample-by-sample and scaled poorly.
**Action:** Always replace sample-by-sample distance calculations between large sets of vectors with batched, vectorized functions like `scipy.spatial.distance.cdist`. When doing so, ensure that operations process in manageable chunks (e.g. 2000 items) to prevent Out-Of-Memory (OOM) errors.
