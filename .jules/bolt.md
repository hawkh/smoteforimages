
## 2024-03-24 - Distance Matrix Computation Optimization
**Learning:** In `ConstrainedSMOTE`, calculating pairwise distances row-by-row with `np.linalg.norm` creates a severe O(N × M) Python loop bottleneck during synthetic embedding filtering.
**Action:** Replace iterative norm calculations with batched `scipy.spatial.distance.cdist` (e.g., in `_filter_by_distance`). Batching (e.g., size 1000) is crucial when vectorizing to prevent OOM errors for large classes while maintaining the C-level performance benefits.
