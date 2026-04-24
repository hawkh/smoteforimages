
## 2024-05-18 - [Optimizing distance filtering in ConstrainedSMOTE]
**Learning:** The `_filter_by_distance` method in `ConstrainedSMOTE` has an unoptimized O(N x M) nested loop where it computes the L2 norm individually for each synthetic sample against all original class samples using `np.linalg.norm(..., axis=1)`. Since there are multiple calls in the inner loop, the Python interpretation overhead is significant.
**Action:** Replace the per-item `np.linalg.norm` distance loop with a vectorized, batched computation using `scipy.spatial.distance.cdist` (processed in manageable chunks like 2000 to avoid memory overhead). Based on benchmarks, this improves performance by more than ~25x (from ~3.6s to ~0.14s for 2,000 queries over 10,000 points) and successfully eliminates the CPU overhead.
