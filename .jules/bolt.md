## 2024-04-14 - Initial Setup
**Learning:** Initializing bolt journal.
**Action:** Always document critical learnings here.
## 2024-04-14 - Vectorized Distance Calculations
**Learning:** Python loops over O(N x M) nested operations with `np.linalg.norm` are massive performance bottlenecks. `scipy.spatial.distance.cdist` combined with chunked batching effectively resolves this without triggering out-of-memory errors.
**Action:** When calculating all-to-all pair distances or filtering via distance thresholds, use native `scipy` or `torch` batched vectorization.
