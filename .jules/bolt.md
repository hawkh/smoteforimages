## 2024-04-19 - [QualityAssessor Diversity metrics]
**Learning:** In `QualityAssessor`, diversity metrics computation was previously optimized to use native `torch.cdist` and `torch.triu_indices` instead of `sklearn.metrics.pairwise.pairwise_distances` via NumPy to eliminate CPU transfer overhead, but the codebase does not reflect this change yet. Wait, I should implement this optimization.
**Action:** Replace `pairwise_distances(flattened, metric='euclidean')` which uses NumPy/scikit-learn with `torch.cdist` directly on GPU tensors.
## 2024-04-19 - [ConstrainedSMOTE filter_by_distance]
**Learning:** The memory mentions that `_filter_by_distance` was optimized using `scipy.spatial.distance.cdist` instead of `np.linalg.norm` in a nested python loop. But it seems the codebase still uses the O(N × M) nested loop. I should optimize this.
**Action:** Replace the loop in `_filter_by_distance` with a vectorized calculation using `scipy.spatial.distance.cdist`.
