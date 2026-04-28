## 2024-05-24 - [Optimize _filter_by_distance in ConstrainedSMOTE]
**Learning:** `ConstrainedSMOTE._filter_by_distance` uses an iterative `np.linalg.norm` approach, representing an unoptimized bottleneck that can be vectorized with `scipy.spatial.distance.cdist`.
**Action:** Vectorize distance calculations per label using `scipy.spatial.distance.cdist` to provide significant speedup instead of looping over each synthetic embedding.
