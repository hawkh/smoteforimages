## 2025-05-24 - [Distance filtering bottleneck]
**Learning:** `_filter_by_distance` calculates Euclidean distances between each synthetic sample and the entire subset of base class embeddings individually in a Python loop, which leads to slow generation performance, especially when `max_distance_threshold` is enabled. Batching the prediction using `scipy.spatial.distance.cdist` per class provides >10x speedup.
**Action:** Always check `for` loops in SMOTE generators that involve single-item array iterations, replacing them with vectorized `scipy` or `numpy` functions whenever possible.
