## 2024-05-07 - [Optimization of ConstrainedSMOTE._filter_by_distance]
**Learning:** O(N x M) nested Python loops for calculating distances between large sets of embeddings (like original and synthetic embeddings) create significant inference latency bottlenecks.
**Action:** Replace unoptimized Python loops with batched, vectorized `scipy.spatial.distance.cdist` calculations, processed in manageable chunks (e.g., 2000 items) to vastly improve performance while avoiding OOM errors with large arrays.
