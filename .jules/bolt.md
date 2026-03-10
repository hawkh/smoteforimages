## 2025-02-23 - Optimize distance filtering in ConstrainedSMOTE
**Learning:** Calculating distance metrics via iterative `for` loops with `np.linalg.norm` creates a severe O(N*M) bottleneck in Python for embedding spaces, taking ~60 seconds to process 10k samples. This blocks efficient generation of large synthetic batches.
**Action:** Replace iterative array operations with `sklearn.neighbors.NearestNeighbors(n_neighbors=1, n_jobs=-1)` to calculate min-distances using vectorized operations and compiled C backends, achieving a >150x speedup (~0.3 seconds for 10k samples).
