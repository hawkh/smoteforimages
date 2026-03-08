
## 2025-03-08 - O(N²) Performance Bottleneck in ConstrainedSMOTE Filtering
**Learning:** Iteratively calculating pairwise distances in python via a `for` loop on thousands of samples across batches is extremely slow (O(N_synthetic * N_original_per_class) scaling). Vectorizing operations with optimized library calls like `sklearn.neighbors.NearestNeighbors` while grouping arrays by unique classes mitigates this issue profoundly.
**Action:** When filtering or comparing large synthetic data clusters against an original reference, always attempt to reshape the operation as a batch or grouped operation that can leverage highly optimized C backends like KDTree/BallTree under the hood. Avoid naive iterative `for` loops per-sample whenever possible.
