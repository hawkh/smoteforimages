## 2024-04-10 - O(N*M) distance calculation bottleneck in SMOTE
**Learning:** `ConstrainedSMOTE._filter_by_distance` used an O(N*M) iterative approach to calculate distances between *each* synthetic point and all real points. For N=10,000 and M=10,000, this takes ~80s.
**Action:** Replace Python iterative distance calculations with vectorized operations grouped by unique labels, leveraging `sklearn.neighbors.NearestNeighbors`, which reduces computation time for N=10,000, M=10,000 from ~80s to <1s.
