## 2025-02-18 - [Optimization of Cluster Determination]
**Learning:** `ConstrainedSMOTE._determine_optimal_clusters` was running KMeans multiple times (k=2..5) with `n_init=10` per class during `fit`. This is expensive and unnecessary for the elbow method. Reducing `n_init=1` provided ~7x speedup for this method.
**Action:** When implementing heuristic checks like elbow method, use lower precision settings (e.g., `n_init=1`) as high precision is not required for the heuristic itself.

## 2025-02-18 - [SMOTE on Balanced Data]
**Learning:** `imbalanced-learn`'s SMOTE with `sampling_strategy='auto'` generates 0 samples if classes are balanced. This caused test flakiness when random data happened to be balanced. Also, `generate_synthetic` was ignoring `n_samples`.
**Action:** Ensure tests force imbalance if testing SMOTE. Also, handle empty synthetic generation gracefully or respect user intent if possible.
