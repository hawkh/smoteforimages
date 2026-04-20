## 2025-02-23 - [Distance Calculation Vectorization]
**Learning:** Python iterative `for` loops combined with `np.linalg.norm(..., axis=1)` for distance thresholding per synthetic sample scale poorly.
**Action:** Use vectorised operations via `scipy.spatial.distance.cdist` grouped by class labels. This compares matrices of synthetic and real embeddings simultaneously, avoiding the Python loop overhead and resulting in massive speedups during the `_filter_by_distance` step in SMOTE.
