## 2024-04-21 - [Vectorizing _filter_by_outlier_detectors]
**Learning:** Found a performance bottleneck in `ConstrainedSMOTE._filter_by_outlier_detectors` where it processes instances sequentially in Python loop calling scikit-learn `predict([emb])` one by one which introduces large overhead from scikit-learn's internal validations for each prediction call.
**Action:** Vectorize and batch predict the synthetic embeddings per class by feeding all synthetic samples for a class directly into the outlier detector's `predict` method at once.
