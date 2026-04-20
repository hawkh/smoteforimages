## 2025-02-04 - [ConstrainedSMOTE Outlier Detection Optimization]
**Learning:** Making repeated single-sample `.predict([emb])` calls on scikit-learn models (like IsolationForest or LocalOutlierFactor) inside an unoptimized Python loop creates significant performance overhead due to model/data validation checks.
**Action:** Always batch predictions by class using `detector.predict(embs_to_check)` instead of predicting one embedding at a time when filtering or evaluating synthesized datasets.
