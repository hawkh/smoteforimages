import time
import numpy as np
from sklearn.ensemble import IsolationForest

def filter_unoptimized(synthetic_embeddings, synthetic_labels, detectors, work_embeddings, class_embs):
    rng = np.random.default_rng(42)
    max_attempts = 3
    valid_mask = np.ones(len(synthetic_embeddings), dtype=bool)

    # Mock slerp
    def _slerp(a, b, t):
        return a * (1-t) + b * t

    for i, (emb, label) in enumerate(zip(synthetic_embeddings, synthetic_labels)):
        label_int = int(label)
        if label_int not in detectors:
            continue

        detector = detectors[label_int]
        score = int(detector.predict([emb])[0])

        if score == -1:
            ce = class_embs[label_int]
            replaced = False
            for _ in range(max_attempts):
                if len(ce) < 2:
                    break
                idx_a = int(rng.integers(len(ce)))
                idx_b = int(rng.integers(len(ce)))
                if idx_a == idx_b: continue
                t_new = float(rng.uniform(0.0, 1.0))
                new_emb = _slerp(ce[idx_a], ce[idx_b], t_new)
                if int(detector.predict([new_emb])[0]) == 1:
                    synthetic_embeddings[i] = new_emb
                    replaced = True
                    break
            if not replaced:
                valid_mask[i] = False

    return synthetic_embeddings[valid_mask], synthetic_labels[valid_mask]

def filter_optimized(synthetic_embeddings, synthetic_labels, detectors, work_embeddings, class_embs):
    rng = np.random.default_rng(42)
    max_attempts = 3
    valid_mask = np.ones(len(synthetic_embeddings), dtype=bool)

    # Mock slerp
    def _slerp(a, b, t):
        return a * (1-t) + b * t

    unique_labels = np.unique(synthetic_labels)
    for label in unique_labels:
        label_int = int(label)
        if label_int not in detectors:
            continue

        class_mask = synthetic_labels == label_int
        class_indices = np.where(class_mask)[0]
        if len(class_indices) == 0:
            continue

        detector = detectors[label_int]
        embs_to_check = synthetic_embeddings[class_indices]
        scores = detector.predict(embs_to_check)

        outlier_local_indices = np.where(scores == -1)[0]
        if len(outlier_local_indices) == 0:
            continue

        ce = class_embs[label_int]
        if len(ce) < 2:
            valid_mask[class_indices[outlier_local_indices]] = False
            continue

        for local_idx in outlier_local_indices:
            global_idx = class_indices[local_idx]
            replaced = False
            for _ in range(max_attempts):
                idx_a = int(rng.integers(len(ce)))
                idx_b = int(rng.integers(len(ce)))
                if idx_a == idx_b: continue
                t_new = float(rng.uniform(0.0, 1.0))
                new_emb = _slerp(ce[idx_a], ce[idx_b], t_new)
                if int(detector.predict([new_emb])[0]) == 1:
                    synthetic_embeddings[global_idx] = new_emb
                    replaced = True
                    break
            if not replaced:
                valid_mask[global_idx] = False

    return synthetic_embeddings[valid_mask], synthetic_labels[valid_mask]

# Setup
N = 10000
D = 512
K = 5
synthetic_embeddings = np.random.randn(N, D)
synthetic_labels = np.random.randint(0, K, N)
class_embs = {k: np.random.randn(100, D) for k in range(K)}
detectors = {}
for k in range(K):
    det = IsolationForest(contamination=0.1, random_state=42)
    det.fit(class_embs[k])
    detectors[k] = det

# Test unoptimized
syn_emb1 = synthetic_embeddings.copy()
syn_lab1 = synthetic_labels.copy()
start = time.time()
res1_e, res1_l = filter_unoptimized(syn_emb1, syn_lab1, detectors, None, class_embs)
t_unopt = time.time() - start

# Test optimized
syn_emb2 = synthetic_embeddings.copy()
syn_lab2 = synthetic_labels.copy()
start = time.time()
res2_e, res2_l = filter_optimized(syn_emb2, syn_lab2, detectors, None, class_embs)
t_opt = time.time() - start

print(f"Unoptimized: {t_unopt:.4f}s")
print(f"Optimized:   {t_opt:.4f}s")
print(f"Speedup:     {t_unopt/t_opt:.2f}x")
print(f"Same mask len? {len(res1_e)} vs {len(res2_e)}")
