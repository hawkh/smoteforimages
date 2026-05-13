import time
import numpy as np
from scipy.spatial.distance import cdist

class MockSMOTE:
    def __init__(self, max_dist=100.0):
        self.max_distance_threshold = max_dist
        self.embeddings = None
        self.labels = None

    def _filter_by_distance(self, synthetic_embeddings, synthetic_labels):
        if self.max_distance_threshold is None:
            return synthetic_embeddings, synthetic_labels

        valid_indices = []
        unique_labels = np.unique(synthetic_labels)

        for label in unique_labels:
            syn_mask = synthetic_labels == label
            syn_idx = np.where(syn_mask)[0]
            if len(syn_idx) == 0:
                continue

            orig_mask = self.labels == label
            orig_embs = self.embeddings[orig_mask]

            if len(orig_embs) == 0:
                continue

            syn_embs_class = synthetic_embeddings[syn_idx]

            chunk_size = 2000
            for i in range(0, len(syn_embs_class), chunk_size):
                chunk_embs = syn_embs_class[i:i + chunk_size]
                chunk_idx = syn_idx[i:i + chunk_size]

                distances = cdist(chunk_embs, orig_embs, metric='euclidean')
                min_distances = np.min(distances, axis=1)

                valid_chunk_mask = min_distances <= self.max_distance_threshold
                valid_indices.extend(chunk_idx[valid_chunk_mask])

        valid_indices.sort()

        if valid_indices:
            return synthetic_embeddings[valid_indices], synthetic_labels[valid_indices]
        return np.array([]), np.array([])


np.random.seed(42)
n_classes = 10
M = 2000
N = 2000
D = 128

orig_embs = np.random.randn(n_classes * M, D).astype(np.float32)
orig_labels = np.repeat(np.arange(n_classes), M)

syn_embs = np.random.randn(n_classes * N, D).astype(np.float32)
syn_labels = np.repeat(np.arange(n_classes), N)

smote = MockSMOTE(max_dist=100.0)
smote.embeddings = orig_embs
smote.labels = orig_labels

t0 = time.time()
smote._filter_by_distance(syn_embs, syn_labels)
t1 = time.time()
print(f"New _filter_by_distance: {t1 - t0:.4f} seconds")
