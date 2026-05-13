import time
import numpy as np
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE

np.random.seed(42)
n_classes = 10
M = 2000
N = 2000
D = 128

orig_embs = np.random.randn(n_classes * M, D).astype(np.float32)
orig_labels = np.repeat(np.arange(n_classes), M)

syn_embs = np.random.randn(n_classes * N, D).astype(np.float32)
syn_labels = np.repeat(np.arange(n_classes), N)

smote = ConstrainedSMOTE(max_distance_threshold=100.0)
smote.embeddings = orig_embs
smote.labels = orig_labels

t0 = time.time()
smote._filter_by_distance(syn_embs, syn_labels)
t1 = time.time()
print(f"Original _filter_by_distance: {t1 - t0:.4f} seconds")
