import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE

# Create dummy data
np.random.seed(42)
n_original = 10000
n_synthetic = 5000
dim = 512
n_classes = 10

original_embeddings = np.random.randn(n_original, dim)
original_labels = np.random.randint(0, n_classes, n_original)

synthetic_embeddings = np.random.randn(n_synthetic, dim)
synthetic_labels = np.random.randint(0, n_classes, n_synthetic)

# Naive implementation
def naive_filter(smote, syn_emb, syn_lab):
    valid_indices = []
    for i, (embedding, label) in enumerate(zip(syn_emb, syn_lab)):
        label_mask = smote.labels == label
        label_embeddings = smote.embeddings[label_mask]
        if len(label_embeddings) > 0:
            distances = np.linalg.norm(label_embeddings - embedding, axis=1)
            min_distance = np.min(distances)
            if min_distance <= smote.max_distance_threshold:
                valid_indices.append(i)
    if valid_indices:
        return syn_emb[valid_indices], syn_lab[valid_indices]
    return np.array([]), np.array([])

# Vectorized implementation
def vectorized_filter(smote, syn_emb, syn_lab):
    valid_mask = np.zeros(len(syn_emb), dtype=bool)
    unique_labels = np.unique(syn_lab)
    for label in unique_labels:
        syn_label_mask = syn_lab == label
        orig_label_mask = smote.labels == label
        orig_label_embeddings = smote.embeddings[orig_label_mask]
        if len(orig_label_embeddings) > 0:
            nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
            nn.fit(orig_label_embeddings)
            syn_label_embeddings = syn_emb[syn_label_mask]
            distances, _ = nn.kneighbors(syn_label_embeddings)
            valid_mask[syn_label_mask] = (distances.flatten() <= smote.max_distance_threshold)
    if np.any(valid_mask):
        return syn_emb[valid_mask], syn_lab[valid_mask]
    return np.array([]), np.array([])

smote = ConstrainedSMOTE(max_distance_threshold=20.0)
smote.embeddings = original_embeddings
smote.labels = original_labels

t0 = time.time()
r1_e, r1_l = naive_filter(smote, synthetic_embeddings, synthetic_labels)
t1 = time.time()
print(f"Naive: {t1-t0:.4f} seconds")

t0 = time.time()
r2_e, r2_l = vectorized_filter(smote, synthetic_embeddings, synthetic_labels)
t1 = time.time()
print(f"Vectorized: {t1-t0:.4f} seconds")

assert np.array_equal(r1_e, r2_e)
assert np.array_equal(r1_l, r2_l)
print("Results match!")
