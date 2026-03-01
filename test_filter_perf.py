import time
import numpy as np
from sklearn.neighbors import NearestNeighbors

def _filter_by_distance_old(max_distance_threshold, embeddings, labels, synthetic_embeddings, synthetic_labels):
    valid_indices = []
    for i, (embedding, label) in enumerate(zip(synthetic_embeddings, synthetic_labels)):
        # Find nearest neighbors in original embeddings of same class
        label_mask = labels == label
        label_embeddings = embeddings[label_mask]

        if len(label_embeddings) > 0:
            distances = np.linalg.norm(label_embeddings - embedding, axis=1)
            min_distance = np.min(distances)

            if min_distance <= max_distance_threshold:
                valid_indices.append(i)

    if valid_indices:
        return synthetic_embeddings[valid_indices], synthetic_labels[valid_indices]
    else:
        return np.array([]), np.array([])

def _filter_by_distance_new(max_distance_threshold, embeddings, labels, synthetic_embeddings, synthetic_labels):
    valid_indices = []
    unique_labels = np.unique(synthetic_labels)

    for label in unique_labels:
        orig_mask = labels == label
        orig_label_embeddings = embeddings[orig_mask]

        if len(orig_label_embeddings) == 0:
            continue

        syn_mask = synthetic_labels == label
        syn_indices = np.where(syn_mask)[0]
        syn_label_embeddings = synthetic_embeddings[syn_mask]

        nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(orig_label_embeddings)
        distances, _ = nn.kneighbors(syn_label_embeddings)

        valid_mask = distances.flatten() <= max_distance_threshold
        valid_indices.extend(syn_indices[valid_mask])

    if valid_indices:
        valid_indices = sorted(valid_indices)
        return synthetic_embeddings[valid_indices], synthetic_labels[valid_indices]
    else:
        return np.array([]), np.array([])

# Generate dummy data
np.random.seed(42)
n_samples = 5000
n_classes = 10
embedding_dim = 512

embeddings = np.random.randn(n_samples, embedding_dim)
labels = np.random.randint(0, n_classes, n_samples)

n_synthetic = 2000
syn_embeddings = np.random.randn(n_synthetic, embedding_dim)
syn_labels = np.random.randint(0, n_classes, n_synthetic)
max_dist = 100.0

print("Testing OLD...")
start = time.time()
old_emb, old_lbl = _filter_by_distance_old(max_dist, embeddings, labels, syn_embeddings, syn_labels)
print(f"Old took: {time.time() - start:.4f}s, found {len(old_emb)}")

print("Testing NEW...")
start = time.time()
new_emb, new_lbl = _filter_by_distance_new(max_dist, embeddings, labels, syn_embeddings, syn_labels)
print(f"New took: {time.time() - start:.4f}s, found {len(new_emb)}")

assert np.array_equal(old_emb, new_emb)
assert np.array_equal(old_lbl, new_lbl)
print("Results match!")
