import time
import numpy as np
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE

# Generate dummy data
np.random.seed(42)
n_samples = 5000
n_classes = 10
embedding_dim = 512

embeddings = np.random.randn(n_samples, embedding_dim)
labels = np.random.randint(0, n_classes, n_samples)

# Synthetic data
n_synthetic = 2000
syn_embeddings = np.random.randn(n_synthetic, embedding_dim)
syn_labels = np.random.randint(0, n_classes, n_synthetic)

# Initialize SMOTE
smote = ConstrainedSMOTE(
    k_neighbors=5,
    max_distance_threshold=10.0,
    use_clustering=False
)
smote.embeddings = embeddings
smote.labels = labels

print("Starting benchmark...")
start_time = time.time()
filtered_emb, filtered_labels = smote._filter_by_distance(syn_embeddings, syn_labels)
end_time = time.time()

print(f"Time taken: {end_time - start_time:.4f} seconds")
print(f"Filtered shape: {filtered_emb.shape}")
