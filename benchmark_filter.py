import time
import numpy as np
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE

# Generate dummy data
np.random.seed(42)
n_original = 10000
n_synthetic = 20000
dim = 512
n_classes = 10

orig_embeddings = np.random.randn(n_original, dim)
orig_labels = np.random.randint(0, n_classes, n_original)

synth_embeddings = np.random.randn(n_synthetic, dim)
synth_labels = np.random.randint(0, n_classes, n_synthetic)

smote = ConstrainedSMOTE(max_distance_threshold=10.0)
smote.embeddings = orig_embeddings
smote.labels = orig_labels

# Time the current implementation
start_time = time.time()
filtered_emb, filtered_labels = smote._filter_by_distance(synth_embeddings, synth_labels)
end_time = time.time()

print(f"Original _filter_by_distance took {end_time - start_time:.4f} seconds.")
print(f"Filtered count: {len(filtered_emb)}")
