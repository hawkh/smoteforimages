import torch
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import time

def compute_diversity_numpy(synthetic_images):
    batch_size = synthetic_images.shape[0]
    flattened = synthetic_images.view(batch_size, -1).cpu().numpy()
    distances = pairwise_distances(flattened, metric='euclidean')
    upper_triangle = distances[np.triu_indices_from(distances, k=1)]
    return float(np.mean(upper_triangle)), float(np.std(upper_triangle)), float(np.min(upper_triangle)), float(np.max(upper_triangle))

def compute_diversity_torch(synthetic_images):
    batch_size = synthetic_images.shape[0]
    flattened = synthetic_images.view(batch_size, -1)
    distances = torch.cdist(flattened, flattened, p=2)
    i, j = torch.triu_indices(batch_size, batch_size, offset=1, device=flattened.device)
    upper_triangle = distances[i, j]
    return float(torch.mean(upper_triangle)), float(torch.std(upper_triangle, unbiased=False)), float(torch.min(upper_triangle)), float(torch.max(upper_triangle))

# Create some dummy data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
images = torch.rand(100, 3, 224, 224, device=device)

# Warmup
compute_diversity_numpy(images)
compute_diversity_torch(images)

# Benchmark numpy
start = time.time()
res_np = compute_diversity_numpy(images)
time_np = time.time() - start

# Benchmark torch
start = time.time()
res_torch = compute_diversity_torch(images)
time_torch = time.time() - start

print(f"NumPy: {res_np}, Time: {time_np:.4f}s")
print(f"PyTorch: {res_torch}, Time: {time_torch:.4f}s")
print(f"Speedup: {time_np/time_torch:.2f}x")
