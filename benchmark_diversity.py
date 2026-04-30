import torch
import time
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

# Mock data
batch_size = 1000
images = torch.rand(batch_size, 3, 64, 64, device='cuda' if torch.cuda.is_available() else 'cpu')

def original_method(images):
    flattened = images.view(batch_size, -1).cpu().numpy()
    distances = pairwise_distances(flattened, metric='euclidean')
    upper_triangle = distances[np.triu_indices_from(distances, k=1)]
    return np.mean(upper_triangle), np.std(upper_triangle), np.min(upper_triangle), np.max(upper_triangle)

def optimized_method(images):
    flattened = images.view(batch_size, -1)
    distances = torch.cdist(flattened, flattened, p=2)
    row_idx, col_idx = torch.triu_indices(batch_size, batch_size, offset=1, device=flattened.device)
    upper_triangle = distances[row_idx, col_idx]
    return torch.mean(upper_triangle).item(), torch.std(upper_triangle, unbiased=False).item(), torch.min(upper_triangle).item(), torch.max(upper_triangle).item()

# Warmup
original_method(images)
optimized_method(images)

start = time.time()
for _ in range(10):
    original_method(images)
orig_time = time.time() - start

start = time.time()
for _ in range(10):
    optimized_method(images)
opt_time = time.time() - start

print(f"Original time: {orig_time:.4f}s")
print(f"Optimized time: {opt_time:.4f}s")
print(f"Speedup: {orig_time/opt_time:.2f}x")
