import torch
import numpy as np
import time
from sklearn.metrics.pairwise import pairwise_distances

def compute_np(images):
    batch_size = images.shape[0]
    flattened = images.view(batch_size, -1).cpu().numpy()
    distances = pairwise_distances(flattened, metric='euclidean')
    upper_triangle = distances[np.triu_indices_from(distances, k=1)]
    return {
        'mean': float(np.mean(upper_triangle)),
        'std': float(np.std(upper_triangle)),
        'min': float(np.min(upper_triangle)),
        'max': float(np.max(upper_triangle)),
    }

def compute_pt(images):
    batch_size = images.shape[0]
    flattened = images.view(batch_size, -1)
    distances = torch.cdist(flattened, flattened, p=2.0)
    row_indices, col_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=flattened.device)
    upper_triangle = distances[row_indices, col_indices]
    return {
        'mean': float(torch.mean(upper_triangle)),
        'std': float(torch.std(upper_triangle, unbiased=False)),
        'min': float(torch.min(upper_triangle)),
        'max': float(torch.max(upper_triangle)),
    }

# Create dummy data
images = torch.rand(1000, 3, 64, 64)
if torch.cuda.is_available():
    images = images.cuda()

# Warmup
res_np = compute_np(images)
res_pt = compute_pt(images)

print("NumPy results:", res_np)
print("PyTorch results:", res_pt)

# Benchmark
import timeit
time_np = timeit.timeit(lambda: compute_np(images), number=10)
time_pt = timeit.timeit(lambda: compute_pt(images), number=10)

print(f"NumPy time (10 runs): {time_np:.4f}s")
print(f"PyTorch time (10 runs): {time_pt:.4f}s")
print(f"Speedup: {time_np / time_pt:.2f}x")
