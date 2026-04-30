import torch
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

# Mock data
batch_size = 100
images = torch.rand(batch_size, 3, 64, 64)

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

o_mean, o_std, o_min, o_max = original_method(images)
op_mean, op_std, op_min, op_max = optimized_method(images)

print(f"Mean: {o_mean:.6f} vs {op_mean:.6f} (diff: {abs(o_mean - op_mean):.6f})")
print(f"Std: {o_std:.6f} vs {op_std:.6f} (diff: {abs(o_std - op_std):.6f})")
print(f"Min: {o_min:.6f} vs {op_min:.6f} (diff: {abs(o_min - op_min):.6f})")
print(f"Max: {o_max:.6f} vs {op_max:.6f} (diff: {abs(o_max - op_max):.6f})")
