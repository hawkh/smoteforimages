## 2024-05-05 - Avoid unsqueeze for PyTorch pairwise distances
**Learning:** Computing pairwise L2 distances using dimension expansion (`unsqueeze(0) - unsqueeze(1)`) allocates an intermediate tensor of size O(K² * D), causing severe memory spikes and poor performance on both CPU and GPU for large tensors.
**Action:** Always use native optimized distance functions like `torch.cdist(x, x, p=2.0)` instead of explicit broadcasting for pairwise operations to achieve O(K²) memory and significant speedups.
