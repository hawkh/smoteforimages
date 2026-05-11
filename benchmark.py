import torch
import time

def compute_repulsion_loss_original(embeddings, labels, margin=0.3):
    unique_labels = torch.unique(labels)
    total_repulsion = embeddings.new_zeros(1)
    n_pairs = 0
    for lbl in unique_labels:
        class_embs = embeddings[labels == lbl]
        if class_embs.size(0) < 2:
            continue
        diff = class_embs.unsqueeze(0) - class_embs.unsqueeze(1)
        dists = diff.norm(dim=-1)
        mask_upper = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
        pair_dists = dists[mask_upper]
        violations = torch.nn.functional.relu(margin - pair_dists)
        total_repulsion = total_repulsion + (violations ** 2).sum()
        n_pairs += len(pair_dists)
    if n_pairs == 0:
        return total_repulsion.squeeze()
    return (total_repulsion / n_pairs).squeeze()

def compute_repulsion_loss_optimized(embeddings, labels, margin=0.3):
    unique_labels = torch.unique(labels)
    total_repulsion = embeddings.new_zeros(1)
    n_pairs = 0
    for lbl in unique_labels:
        class_embs = embeddings[labels == lbl]
        if class_embs.size(0) < 2:
            continue
        dists = torch.cdist(class_embs.float(), class_embs.float(), p=2.0)
        mask_upper = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
        pair_dists = dists[mask_upper]
        violations = torch.nn.functional.relu(margin - pair_dists)
        total_repulsion = total_repulsion + (violations ** 2).sum()
        n_pairs += len(pair_dists)
    if n_pairs == 0:
        return total_repulsion.squeeze()
    return (total_repulsion / n_pairs).squeeze()

# Benchmarking
torch.manual_seed(42)
embeddings = torch.randn(4000, 512)
labels = torch.randint(0, 10, (4000,))

# Warmup
compute_repulsion_loss_original(embeddings, labels)
compute_repulsion_loss_optimized(embeddings, labels)

# Verify correctness
out_orig = compute_repulsion_loss_original(embeddings, labels)
out_opt = compute_repulsion_loss_optimized(embeddings, labels)
print(f"Diff: {torch.abs(out_orig - out_opt).item():.6f}")

t0 = time.time()
for _ in range(10):
    compute_repulsion_loss_original(embeddings, labels)
t1 = time.time()
print(f"Original: {t1 - t0:.4f}s")

t0 = time.time()
for _ in range(10):
    compute_repulsion_loss_optimized(embeddings, labels)
t1 = time.time()
print(f"Optimized: {t1 - t0:.4f}s")
