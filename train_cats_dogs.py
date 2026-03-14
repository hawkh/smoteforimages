#!/usr/bin/env python3
"""
Train SMOTE image synthesis pipeline on cat and dog photos, then generate
synthetic images.

Data source: CIFAR-10 (auto-downloaded) — cats (class 3) and dogs (class 5).

Usage:
    python train_cats_dogs.py [--n-per-class N] [--epochs E] [--n-synthetic N]

Outputs (saved to ./synthetic_output/):
    real_samples.png      — grid of real cat/dog images used for training
    synthetic_samples.png — grid of SMOTE-generated synthetic images
    comparison.png        — real (top rows) vs synthetic (bottom rows)
    cats_dogs_pipeline_encoder.pth / _decoder.pth — saved weights
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, STL10

# Suppress torchvision's pretrained= deprecation warnings from existing code
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder
from smote_image_synthesis.decoders.autoencoder_decoder import AutoencoderDecoder
from smote_image_synthesis.decoders.dcgan_decoder import DCGANDecoder
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE
from smote_image_synthesis.quality.assessor import QualityAssessor
from smote_image_synthesis.pipeline import SynthesisPipeline

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('train_cats_dogs')

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 32   # native CIFAR-10 resolution — no blur from upscaling
# STL-10 class indices for cats/dogs
STL_CAT      = 3
STL_DOG      = 5
# CIFAR-10 fallback
CIFAR_CAT    = 3
CIFAR_DOG    = 5
CLASS_NAMES  = {0: 'cat', 1: 'dog'}
OUTPUT_DIR   = Path('synthetic_output')
DATA_DIR     = Path('data')


# ── Data loading ──────────────────────────────────────────────────────────────

def load_cats_and_dogs(n_per_class: int) -> tuple:
    """Load STL-10 cats & dogs (native 96×96, downscaled to IMAGE_SIZE).
    Falls back to CIFAR-10 if STL-10 download fails."""
    # No Resize needed — CIFAR-10 is natively IMAGE_SIZE×IMAGE_SIZE
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    logger.info("Loading CIFAR-10 (train+test splits, native 32×32)...")
    train_ds = CIFAR10(root=str(DATA_DIR), train=True,  download=True, transform=transform)
    test_ds  = CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=transform)
    cat_idx, dog_idx = CIFAR_CAT, CIFAR_DOG

    images_list, labels_list = [], []
    counts = {0: 0, 1: 0}
    for dataset in (train_ds, test_ds):
        for img, label in dataset:
            if label == cat_idx and counts[0] < n_per_class:
                images_list.append(img); labels_list.append(0); counts[0] += 1
            elif label == dog_idx and counts[1] < n_per_class:
                images_list.append(img); labels_list.append(1); counts[1] += 1
            if counts[0] >= n_per_class and counts[1] >= n_per_class:
                break

    images = torch.stack(images_list)
    labels = np.array(labels_list)
    logger.info(f"  {counts[0]} cats + {counts[1]} dogs = {len(images)} images @ {IMAGE_SIZE}×{IMAGE_SIZE}")
    return images, labels


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_image_grid(
    images: torch.Tensor,
    labels: np.ndarray,
    path: Path,
    title: str,
    cols: int = 5,
) -> None:
    """Save up to 25 images as a labelled grid PNG."""
    n    = min(len(images), 25)
    imgs = images[:n]
    labs = labels[:n] if labels is not None else None

    # Denormalise [-1, 1] → [0, 1]
    imgs = (imgs.clamp(-1, 1) + 1) / 2

    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2 + 0.4))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(imgs[i].cpu().permute(1, 2, 0).numpy().clip(0, 1))
            if labs is not None:
                ax.set_title(CLASS_NAMES.get(int(labs[i]), str(labs[i])), fontsize=8)
        ax.axis('off')

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved → {path}")


def save_comparison(
    real: torch.Tensor,
    real_labels: np.ndarray,
    synthetic: torch.Tensor,
    syn_labels: np.ndarray,
    path: Path,
) -> None:
    """Save a side-by-side comparison grid (real top, synthetic bottom)."""
    n       = min(10, len(real), len(synthetic))
    combined = torch.cat([real[:n], synthetic[:n].cpu()], dim=0)
    combined_labels = np.concatenate([real_labels[:n], syn_labels[:n]])
    save_image_grid(
        combined, combined_labels, path,
        title=f'Top row: real  |  Bottom row: SMOTE-synthetic  (n={n} each)',
        cols=n,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train SMOTE pipeline on cats & dogs')
    p.add_argument('--n-per-class', type=int, default=300,
                   help='Training images per class (default: 300)')
    p.add_argument('--epochs', type=int, default=150,
                   help='E2E training epochs (default: 150)')
    p.add_argument('--image-size', type=int, default=IMAGE_SIZE,
                   help=f'Image size (default: {IMAGE_SIZE})')
    p.add_argument('--n-synthetic', type=int, default=50,
                   help='Synthetic images to generate (default: 50)')
    p.add_argument('--embedding-dim', type=int, default=512,
                   help='Encoder embedding dimension (default: 512)')
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    images, labels = load_cats_and_dogs(args.n_per_class)
    save_image_grid(
        images, labels,
        OUTPUT_DIR / 'real_samples.png',
        title=f'Real CIFAR-10 Cats & Dogs (n={len(images)})',
    )

    # ── 2. Build components ───────────────────────────────────────────────────
    logger.info("Building pipeline components...")

    encoder = ResNetEncoder(
        architecture='resnet18',
        embedding_dim=args.embedding_dim,
        pretrained=True,
        device=device,
        freeze_backbone=False,  # Joint E2E training will update the whole encoder
    )

    decoder = DCGANDecoder(
        embedding_dim=args.embedding_dim,
        image_shape=(3, IMAGE_SIZE, IMAGE_SIZE),
        base_channels=512,   # 7M params → sharper details
        device=device,
    )

    smote = ConstrainedSMOTE(
        k_neighbors=5,
        sampling_strategy='auto',
        use_clustering=True,
        normalize_embeddings=True,
        random_state=42,
    )

    # Lightweight assessor — MSE/PSNR only, no InceptionV3 download needed
    quality_assessor = QualityAssessor(
        metrics=['mse', 'psnr'],
        compute_diversity=True,
    )

    pipeline = SynthesisPipeline(
        encoder=encoder,
        decoder=decoder,
        smote=smote,
        quality_assessor=quality_assessor,
    )

    # ── 3. Train ──────────────────────────────────────────────────────────────
    logger.info(f"Training decoder for {args.epochs} epochs "
                f"on {len(images)} images...")
    pipeline.fit(
        images=images.to(device),
        labels=labels,
        train_decoder=True,
        decoder_epochs=args.epochs,
    )
    logger.info("Training complete.")

    # ── 4. Generate synthetic images ──────────────────────────────────────────
    logger.info(f"Generating {args.n_synthetic} synthetic images via SMOTE...")
    synthetic_images, synthetic_labels = pipeline.generate_synthetic_images(
        n_samples=args.n_synthetic
    )

    if len(synthetic_images) == 0:
        logger.error("No synthetic images generated. Check SMOTE fit / class counts.")
        return

    n_cats = int((synthetic_labels == 0).sum())
    n_dogs = int((synthetic_labels == 1).sum())
    logger.info(f"  Generated {len(synthetic_images)} synthetic images "
                f"({n_cats} cats, {n_dogs} dogs)")

    save_image_grid(
        synthetic_images.cpu(), synthetic_labels,
        OUTPUT_DIR / 'synthetic_samples.png',
        title=f'SMOTE-Synthetic Cats & Dogs (n={len(synthetic_images)})',
    )

    # ── 5. Comparison grid ────────────────────────────────────────────────────
    save_comparison(
        images, labels,
        synthetic_images, synthetic_labels,
        OUTPUT_DIR / 'comparison.png',
    )

    # ── 6. Quick quality report ───────────────────────────────────────────────
    logger.info("Computing quality metrics...")
    n_eval = min(len(synthetic_images), len(images))
    try:
        metrics = pipeline.evaluate_quality(
            synthetic_images=synthetic_images[:n_eval].to(device),
            real_images=images[:n_eval].to(device),
        )
        logger.info("Quality metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k.upper():<8} {v:.4f}")
    except Exception as e:
        logger.warning(f"Quality evaluation skipped: {e}")

    # ── 7. Save pipeline weights ──────────────────────────────────────────────
    pipeline.save_pipeline(str(OUTPUT_DIR / 'cats_dogs_pipeline'))

    logger.info("\n" + "=" * 50)
    logger.info(f"Done!  Results in: {OUTPUT_DIR.resolve()}")
    logger.info("  real_samples.png       — training images")
    logger.info("  synthetic_samples.png  — SMOTE-generated images")
    logger.info("  comparison.png         — real vs synthetic grid")
    logger.info("  cats_dogs_pipeline_*   — saved model weights")


if __name__ == '__main__':
    main()
