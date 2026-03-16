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
from torchvision.datasets import CIFAR10

from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder
from smote_image_synthesis.decoders.autoencoder_decoder import AutoencoderDecoder
from smote_image_synthesis.decoders.dcgan_decoder import DCGANDecoder
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE
from smote_image_synthesis.quality.assessor import QualityAssessor
from smote_image_synthesis.pipeline import SynthesisPipeline

warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('train_cats_dogs')

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_IMAGE_SIZE = 32   # CIFAR-10 native resolution
CIFAR_CAT    = 3
CIFAR_DOG    = 5
CLASS_NAMES  = {0: 'cat', 1: 'dog'}
OUTPUT_DIR   = Path('synthetic_output')
DATA_DIR     = Path('data')


# ── Data loading ──────────────────────────────────────────────────────────────

def load_cats_and_dogs(n_per_class: int, image_size: int = DEFAULT_IMAGE_SIZE) -> tuple:
    """Load CIFAR-10 cats & dogs, optionally resizing to image_size."""
    resize_ops = []
    if image_size != 32:
        resize_ops = [transforms.Resize(image_size)]
    transform = transforms.Compose([
        *resize_ops,
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    logger.info(f"Loading CIFAR-10 (train+test splits, output {image_size}×{image_size})...")
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
    logger.info(f"  {counts[0]} cats + {counts[1]} dogs = {len(images)} images @ {image_size}×{image_size}")
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
    p.add_argument('--image-size', type=int, default=DEFAULT_IMAGE_SIZE,
                   help=f'Image size (default: {DEFAULT_IMAGE_SIZE}). Use 64 for higher res.')
    p.add_argument('--n-synthetic', type=int, default=50,
                   help='Synthetic images to generate (default: 50)')
    p.add_argument('--embedding-dim', type=int, default=512,
                   help='Encoder embedding dimension (default: 512)')
    p.add_argument('--decoder', choices=['dcgan', 'autoencoder'], default='dcgan',
                   help='Decoder architecture (default: dcgan)')
    p.add_argument('--architecture', choices=['resnet18', 'resnet50'], default='resnet18',
                   help='Encoder backbone (default: resnet18)')
    p.add_argument('--resume', type=str, default=None, metavar='PATH',
                   help='Resume from a checkpoint directory (e.g. synthetic_output/ckpt_epoch_50)')
    p.add_argument('--save-every', type=int, default=0,
                   help='Save pipeline checkpoint every N epochs (0 = only at end)')
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    image_size = args.image_size

    # ── 1. Load data ──────────────────────────────────────────────────────────
    images, labels = load_cats_and_dogs(args.n_per_class, image_size=image_size)
    save_image_grid(
        images, labels,
        OUTPUT_DIR / 'real_samples.png',
        title=f'Real CIFAR-10 Cats & Dogs (n={len(images)})',
    )

    # ── 2. Build components ───────────────────────────────────────────────────
    logger.info(f"Building pipeline: encoder={args.architecture}, decoder={args.decoder}, "
                f"image_size={image_size}×{image_size}, emb={args.embedding_dim}")

    encoder = ResNetEncoder(
        architecture=args.architecture,
        embedding_dim=args.embedding_dim,
        pretrained=True,
        device=device,
        freeze_backbone=False,
        normalize_output=True,   # L2-normalise → unit hypersphere for ideal SLERP
    )

    if args.decoder == 'dcgan':
        decoder = DCGANDecoder(
            embedding_dim=args.embedding_dim,
            image_shape=(3, image_size, image_size),
            base_channels=512,
            num_classes=2,            # class-conditional: cats (0) vs dogs (1)
            class_embed_dim=64,
            use_self_attention=True,  # SAGAN-style attention at 16×16
            device=device,
        )
    else:
        from smote_image_synthesis.decoders.autoencoder_decoder import AutoencoderDecoder
        decoder = AutoencoderDecoder(
            embedding_dim=args.embedding_dim,
            image_shape=(3, image_size, image_size),
            device=device,
        )

    smote = ConstrainedSMOTE(
        k_neighbors=5,
        sampling_strategy='auto',
        use_clustering=True,
        normalize_embeddings=False,  # Encoder already L2-normalises; skip StandardScaler
        use_slerp=True,              # Geodesic interpolation on embedding hypersphere
        random_state=42,
    )

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

    # ── 3. Resume from checkpoint if requested ────────────────────────────────
    start_epoch = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            logger.info(f"Resuming from checkpoint: {ckpt_path}")
            pipeline.load_pipeline(str(ckpt_path))
            # Infer epoch from directory name if it follows ckpt_epoch_N pattern
            try:
                start_epoch = int(ckpt_path.name.split('_')[-1])
                logger.info(f"  Resuming from epoch {start_epoch}")
            except ValueError:
                pass
        else:
            logger.warning(f"Checkpoint not found: {ckpt_path}. Starting from scratch.")

    remaining_epochs = max(0, args.epochs - start_epoch)
    if remaining_epochs == 0:
        logger.info("Training already complete (start_epoch >= epochs). Skipping.")
    else:
        # ── 4. Train ──────────────────────────────────────────────────────────
        if args.save_every > 0:
            # Train in segments, saving a checkpoint after each segment.
            # start_epoch + total_epochs are passed so the LR schedule and
            # GAN warmup phase are computed against the full training run.
            epoch_cursor = start_epoch
            while epoch_cursor < args.epochs:
                segment = min(args.save_every, args.epochs - epoch_cursor)
                logger.info(f"Training epochs {epoch_cursor}–{epoch_cursor + segment - 1} "
                            f"/ {args.epochs} total...")
                pipeline.fit(
                    images=images.to(device),
                    labels=labels,
                    train_decoder=True,
                    decoder_epochs=segment,
                    start_epoch=epoch_cursor,
                    total_epochs=args.epochs,
                )
                epoch_cursor += segment
                ckpt_dir = OUTPUT_DIR / f'ckpt_epoch_{epoch_cursor}'
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                pipeline.save_pipeline(str(ckpt_dir / 'cats_dogs_pipeline'))
                logger.info(f"  Checkpoint saved → {ckpt_dir}")
        else:
            logger.info(f"Training for {remaining_epochs} epochs on {len(images)} images...")
            pipeline.fit(
                images=images.to(device),
                labels=labels,
                train_decoder=True,
                decoder_epochs=remaining_epochs,
                start_epoch=start_epoch,
                total_epochs=args.epochs,
            )
        logger.info("Training complete.")

    # ── 5. Generate synthetic images ──────────────────────────────────────────
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

    # ── 6. Comparison grid ────────────────────────────────────────────────────
    save_comparison(
        images, labels,
        synthetic_images, synthetic_labels,
        OUTPUT_DIR / 'comparison.png',
    )

    # ── 7. Quick quality report ───────────────────────────────────────────────
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

    # ── 8. Save final pipeline weights ───────────────────────────────────────
    pipeline.save_pipeline(str(OUTPUT_DIR / 'cats_dogs_pipeline'))

    logger.info("\n" + "=" * 50)
    logger.info(f"Done!  Results in: {OUTPUT_DIR.resolve()}")
    logger.info("  real_samples.png       — training images")
    logger.info("  synthetic_samples.png  — SMOTE-generated images")
    logger.info("  comparison.png         — real vs synthetic grid")
    logger.info("  cats_dogs_pipeline_*   — saved model weights")


if __name__ == '__main__':
    main()
