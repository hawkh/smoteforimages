#!/usr/bin/env python3
"""
General-purpose SMOTE image synthesis pipeline.

Plug in ANY image dataset that follows the standard layout:

    data_root/
        class_a/  img1.jpg  img2.png ...
        class_b/  img1.jpg ...
        class_c/  ...

The pipeline will:
  1. Load images (auto-detects class labels from sub-folder names)
  2. Train ResNet18 encoder + DCGAN decoder jointly (end-to-end)
  3. Fit SLERP-SMOTE on the learned embedding space
  4. Generate synthetic images to balance every minority class
  5. Save outputs, quality metrics, and a grid comparison

Usage examples
--------------
  # Minimal — just point at your data folder:
  python run_pipeline.py --data-dir /path/to/dataset

  # With options:
  python run_pipeline.py \\
      --data-dir /path/to/dataset \\
      --image-size 64 \\
      --epochs 200 \\
      --n-per-class 500 \\
      --n-synthetic 200 \\
      --output-dir my_output \\
      --balance-to majority     # 'majority'|'mean'|N (exact count per class)

  # Resume from a checkpoint:
  python run_pipeline.py --data-dir /path/to/dataset --resume my_output/ckpt_epoch_100

Outputs (in --output-dir)
-------------------------
  real_samples.png        grid of real images per class
  synthetic_samples.png   grid of SMOTE-generated images
  comparison.png          real (top) vs synthetic (bottom)
  class_balance.png       before / after bar chart
  pipeline_encoder.pth    saved encoder weights
  pipeline_decoder.pth    saved decoder weights
  quality_metrics.json    MSE, PSNR, SSIM etc.
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder
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
logger = logging.getLogger('run_pipeline')

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


# ── Dataset loading ────────────────────────────────────────────────────────────

class FolderDataset(Dataset):
    """Minimal image dataset: data_root/<class>/<image>."""

    def __init__(self, root: Path, transform):
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        if not class_dirs:
            raise ValueError(
                f"No sub-directories found in {root}. "
                "Expected layout: data_root/<class>/<images>"
            )
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir.name] = idx
            self.idx_to_class[idx] = class_dir.name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((img_path, idx))

        if not self.samples:
            raise ValueError(f"No images found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        try:
            img = Image.open(path).convert('RGB')
        except (UnidentifiedImageError, OSError) as e:
            logger.warning(f"Skipping corrupt image {path}: {e}")
            # Return a black image as fallback
            img = Image.new('RGB', (224, 224), 0)
        return self.transform(img), label


def load_dataset(
    data_dir: Path,
    image_size: int,
    n_per_class: Optional[int],
) -> Tuple[torch.Tensor, np.ndarray, Dict[int, str]]:
    """Load images from a folder-based dataset.

    Args:
        data_dir: Root directory with one sub-folder per class.
        image_size: Resize target (square).
        n_per_class: Cap per class. None = use all images.

    Returns:
        (images [N,C,H,W], labels [N], idx_to_class mapping)
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    ds = FolderDataset(data_dir, transform)
    logger.info(f"Found {len(ds.class_to_idx)} classes: {list(ds.class_to_idx.keys())}")

    # Group by class, optionally cap
    class_images: Dict[int, List[torch.Tensor]] = {i: [] for i in ds.idx_to_class}
    for img, lbl in ds:
        if n_per_class is None or len(class_images[lbl]) < n_per_class:
            class_images[lbl].append(img)

    images_list, labels_list = [], []
    for cls_idx, imgs in class_images.items():
        for img in imgs:
            images_list.append(img)
            labels_list.append(cls_idx)
        logger.info(f"  Class {ds.idx_to_class[cls_idx]:>20s}: {len(imgs):>5d} images")

    images = torch.stack(images_list)
    labels = np.array(labels_list, dtype=np.int64)
    return images, labels, ds.idx_to_class


def compute_target_counts(
    labels: np.ndarray,
    balance_to: str,
) -> Dict[int, int]:
    """Compute target count per class for balanced synthesis.

    Args:
        labels: Current label array.
        balance_to: 'majority' | 'mean' | integer string.

    Returns:
        Dict mapping class_idx → number of synthetics needed.
    """
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))

    if balance_to == 'majority':
        target = int(np.max(counts))
    elif balance_to == 'mean':
        target = int(np.mean(counts))
    else:
        target = int(balance_to)

    need = {}
    for cls, cnt in class_counts.items():
        deficit = target - cnt
        if deficit > 0:
            need[cls] = deficit
    return need


# ── Visualisation ──────────────────────────────────────────────────────────────

def save_image_grid(
    images: torch.Tensor,
    labels: np.ndarray,
    idx_to_class: Dict[int, str],
    path: Path,
    title: str,
    cols: int = 6,
) -> None:
    """Save up to cols*5 images as a labelled grid PNG."""
    n = min(len(images), cols * 5)
    imgs = (images[:n].clamp(-1, 1) + 1) / 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2 + 0.4))
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(imgs[i].cpu().permute(1, 2, 0).numpy().clip(0, 1))
            ax.set_title(idx_to_class.get(int(labels[i]), str(labels[i])), fontsize=7)
        ax.axis('off')

    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved → {path}")


def save_class_balance_chart(
    labels_before: np.ndarray,
    labels_after: np.ndarray,
    idx_to_class: Dict[int, str],
    path: Path,
) -> None:
    """Bar chart: class counts before vs after augmentation."""
    classes = sorted(idx_to_class.keys())
    names = [idx_to_class[c] for c in classes]
    counts_before = [int((labels_before == c).sum()) for c in classes]
    counts_after  = [int((labels_after  == c).sum()) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.2), 4))
    ax.bar(x - width / 2, counts_before, width, label='Real', color='steelblue')
    ax.bar(x + width / 2, counts_after,  width, label='Real + Synthetic', color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Samples')
    ax.set_title('Class balance before / after SMOTE augmentation')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved → {path}")


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='General-purpose SMOTE image synthesis — works with any dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data-dir', required=True,
                   help='Root folder: data_dir/<class>/<images>')
    p.add_argument('--output-dir', default='pipeline_output',
                   help='Where to save outputs')
    p.add_argument('--image-size', type=int, default=64,
                   help='Resize images to this square size')
    p.add_argument('--n-per-class', type=int, default=None,
                   help='Max images per class to load (None = all)')
    p.add_argument('--epochs', type=int, default=150,
                   help='E2E training epochs')
    p.add_argument('--n-synthetic', type=int, default=None,
                   help='Total synthetic images to generate. None = auto-balance to majority class')
    p.add_argument('--balance-to', default='majority',
                   help="Balancing target: 'majority', 'mean', or an integer count")
    p.add_argument('--embedding-dim', type=int, default=512,
                   help='Encoder embedding dimension')
    p.add_argument('--architecture', choices=['resnet18', 'resnet50'], default='resnet18',
                   help='Encoder backbone')
    p.add_argument('--base-channels', type=int, default=256,
                   help='DCGAN base channels (256 for fast, 512 for quality)')
    p.add_argument('--batch-size', type=int, default=32,
                   help='Training batch size')
    p.add_argument('--resume', type=str, default=None, metavar='PATH',
                   help='Resume from checkpoint directory')
    p.add_argument('--save-every', type=int, default=0,
                   help='Save checkpoint every N epochs (0 = end only)')
    p.add_argument('--no-pretrained', action='store_true',
                   help='Do not use ImageNet pretrained weights')
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    images, labels, idx_to_class = load_dataset(
        data_dir, args.image_size, args.n_per_class
    )
    n_classes = len(idx_to_class)
    logger.info(f"Total: {len(images)} images across {n_classes} classes")

    save_image_grid(
        images, labels, idx_to_class,
        output_dir / 'real_samples.png',
        title=f'Real samples ({len(images)} total, {n_classes} classes)',
    )

    # ── 2. Build pipeline ─────────────────────────────────────────────────────
    logger.info(
        f"Building pipeline: {args.architecture}, emb={args.embedding_dim}, "
        f"image={args.image_size}×{args.image_size}, classes={n_classes}"
    )

    encoder = ResNetEncoder(
        architecture=args.architecture,
        embedding_dim=args.embedding_dim,
        pretrained=not args.no_pretrained,
        device=device,
        freeze_backbone=False,
        normalize_output=True,       # L2-normalise → unit sphere for SLERP
    )

    decoder = DCGANDecoder(
        embedding_dim=args.embedding_dim,
        image_shape=(3, args.image_size, args.image_size),
        base_channels=args.base_channels,
        num_classes=n_classes,       # class-conditional generation
        class_embed_dim=64,
        use_self_attention=True,     # SAGAN attention at 16×16
        device=device,
    )

    smote = ConstrainedSMOTE(
        k_neighbors=5,
        sampling_strategy='auto',
        use_clustering=True,
        normalize_embeddings=False,  # encoder already L2-normalises
        use_slerp=True,              # geodesic interpolation
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

    # ── 3. Resume from checkpoint ─────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            logger.info(f"Resuming from: {ckpt_path}")
            pipeline.load_pipeline(str(ckpt_path / 'pipeline'))
            try:
                start_epoch = int(ckpt_path.name.split('_')[-1])
                logger.info(f"  Continuing from epoch {start_epoch}")
            except ValueError:
                pass
        else:
            logger.warning(f"Checkpoint not found: {ckpt_path}. Starting fresh.")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    remaining = max(0, args.epochs - start_epoch)
    if remaining == 0:
        logger.info("Training already complete (start_epoch >= epochs). Skipping.")
    else:
        if args.save_every > 0:
            cursor = start_epoch
            while cursor < args.epochs:
                segment = min(args.save_every, args.epochs - cursor)
                logger.info(f"Training epochs {cursor}–{cursor+segment-1}/{args.epochs} ...")
                pipeline.fit(
                    images=images.to(device),
                    labels=labels,
                    train_decoder=True,
                    decoder_epochs=segment,
                    start_epoch=cursor,
                    total_epochs=args.epochs,
                )
                cursor += segment
                ckpt_dir = output_dir / f'ckpt_epoch_{cursor}'
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                pipeline.save_pipeline(str(ckpt_dir / 'pipeline'))
                logger.info(f"  Checkpoint → {ckpt_dir}")
        else:
            logger.info(f"Training {remaining} epochs on {len(images)} images ...")
            pipeline.fit(
                images=images.to(device),
                labels=labels,
                train_decoder=True,
                decoder_epochs=remaining,
                start_epoch=start_epoch,
                total_epochs=args.epochs,
            )
        logger.info("Training complete.")

    # ── 5. Determine how many synthetics to generate ──────────────────────────
    if args.n_synthetic is not None:
        n_synthetic = args.n_synthetic
    else:
        # Compute deficit to balance to the target
        need = compute_target_counts(labels, args.balance_to)
        n_synthetic = sum(need.values())
        if n_synthetic == 0:
            logger.info("Dataset is already balanced — generating 10 samples for demo.")
            n_synthetic = 10
        else:
            logger.info(
                f"Auto-balance mode ({args.balance_to}): "
                f"generating {n_synthetic} synthetics"
            )
            for cls, cnt in need.items():
                logger.info(f"  {idx_to_class[cls]}: +{cnt}")

    # ── 6. Generate synthetic images ──────────────────────────────────────────
    logger.info(f"Generating {n_synthetic} synthetic images ...")
    syn_images, syn_labels = pipeline.generate_synthetic_images(n_samples=n_synthetic)

    if len(syn_images) == 0:
        logger.error("No synthetic images generated. Check SMOTE fit / class counts.")
        return

    # Log per-class breakdown
    for cls_idx, cls_name in idx_to_class.items():
        n = int((syn_labels == cls_idx).sum())
        logger.info(f"  {cls_name}: {n} synthetic samples")

    # ── 7. Save visualisations ────────────────────────────────────────────────
    save_image_grid(
        syn_images.cpu(), syn_labels, idx_to_class,
        output_dir / 'synthetic_samples.png',
        title=f'SMOTE-Synthetic ({len(syn_images)} images)',
    )

    # Comparison: real (top) vs synthetic (bottom)
    n_cmp = min(12, len(images), len(syn_images))
    combined = torch.cat([images[:n_cmp], syn_images[:n_cmp].cpu()])
    combined_labels = np.concatenate([labels[:n_cmp], syn_labels[:n_cmp]])
    save_image_grid(
        combined, combined_labels, idx_to_class,
        output_dir / 'comparison.png',
        title=f'Top: real  |  Bottom: synthetic  (n={n_cmp} each)',
        cols=n_cmp,
    )

    # Class balance bar chart
    all_labels_after = np.concatenate([labels, syn_labels])
    save_class_balance_chart(
        labels, all_labels_after, idx_to_class,
        output_dir / 'class_balance.png',
    )

    # ── 8. Quality metrics ────────────────────────────────────────────────────
    logger.info("Computing quality metrics ...")
    n_eval = min(len(syn_images), len(images), 100)
    try:
        metrics = pipeline.evaluate_quality(
            synthetic_images=syn_images[:n_eval].to(device),
            real_images=images[:n_eval].to(device),
        )
        logger.info("Quality metrics:")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                logger.info(f"  {k.upper():<10} {v:.4f}")
        with open(output_dir / 'quality_metrics.json', 'w') as f:
            json.dump(
                {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                 for k, v in metrics.items()},
                f, indent=2,
            )
        logger.info(f"  Saved → {output_dir / 'quality_metrics.json'}")
    except Exception as e:
        logger.warning(f"Quality evaluation skipped: {e}")

    # ── 9. Save final weights ─────────────────────────────────────────────────
    pipeline.save_pipeline(str(output_dir / 'pipeline'))

    logger.info("\n" + "=" * 55)
    logger.info(f"Done!  Results in: {output_dir.resolve()}")
    logger.info("  real_samples.png       — input images grid")
    logger.info("  synthetic_samples.png  — SMOTE-generated grid")
    logger.info("  comparison.png         — real vs synthetic")
    logger.info("  class_balance.png      — before/after bar chart")
    logger.info("  quality_metrics.json   — MSE, PSNR, etc.")
    logger.info("  pipeline_*.pth         — saved weights")


if __name__ == '__main__':
    main()
