import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def save_comparison_plot(
    output_path: Path,
    real_images: torch.Tensor,
    real_labels: np.ndarray,
    synthetic_images: torch.Tensor,
    synthetic_labels: np.ndarray,
    n_display: int = 8
):
    """
    Saves a comparison plot of real and synthetic images.

    Args:
        output_path: Path to save the output image.
        real_images: Tensor of real images.
        real_labels: Numpy array of real labels.
        synthetic_images: Tensor of synthetic images.
        synthetic_labels: Numpy array of synthetic labels.
        n_display: Number of images to display.
    """
    n_display = min(n_display, len(real_images), len(synthetic_images))
    if n_display == 0:
        logger.warning("No images to display in comparison plot.")
        return

    fig, axes = plt.subplots(2, n_display, figsize=(2 * n_display, 4))

    for i in range(n_display):
        # Real images (top row)
        real_img = real_images[i].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(real_img)
        axes[0, i].set_title(f'Real (Class {real_labels[i]})')
        axes[0, i].axis('off')

        # Synthetic images (bottom row)
        synth_img = synthetic_images[i].permute(1, 2, 0).cpu().detach().numpy()
        if synth_img.min() < 0:  # Normalize if needed
            synth_img = (synth_img - synth_img.min()) / (synth_img.max() - synth_img.min())
        synth_img = np.clip(synth_img, 0, 1)

        axes[1, i].imshow(synth_img)
        axes[1, i].set_title(f'Synthetic (Class {synthetic_labels[i]})')
        axes[1, i].axis('off')

    plt.suptitle('Real vs Synthetic Images Comparison')
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Sample comparison saved to {output_path}")
