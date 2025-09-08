import torch
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def create_synthetic_dataset(n_samples: int = 200, image_size: int = 64) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Create a synthetic dataset for demonstration.

    Args:
        n_samples: Number of samples to generate
        image_size: Size of square images

    Returns:
        Tuple of (images, labels)
    """
    logger.info(f"Creating synthetic dataset with {n_samples} samples")

    # Create different classes with distinct patterns
    images = []
    labels = []

    n_per_class = n_samples // 3

    # Class 0: Horizontal stripes
    for i in range(n_per_class):
        img = torch.zeros(3, image_size, image_size)
        stripe_width = 8
        for y in range(0, image_size, stripe_width * 2):
            img[:, y:y+stripe_width, :] = torch.rand(3, 1, 1) * 0.8 + 0.2

        # Add noise
        img += torch.randn_like(img) * 0.1
        img = torch.clamp(img, 0, 1)

        images.append(img)
        labels.append(0)

    # Class 1: Vertical stripes
    for i in range(n_per_class):
        img = torch.zeros(3, image_size, image_size)
        stripe_width = 8
        for x in range(0, image_size, stripe_width * 2):
            img[:, :, x:x+stripe_width] = torch.rand(3, 1, 1) * 0.8 + 0.2

        # Add noise
        img += torch.randn_like(img) * 0.1
        img = torch.clamp(img, 0, 1)

        images.append(img)
        labels.append(1)

    # Class 2: Checkerboard pattern
    for i in range(n_samples - 2 * n_per_class):
        img = torch.zeros(3, image_size, image_size)
        square_size = 16

        for y in range(0, image_size, square_size):
            for x in range(0, image_size, square_size):
                if (y // square_size + x // square_size) % 2 == 0:
                    color = torch.rand(3) * 0.8 + 0.2
                    img[:, y:y+square_size, x:x+square_size] = color.view(3, 1, 1)

        # Add noise
        img += torch.randn_like(img) * 0.1
        img = torch.clamp(img, 0, 1)

        images.append(img)
        labels.append(2)

    images_tensor = torch.stack(images)
    labels_array = np.array(labels)

    logger.info(f"Dataset created with shape {images_tensor.shape}")
    logger.info(f"Class distribution: {np.bincount(labels_array)}")

    return images_tensor, labels_array
