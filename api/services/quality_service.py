"""Quality evaluation service wrapping QualityAssessor."""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def evaluate_quality(
    pipeline,
    synthetic_images: torch.Tensor,
    real_images: torch.Tensor,
    metrics: list[str] = None,
    n_samples: int = 100,
) -> dict:
    """Run quality evaluation on synthetic vs real images.

    Returns dict with 'metrics' and optionally 'diversity' keys.
    """
    if metrics is None:
        metrics = ["mse", "psnr", "ssim"]

    # Subsample if needed
    if synthetic_images.size(0) > n_samples:
        idx = torch.randperm(synthetic_images.size(0))[:n_samples]
        synthetic_images = synthetic_images[idx]
    if real_images.size(0) > n_samples:
        idx = torch.randperm(real_images.size(0))[:n_samples]
        real_images = real_images[idx]

    try:
        result = pipeline.evaluate_quality(synthetic_images, real_images)
        # result is a flat dict of metric_name -> value
        return {
            "metrics": {k: round(float(v), 6) for k, v in result.items()
                       if isinstance(v, (int, float))},
        }
    except Exception as e:
        logger.exception("Quality evaluation failed")
        return {"metrics": {}, "error": str(e)}
