"""Quality evaluation routes."""

import logging
from pathlib import Path

import torch
from PIL import Image
from fastapi import APIRouter, HTTPException
from torchvision import transforms

from api.config import DATASETS_DIR, OUTPUTS_DIR
from api.models.requests import EvaluateRequest
from api.models.responses import QualityReport
from api.services.pipeline_manager import get_manager
from api.services import quality_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/quality", tags=["quality"])


@router.post("/evaluate/{run_id}", response_model=QualityReport)
async def evaluate_quality(run_id: str, req: EvaluateRequest):
    """Run quality assessment on generated vs real images."""
    manager = get_manager()
    state = manager.get_run(run_id)
    if state is None:
        raise HTTPException(404, f"Run {run_id} not found")

    # Load real images
    dataset_path = DATASETS_DIR / state.dataset_id
    image_size = state.config.get("image_size", 64)
    real_images = _load_images_from_dir(dataset_path, image_size)

    # Load synthetic images
    syn_dir = OUTPUTS_DIR / run_id / "synthetic"
    if not syn_dir.exists():
        raise HTTPException(400, "No generated images found — run generate first")
    synthetic_images = _load_images_from_dir(syn_dir, image_size)

    if real_images is None or synthetic_images is None:
        raise HTTPException(500, "Failed to load images for evaluation")

    result = quality_service.evaluate_quality(
        state.pipeline,
        synthetic_images,
        real_images,
        metrics=req.metrics,
        n_samples=req.n_eval_samples,
    )

    state.quality_result = result

    return QualityReport(
        run_id=run_id,
        metrics=result.get("metrics", {}),
        diversity=result.get("diversity"),
        per_class=result.get("per_class"),
    )


@router.get("/{run_id}/report", response_model=QualityReport)
async def get_report(run_id: str):
    """Get cached quality report."""
    manager = get_manager()
    state = manager.get_run(run_id)
    if state is None:
        raise HTTPException(404, f"Run {run_id} not found")
    if state.quality_result is None:
        raise HTTPException(400, "No quality report — run evaluate first")

    return QualityReport(
        run_id=run_id,
        metrics=state.quality_result.get("metrics", {}),
        diversity=state.quality_result.get("diversity"),
        per_class=state.quality_result.get("per_class"),
    )


def _load_images_from_dir(root: Path, image_size: int):
    """Load all images from a directory tree into a tensor."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    tensors = []
    for img_path in root.rglob("*"):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            tensors.append(transform(img))
        except Exception:
            continue
    if not tensors:
        return None

    from api.config import DEVICE
    return torch.stack(tensors).to(DEVICE)
