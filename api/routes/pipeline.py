"""Pipeline routes — configure, train, generate, status."""

import asyncio
import io
import zipfile
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from torchvision import transforms

from api.config import DATASETS_DIR, OUTPUTS_DIR
from api.models.requests import PipelineConfigRequest, TrainRequest, GenerateRequest
from api.models.responses import (
    ConfigureResponse, TrainStartResponse, TrainStatusResponse,
    GenerateResponse, PaginatedImages, ImageResult, RunSummary,
)
from api.services.pipeline_manager import get_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router.post("/configure", response_model=ConfigureResponse)
async def configure_pipeline(req: PipelineConfigRequest):
    """Create a pipeline run with the given configuration."""
    manager = get_manager()
    try:
        state = manager.create_run(req.dataset_id, req.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Failed to configure pipeline")
        raise HTTPException(500, f"Pipeline configuration failed: {e}")

    return ConfigureResponse(
        run_id=state.run_id,
        config_summary=state.config,
    )


@router.post("/train/{run_id}", response_model=TrainStartResponse, status_code=202)
async def start_training(run_id: str, req: TrainRequest):
    """Start training in a background thread."""
    manager = get_manager()
    state = manager.get_run(run_id)
    if state is None:
        raise HTTPException(404, f"Run {run_id} not found")
    if state.status == "training":
        raise HTTPException(409, "Training already in progress")

    # Load dataset images
    dataset_path = DATASETS_DIR / state.dataset_id
    image_size = state.config.get("image_size", 64)
    images, labels = _load_dataset_tensors(dataset_path, image_size)

    if images is None:
        raise HTTPException(400, "Failed to load dataset images")

    state.status = "training"
    state.history.clear()

    # Launch training in background
    from api.services.training_runner import TrainingRunner

    def on_epoch(msg: dict):
        """Callback from training thread — push to async queue."""
        try:
            state.progress_queue.put_nowait(msg)
        except asyncio.QueueFull:
            pass  # Drop if queue is full
        state.history.append(msg)
        if msg["type"] in ("complete", "error"):
            state.status = "trained" if msg["type"] == "complete" else "error"
            if msg["type"] == "error":
                state.error = msg["data"].get("message")

    runner = TrainingRunner(state.pipeline, on_epoch)

    # Run in thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        runner.run,
        images, labels,
        req.epochs, req.batch_size, req.learning_rate,
    )

    return TrainStartResponse(
        status="started",
        run_id=run_id,
        ws_url=f"/ws/training/{run_id}",
    )


@router.get("/status/{run_id}", response_model=TrainStatusResponse)
async def get_status(run_id: str):
    """Get training status (poll fallback for WebSocket)."""
    manager = get_manager()
    state = manager.get_run(run_id)
    if state is None:
        raise HTTPException(404, f"Run {run_id} not found")

    # Find last epoch event
    last_epoch = None
    for msg in reversed(state.history):
        if msg.get("type") == "epoch":
            last_epoch = msg["data"]
            break

    return TrainStatusResponse(
        run_id=run_id,
        status=state.status,
        phase=last_epoch.get("phase") if last_epoch else None,
        epoch=last_epoch.get("epoch") if last_epoch else None,
        total_epochs=last_epoch.get("total_epochs") if last_epoch else None,
        is_complete=state.status in ("trained", "complete"),
        metrics=last_epoch if last_epoch else None,
    )


@router.post("/stop/{run_id}")
async def stop_training(run_id: str):
    """Request training stop."""
    manager = get_manager()
    state = manager.get_run(run_id)
    if state is None:
        raise HTTPException(404, f"Run {run_id} not found")
    # The runner checks _stop_requested each epoch
    state.status = "error"
    return {"status": "stopping", "run_id": run_id}


@router.get("/runs", response_model=list[RunSummary])
async def list_runs():
    """List all pipeline runs."""
    return get_manager().list_runs()


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Get full run details."""
    manager = get_manager()
    state = manager.get_run(run_id)
    if state is None:
        raise HTTPException(404, f"Run {run_id} not found")
    return {
        "run_id": state.run_id,
        "dataset_id": state.dataset_id,
        "status": state.status,
        "config": state.config,
        "error": state.error,
        "generation_result": state.generation_result,
        "quality_result": state.quality_result,
        "history_length": len(state.history),
    }


@router.post("/generate/{run_id}", response_model=GenerateResponse)
async def generate_images(run_id: str, req: GenerateRequest):
    """Generate synthetic images from a trained pipeline."""
    manager = get_manager()
    state = manager.get_run(run_id)
    if state is None:
        raise HTTPException(404, f"Run {run_id} not found")
    if state.status not in ("trained", "complete"):
        raise HTTPException(400, f"Pipeline not trained (status={state.status})")

    state.status = "generating"

    try:
        # Fit SMOTE on training embeddings first
        dataset_path = DATASETS_DIR / state.dataset_id
        image_size = state.config.get("image_size", 64)
        images, labels = _load_dataset_tensors(dataset_path, image_size)

        state.pipeline.fit(
            images, labels.numpy(),
            train_decoder=False,
            decoder_epochs=0,
        )

        # Generate
        syn_images, syn_labels = state.pipeline.generate_synthetic_images(
            n_samples=req.n_samples
        )

        # Save as PNGs
        output_dir = OUTPUTS_DIR / run_id / "synthetic"
        class_names = state.config.get("class_names", [])
        class_breakdown = {}

        for i in range(syn_images.size(0)):
            label_idx = int(syn_labels[i])
            class_name = class_names[label_idx] if label_idx < len(class_names) else f"class_{label_idx}"

            class_dir = output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Denormalize from [-1,1] to [0,255]
            img_tensor = syn_images[i].clamp(-1, 1)
            img_array = ((img_tensor + 1) / 2 * 255).byte().cpu().permute(1, 2, 0).numpy()
            img = Image.fromarray(img_array)
            img.save(class_dir / f"syn_{i:04d}.png")

            class_breakdown[class_name] = class_breakdown.get(class_name, 0) + 1

        state.status = "complete"
        state.generation_result = {
            "n_generated": syn_images.size(0),
            "class_breakdown": class_breakdown,
            "output_dir": str(output_dir),
        }

        return GenerateResponse(
            run_id=run_id,
            n_generated=syn_images.size(0),
            class_breakdown=class_breakdown,
            output_dir=str(output_dir),
        )

    except Exception as e:
        state.status = "error"
        state.error = str(e)
        logger.exception("Generation failed")
        raise HTTPException(500, f"Generation failed: {e}")


@router.get("/generate/{run_id}/results", response_model=PaginatedImages)
async def get_results(
    run_id: str,
    page: int = 1,
    per_page: int = 24,
    class_filter: Optional[str] = None,
):
    """Get paginated generated image URLs."""
    output_dir = OUTPUTS_DIR / run_id / "synthetic"
    if not output_dir.exists():
        raise HTTPException(404, "No generated images found")

    all_images = []
    for class_dir in sorted(output_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_filter and class_dir.name != class_filter:
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                all_images.append(ImageResult(
                    url=f"/static/outputs/{run_id}/synthetic/{class_dir.name}/{img_path.name}",
                    class_name=class_dir.name,
                    filename=img_path.name,
                    is_synthetic=True,
                ))

    total = len(all_images)
    start = (page - 1) * per_page
    end = start + per_page

    return PaginatedImages(
        images=all_images[start:end],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/generate/{run_id}/download")
async def download_results(run_id: str):
    """Download all generated images as a ZIP."""
    output_dir = OUTPUTS_DIR / run_id / "synthetic"
    if not output_dir.exists():
        raise HTTPException(404, "No generated images found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_path in output_dir.rglob("*.png"):
            arcname = str(img_path.relative_to(output_dir))
            zf.write(img_path, arcname)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=synthetic_{run_id}.zip"},
    )


def _load_dataset_tensors(dataset_path: Path, image_size: int):
    """Load all images from a class-folder dataset into tensors."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    images_list = []
    labels_list = []
    classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])

    for label_idx, class_name in enumerate(classes):
        class_dir = dataset_path / class_name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img)
                images_list.append(tensor)
                labels_list.append(label_idx)
            except Exception:
                continue

    if not images_list:
        return None, None

    from api.config import DEVICE
    images = torch.stack(images_list).to(DEVICE)
    labels = torch.tensor(labels_list, dtype=torch.long)
    return images, labels
