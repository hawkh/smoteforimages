"""Dataset management — upload, list, delete, preview."""

import base64
import io
import shutil
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image

from api.config import DATASETS_DIR


def upload_dataset(
    files: list[tuple[str, bytes]],
    name: Optional[str] = None,
) -> dict:
    """Save uploaded files preserving class_name/image.ext structure.

    Args:
        files: List of (relative_path, file_bytes) pairs.
               relative_path should be like "cats/img1.jpg".
        name: Optional human-readable dataset name.

    Returns:
        Dataset metadata dict.
    """
    dataset_id = uuid.uuid4().hex[:12]
    dataset_dir = DATASETS_DIR / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for rel_path, data in files:
        # Sanitize: use only last two components (class/filename)
        parts = Path(rel_path).parts
        if len(parts) >= 2:
            class_name = parts[-2]
            filename = parts[-1]
        else:
            class_name = "uncategorized"
            filename = parts[-1] if parts else f"img_{uuid.uuid4().hex[:6]}.jpg"

        class_dir = dataset_dir / class_name
        class_dir.mkdir(exist_ok=True)
        (class_dir / filename).write_bytes(data)

    return get_dataset_info(dataset_id, name=name)


def list_datasets() -> list[dict]:
    """List all uploaded datasets with class counts."""
    datasets = []
    for d in sorted(DATASETS_DIR.iterdir()):
        if d.is_dir():
            datasets.append(get_dataset_info(d.name))
    return datasets


def get_dataset_info(dataset_id: str, name: Optional[str] = None) -> dict:
    """Get metadata for a dataset."""
    dataset_dir = DATASETS_DIR / dataset_id
    if not dataset_dir.exists():
        return None

    classes = []
    total = 0
    for class_dir in sorted(dataset_dir.iterdir()):
        if class_dir.is_dir():
            count = len([
                f for f in class_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
            ])
            classes.append({"name": class_dir.name, "count": count})
            total += count

    return {
        "dataset_id": dataset_id,
        "name": name or dataset_id,
        "classes": classes,
        "total_images": total,
    }


def get_dataset_detail(dataset_id: str, max_previews: int = 6) -> Optional[dict]:
    """Get dataset info with base64 thumbnail previews."""
    info = get_dataset_info(dataset_id)
    if info is None:
        return None

    dataset_dir = DATASETS_DIR / dataset_id
    sample_images = {}

    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        thumbs = []
        image_files = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
        ])
        for img_path in image_files[:max_previews]:
            try:
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((96, 96))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=75)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                thumbs.append(f"data:image/jpeg;base64,{b64}")
            except Exception:
                continue
        sample_images[class_dir.name] = thumbs

    return {**info, "sample_images": sample_images}


def delete_dataset(dataset_id: str) -> bool:
    """Delete a dataset and all its files."""
    dataset_dir = DATASETS_DIR / dataset_id
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
        return True
    return False


def get_dataset_path(dataset_id: str) -> Optional[Path]:
    """Get filesystem path for a dataset, or None if not found."""
    p = DATASETS_DIR / dataset_id
    return p if p.exists() else None
