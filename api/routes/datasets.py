"""Dataset management routes — upload, list, detail, delete."""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional

from api.models.responses import DatasetResponse, DatasetDetailResponse
from api.services import dataset_service

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    files: list[UploadFile] = File(...),
    paths: list[str] = Form(...),
    name: Optional[str] = Form(None),
):
    """Upload a dataset. Send files with their relative paths (class/filename)."""
    if len(files) != len(paths):
        raise HTTPException(400, "files and paths must have same length")

    file_pairs = []
    for f, p in zip(files, paths):
        data = await f.read()
        file_pairs.append((p, data))

    result = dataset_service.upload_dataset(file_pairs, name=name)
    return result


@router.get("", response_model=list[DatasetResponse])
async def list_datasets():
    """List all uploaded datasets."""
    return dataset_service.list_datasets()


@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
async def get_dataset(dataset_id: str):
    """Get dataset details with thumbnail previews."""
    result = dataset_service.get_dataset_detail(dataset_id)
    if result is None:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    return result


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: str):
    """Delete a dataset."""
    if not dataset_service.delete_dataset(dataset_id):
        raise HTTPException(404, f"Dataset {dataset_id} not found")
