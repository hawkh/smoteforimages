"""Pydantic response models for the SMOTE synthesis API."""

from typing import Any, Optional
from pydantic import BaseModel


class ClassInfo(BaseModel):
    name: str
    count: int


class DatasetResponse(BaseModel):
    dataset_id: str
    name: str
    classes: list[ClassInfo]
    total_images: int


class DatasetDetailResponse(DatasetResponse):
    sample_images: dict[str, list[str]]  # class_name -> list of base64 thumbnails


class RunSummary(BaseModel):
    run_id: str
    dataset_id: str
    status: str  # idle | training | trained | generating | complete | error
    config: dict[str, Any]
    created_at: str


class ConfigureResponse(BaseModel):
    run_id: str
    config_summary: dict[str, Any]


class TrainStartResponse(BaseModel):
    status: str
    run_id: str
    ws_url: str


class TrainStatusResponse(BaseModel):
    run_id: str
    status: str
    phase: Optional[str] = None
    epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    is_complete: bool = False
    metrics: Optional[dict[str, float]] = None


class GenerateResponse(BaseModel):
    run_id: str
    n_generated: int
    class_breakdown: dict[str, int]
    output_dir: str


class ImageResult(BaseModel):
    url: str
    class_name: str
    filename: str
    is_synthetic: bool


class PaginatedImages(BaseModel):
    images: list[ImageResult]
    total: int
    page: int
    per_page: int


class QualityReport(BaseModel):
    run_id: str
    metrics: dict[str, float]
    diversity: Optional[dict[str, float]] = None
    per_class: Optional[dict[str, dict[str, float]]] = None


class PatentSection(BaseModel):
    id: str
    title: str
    content_md: str


class EquationInfo(BaseModel):
    id: str
    name: str
    latex: str
    description: str
