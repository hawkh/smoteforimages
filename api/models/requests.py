"""Pydantic request models for the SMOTE synthesis API."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class PipelineConfigRequest(BaseModel):
    """Configuration for a new pipeline run."""
    dataset_id: str
    image_size: int = Field(default=64, ge=32, le=512)
    embedding_dim: int = Field(default=512, ge=128, le=2048)
    architecture: Literal["resnet18", "resnet50"] = "resnet18"
    pretrained: bool = True

    # Decoder
    base_channels: int = Field(default=256, ge=64, le=512)
    use_self_attention: bool = True
    class_embed_dim: int = Field(default=64, ge=16, le=256)

    # SMOTE
    use_slerp: bool = True
    use_vmf: bool = False
    vmf_concentration_scale: float = Field(default=1.0, gt=0.0, le=10.0)
    k_neighbors: int = Field(default=5, ge=2, le=20)
    density_weighted_t: bool = False
    use_cluster_constraints: bool = False
    use_outlier_detection: bool = False
    track_ancestry: bool = False

    # Quality
    quality_metrics: list[str] = ["mse", "psnr", "ssim"]


class TrainRequest(BaseModel):
    """Start training for a configured pipeline run."""
    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=32, ge=4, le=256)
    learning_rate: float = Field(default=2e-4, gt=0, le=0.1)
    resume_from: Optional[str] = None


class GenerateRequest(BaseModel):
    """Generate synthetic images from a trained pipeline."""
    n_samples: Optional[int] = None
    balance_to: Optional[Literal["majority", "mean"]] = "majority"


class EvaluateRequest(BaseModel):
    """Run quality evaluation on generated images."""
    metrics: list[str] = ["mse", "psnr", "ssim"]
    n_eval_samples: int = Field(default=100, ge=10, le=5000)
    per_class: bool = False
