"""
SMOTE-based Synthetic Image Generation System

A modular system for generating synthetic images using SMOTE applied to image embeddings.
"""

__version__ = "0.1.0"
__author__ = "SMOTE Image Synthesis Team"

from .data.models import (
    EmbeddingData,
    SyntheticSample,
    PipelineConfig,
    EncoderConfig,
    SMOTEConfig,
    DecoderConfig,
    QualityConfig
)
from .pipeline import SynthesisPipeline

__all__ = [
    "EmbeddingData",
    "SyntheticSample",
    "PipelineConfig",
    "EncoderConfig",
    "SMOTEConfig",
    "DecoderConfig",
    "QualityConfig",
    "SynthesisPipeline",
]