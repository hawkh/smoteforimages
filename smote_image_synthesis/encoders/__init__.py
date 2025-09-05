"""
Image encoder modules for converting images to embeddings.
"""

from .base import ImageEncoder
from .resnet_encoder import ResNetEncoder

__all__ = ["ImageEncoder", "ResNetEncoder"]