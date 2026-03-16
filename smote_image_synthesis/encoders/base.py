"""
Base interface for image encoders.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import logging
import json

logger = logging.getLogger(__name__)


class ImageEncoder(ABC):
    """
    Abstract base class for image encoders.
    
    Provides common functionality for model saving/loading, embedding dimension validation,
    and configuration management. Concrete implementations should inherit from this class
    and implement the abstract methods.
    """
    
    def __init__(
        self, 
        architecture: str,
        embedding_dim: int, 
        pretrained: bool = True,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the encoder.
        
        Args:
            architecture: Name of the encoder architecture (e.g., 'resnet50', 'efficientnet-b0')
            embedding_dim: Dimension of output embeddings
            pretrained: Whether to use pretrained weights
            device: Device to run the model on (CPU/GPU)
            config: Additional configuration parameters
        """
        self.architecture = architecture
        self.embedding_dim = embedding_dim
        self.pretrained = pretrained
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}
        
        # Model will be initialized by concrete implementations
        self.model: Optional[nn.Module] = None
        self._is_trained = False
        
        # Validate embedding dimension
        self._validate_embedding_dim()
        
    def _validate_embedding_dim(self) -> None:
        """Validate that embedding dimension is valid."""
        if not isinstance(self.embedding_dim, int) or self.embedding_dim <= 0:
            raise ValueError(f"Embedding dimension must be a positive integer, got {self.embedding_dim}")
        
        # Check if embedding dimension is reasonable (not too large)
        if self.embedding_dim > 4096:
            logger.warning(f"Large embedding dimension ({self.embedding_dim}) may cause memory issues")
    
    @abstractmethod
    def _build_model(self) -> nn.Module:
        """
        Build the encoder model architecture.
        
        Returns:
            PyTorch model
        """
        pass
    
    @abstractmethod
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of images to embeddings.
        
        Args:
            images: Batch of images [B, C, H, W]
            
        Returns:
            Embeddings [B, embedding_dim]
        """
        pass
        
    def encode_batch(
        self, 
        image_batch: List[torch.Tensor], 
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode a list of image batches with memory management.
        
        Args:
            image_batch: List of image tensors
            batch_size: Optional batch size for processing large lists
            
        Returns:
            Combined embeddings [total_images, embedding_dim]
        """
        if not image_batch:
            raise ValueError("Empty image batch provided")
        
        # If batch_size is specified and we have many batches, process in chunks
        if batch_size is not None and len(image_batch) > batch_size:
            return self._encode_large_batch(image_batch, batch_size)
        
        # Concatenate all tensors and encode
        try:
            combined_images = torch.cat(image_batch, dim=0)
            return self.encode(combined_images)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("Out of memory error, trying with smaller batches")
                # Fallback to processing individual batches
                return self._encode_individual_batches(image_batch)
            else:
                raise e
    
    def _encode_large_batch(self, image_batch: List[torch.Tensor], batch_size: int) -> torch.Tensor:
        """Process large batches in chunks to manage memory."""
        all_embeddings = []
        
        for i in range(0, len(image_batch), batch_size):
            chunk = image_batch[i:i + batch_size]
            chunk_embeddings = self.encode_batch(chunk)
            all_embeddings.append(chunk_embeddings)
        
        return torch.cat(all_embeddings, dim=0)
    
    def _encode_individual_batches(self, image_batch: List[torch.Tensor]) -> torch.Tensor:
        """Encode each batch individually and combine results."""
        all_embeddings = []
        
        for batch in image_batch:
            embeddings = self.encode(batch)
            all_embeddings.append(embeddings)
        
        return torch.cat(all_embeddings, dim=0)
        
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim
    
    def get_architecture(self) -> str:
        """Get the encoder architecture name."""
        return self.architecture
    
    def get_device(self) -> torch.device:
        """Get the device the model is running on."""
        return self.device
    
    def to_device(self, device: torch.device) -> 'ImageEncoder':
        """
        Move the encoder to a different device.
        
        Args:
            device: Target device
            
        Returns:
            Self for method chaining
        """
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self
    
    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save the encoder model and configuration.
        
        Args:
            path: Path to save the model (without extension)
        """
        if self.model is None:
            raise ValueError("No model to save. Model must be built first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        model_path = path.with_suffix('.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'architecture': self.architecture,
            'embedding_dim': self.embedding_dim,
            'pretrained': self.pretrained,
            'config': self.config,
            'is_trained': self._is_trained
        }, model_path)
        
        # Save configuration as JSON
        config_path = path.with_suffix('.json')
        config_data = {
            'architecture': self.architecture,
            'embedding_dim': self.embedding_dim,
            'pretrained': self.pretrained,
            'config': self.config,
            'model_path': str(model_path)
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Configuration saved to {config_path}")
        
    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load the encoder model and configuration.
        
        Args:
            path: Path to the saved model (without extension)
        """
        path = Path(path)
        
        # Load model
        model_path = path.with_suffix('.pth')
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # Validate compatibility
        if checkpoint['architecture'] != self.architecture:
            raise ValueError(
                f"Architecture mismatch: expected {self.architecture}, "
                f"got {checkpoint['architecture']}"
            )
        
        if checkpoint['embedding_dim'] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {checkpoint['embedding_dim']}"
            )
        
        # Build model if not already built
        if self.model is None:
            self.model = self._build_model()
            self.model = self.model.to(self.device)
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._is_trained = checkpoint.get('is_trained', False)
        
        # Update configuration
        self.config.update(checkpoint.get('config', {}))
        
        logger.info(f"Model loaded from {model_path}")
    
    @classmethod
    def load_from_config(cls, config_path: Union[str, Path]) -> 'ImageEncoder':
        """
        Load encoder from configuration file.
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            Loaded encoder instance
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # This method should be overridden by concrete implementations
        # to return the correct encoder type
        raise NotImplementedError("Subclasses must implement load_from_config")
        
    def validate_input(self, images: torch.Tensor) -> Tuple[bool, str]:
        """
        Validate input tensor format.
        
        Args:
            images: Input image tensor
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(images, torch.Tensor):
            return False, f"Expected torch.Tensor, got {type(images)}"
        
        if len(images.shape) != 4:  # [B, C, H, W]
            return False, f"Expected 4D tensor [B, C, H, W], got shape {images.shape}"
        
        if images.shape[1] not in [1, 3]:  # Grayscale or RGB
            return False, f"Expected 1 or 3 channels, got {images.shape[1]}"
        
        if images.shape[0] == 0:
            return False, "Batch size cannot be zero"
        
        if images.shape[2] == 0 or images.shape[3] == 0:
            return False, f"Invalid image dimensions: {images.shape[2]}x{images.shape[3]}"
        
        # Check for valid value range (assuming normalized images)
        if torch.isnan(images).any():
            return False, "Input contains NaN values"
        
        if torch.isinf(images).any():
            return False, "Input contains infinite values"
        
        return True, "Valid input"
    
    def validate_input_format(self, image_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Validate if an image file format is supported.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            return False, f"File not found: {image_path}"
        
        # Check file extension
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        if image_path.suffix.lower() not in supported_extensions:
            return False, f"Unsupported file format: {image_path.suffix}"
        
        return True, "Valid image format"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the encoder model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'architecture': self.architecture,
            'embedding_dim': self.embedding_dim,
            'pretrained': self.pretrained,
            'device': str(self.device),
            'is_trained': self._is_trained,
            'config': self.config
        }
        
        if self.model is not None:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            info.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
            })
        
        return info
    
    def set_training_mode(self, training: bool = True) -> None:
        """
        Set the model to training or evaluation mode.
        
        Args:
            training: Whether to set training mode
        """
        if self.model is not None:
            self.model.train(training)
    
    def eval(self) -> None:
        """Set the model to evaluation mode."""
        self.set_training_mode(False)
    
    def train(self) -> None:
        """Set the model to training mode."""
        self.set_training_mode(True)
    
    def __repr__(self) -> str:
        """String representation of the encoder."""
        return (f"{self.__class__.__name__}("
                f"architecture='{self.architecture}', "
                f"embedding_dim={self.embedding_dim}, "
                f"pretrained={self.pretrained}, "
                f"device='{self.device}')")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()