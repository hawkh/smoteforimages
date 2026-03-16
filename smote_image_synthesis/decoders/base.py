"""
Base interface for image decoders.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Any, Union
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import logging
import json

logger = logging.getLogger(__name__)


class BaseDecoder(ABC):
    """
    Abstract base class for image decoders.
    
    Provides common functionality for model saving/loading, configuration management,
    batch decoding capabilities with memory management, and validation.
    """
    
    def __init__(
        self, 
        embedding_dim: int, 
        image_shape: Tuple[int, int, int],
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the decoder.
        
        Args:
            embedding_dim: Dimension of input embeddings
            image_shape: Output image shape (C, H, W)
            device: Device to run the model on (CPU/GPU)
            config: Additional configuration parameters
        """
        self.embedding_dim = embedding_dim
        self.image_shape = image_shape
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}
        
        # Model will be initialized by concrete implementations
        self.model: Optional[nn.Module] = None
        self._is_trained = False
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if not isinstance(self.embedding_dim, int) or self.embedding_dim <= 0:
            raise ValueError(f"Embedding dimension must be a positive integer, got {self.embedding_dim}")
        
        if len(self.image_shape) != 3:
            raise ValueError(f"Image shape must be (C, H, W), got {self.image_shape}")
        
        c, h, w = self.image_shape
        if not all(isinstance(dim, int) and dim > 0 for dim in [c, h, w]):
            raise ValueError(f"All image dimensions must be positive integers, got {self.image_shape}")
        
        if c not in [1, 3]:
            logger.warning(f"Unusual number of channels: {c}. Expected 1 (grayscale) or 3 (RGB)")
        
        # Check if embedding dimension is reasonable
        if self.embedding_dim > 4096:
            logger.warning(f"Large embedding dimension ({self.embedding_dim}) may cause memory issues")
    
    @abstractmethod
    def _build_model(self) -> nn.Module:
        """
        Build the decoder model architecture.
        
        Returns:
            PyTorch model
        """
        pass
        
    @abstractmethod
    def decode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Decode embeddings to images.
        
        Args:
            embeddings: Input embeddings [B, embedding_dim]
            
        Returns:
            Decoded images [B, C, H, W]
        """
        pass
    
    def decode_batch(
        self, 
        embedding_batch: List[torch.Tensor], 
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Decode a list of embedding batches with memory management.
        
        Args:
            embedding_batch: List of embedding tensors
            batch_size: Optional batch size for processing large lists
            
        Returns:
            Combined decoded images [total_embeddings, C, H, W]
        """
        if not embedding_batch:
            raise ValueError("Empty embedding batch provided")
        
        # If batch_size is specified and we have many batches, process in chunks
        if batch_size is not None and len(embedding_batch) > batch_size:
            return self._decode_large_batch(embedding_batch, batch_size)
        
        # Concatenate all tensors and decode
        try:
            combined_embeddings = torch.cat(embedding_batch, dim=0)
            return self.decode(combined_embeddings)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("Out of memory error, trying with smaller batches")
                # Fallback to processing individual batches
                return self._decode_individual_batches(embedding_batch)
            else:
                raise e
    
    def _decode_large_batch(self, embedding_batch: List[torch.Tensor], batch_size: int) -> torch.Tensor:
        """Process large batches in chunks to manage memory."""
        all_images = []
        
        for i in range(0, len(embedding_batch), batch_size):
            chunk = embedding_batch[i:i + batch_size]
            chunk_images = self.decode_batch(chunk)
            all_images.append(chunk_images)
        
        return torch.cat(all_images, dim=0)
    
    def _decode_individual_batches(self, embedding_batch: List[torch.Tensor]) -> torch.Tensor:
        """Decode each batch individually and combine results."""
        all_images = []
        
        for batch in embedding_batch:
            images = self.decode(batch)
            all_images.append(images)
        
        return torch.cat(all_images, dim=0)
        
    @abstractmethod
    def train_decoder(self, embeddings: torch.Tensor, images: torch.Tensor) -> None:
        """
        Train the decoder on embedding-image pairs.
        
        Args:
            embeddings: Training embeddings [B, embedding_dim]
            images: Target images [B, C, H, W]
        """
        pass
        
    def get_embedding_dim(self) -> int:
        """Get the expected embedding dimension."""
        return self.embedding_dim
        
    def get_image_shape(self) -> Tuple[int, int, int]:
        """Get the output image shape."""
        return self.image_shape
    
    def get_device(self) -> torch.device:
        """Get the device the model is running on."""
        return self.device
    
    def to_device(self, device: torch.device) -> 'BaseDecoder':
        """
        Move the decoder to a different device.
        
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
        Save the decoder model and configuration.
        
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
            'embedding_dim': self.embedding_dim,
            'image_shape': self.image_shape,
            'config': self.config,
            'is_trained': self._is_trained,
            'decoder_type': self.__class__.__name__
        }, model_path)
        
        # Save configuration as JSON
        config_path = path.with_suffix('.json')
        config_data = {
            'embedding_dim': self.embedding_dim,
            'image_shape': self.image_shape,
            'config': self.config,
            'decoder_type': self.__class__.__name__,
            'model_path': str(model_path)
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Configuration saved to {config_path}")
        
    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load the decoder model and configuration.
        
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
        if checkpoint['embedding_dim'] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {checkpoint['embedding_dim']}"
            )
        
        if checkpoint['image_shape'] != self.image_shape:
            raise ValueError(
                f"Image shape mismatch: expected {self.image_shape}, "
                f"got {checkpoint['image_shape']}"
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
    def load_from_config(cls, config_path: Union[str, Path]) -> 'BaseDecoder':
        """
        Load decoder from configuration file.
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            Loaded decoder instance
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # This method should be overridden by concrete implementations
        # to return the correct decoder type
        raise NotImplementedError("Subclasses must implement load_from_config")
        
    def validate_embeddings(self, embeddings: torch.Tensor) -> Tuple[bool, str]:
        """
        Validate input embeddings format.
        
        Args:
            embeddings: Input embedding tensor
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(embeddings, torch.Tensor):
            return False, f"Expected torch.Tensor, got {type(embeddings)}"
        
        if len(embeddings.shape) != 2:  # [B, embedding_dim]
            return False, f"Expected 2D tensor [B, embedding_dim], got shape {embeddings.shape}"
        
        if embeddings.shape[1] != self.embedding_dim:
            return False, f"Expected embedding dimension {self.embedding_dim}, got {embeddings.shape[1]}"
        
        if embeddings.shape[0] == 0:
            return False, "Batch size cannot be zero"
        
        # Check for valid values
        if torch.isnan(embeddings).any():
            return False, "Input contains NaN values"
        
        if torch.isinf(embeddings).any():
            return False, "Input contains infinite values"
        
        return True, "Valid embeddings"
    
    def validate_images(self, images: torch.Tensor) -> Tuple[bool, str]:
        """
        Validate target images format for training.
        
        Args:
            images: Target image tensor
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(images, torch.Tensor):
            return False, f"Expected torch.Tensor, got {type(images)}"
        
        if len(images.shape) != 4:  # [B, C, H, W]
            return False, f"Expected 4D tensor [B, C, H, W], got shape {images.shape}"
        
        expected_shape = (images.shape[0],) + self.image_shape
        if images.shape != expected_shape:
            return False, f"Expected shape {expected_shape}, got {images.shape}"
        
        if images.shape[0] == 0:
            return False, "Batch size cannot be zero"
        
        # Check for valid values
        if torch.isnan(images).any():
            return False, "Images contain NaN values"
        
        if torch.isinf(images).any():
            return False, "Images contain infinite values"
        
        return True, "Valid images"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the decoder model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'decoder_type': self.__class__.__name__,
            'embedding_dim': self.embedding_dim,
            'image_shape': self.image_shape,
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
        """String representation of the decoder."""
        return (f"{self.__class__.__name__}("
                f"embedding_dim={self.embedding_dim}, "
                f"image_shape={self.image_shape}, "
                f"device='{self.device}')")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()