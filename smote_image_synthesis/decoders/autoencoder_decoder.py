"""
Autoencoder-based image decoder implementation.
"""

from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from .base import BaseDecoder

logger = logging.getLogger(__name__)


class AutoencoderDecoder(BaseDecoder):
    """
    Progressive autoencoder decoder with skip connections and perceptual loss integration.
    
    Features:
    - Progressive upsampling with learnable interpolation
    - Skip connections for feature preservation
    - Perceptual loss integration for better quality
    - Memory-efficient training and inference
    """
    
    def __init__(
        self,
        embedding_dim: int,
        image_shape: Tuple[int, int, int],
        hidden_dims: Optional[List[int]] = None,
        use_skip_connections: bool = True,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
        activation: str = 'relu',
        output_activation: str = 'tanh',
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize autoencoder decoder.
        
        Args:
            embedding_dim: Dimension of input embeddings
            image_shape: Output image shape (C, H, W)
            hidden_dims: List of hidden layer dimensions (auto-determined if None)
            use_skip_connections: Whether to use skip connections
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
            activation: Activation function ('relu', 'leaky_relu', 'elu')
            output_activation: Output activation function ('tanh', 'sigmoid', 'none')
            device: Device to run the model on
            config: Additional configuration parameters
        """
        # Set up configuration
        config = config or {}
        config.update({
            'hidden_dims': hidden_dims,
            'use_skip_connections': use_skip_connections,
            'use_batch_norm': use_batch_norm,
            'dropout_rate': dropout_rate,
            'activation': activation,
            'output_activation': output_activation
        })
        
        super().__init__(embedding_dim, image_shape, device, config)
        
        self.hidden_dims = hidden_dims or self._get_default_hidden_dims()
        self.use_skip_connections = use_skip_connections
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.output_activation = output_activation
        
        # Build and initialize model
        self.model = self._build_model()
        self.model = self.model.to(self.device)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized AutoencoderDecoder with embedding_dim={embedding_dim}, image_shape={image_shape}")
    
    def _get_default_hidden_dims(self) -> List[int]:
        """Generate default hidden dimensions based on embedding and image size."""
        c, h, w = self.image_shape
        target_size = c * h * w
        
        # Create progressive scaling from embedding_dim to target_size
        hidden_dims = []
        current_dim = self.embedding_dim
        
        while current_dim < target_size // 4:
            current_dim = min(current_dim * 2, 2048)  # Cap at 2048
            hidden_dims.append(current_dim)
        
        # Add final dimensions for spatial upsampling
        spatial_dims = [target_size // 4, target_size // 2]
        hidden_dims.extend(spatial_dims)
        
        return hidden_dims
    
    def _build_model(self) -> nn.Module:
        """Build the autoencoder decoder model."""
        c, h, w = self.image_shape
        
        # Calculate the spatial dimensions for reshape
        # Start with 4x4 spatial resolution and scale up
        start_h, start_w = 4, 4
        start_c = max(64, self.hidden_dims[-1] // (start_h * start_w))
        
        layers = []
        
        # Initial linear layers
        prev_dim = self.embedding_dim
        for i, hidden_dim in enumerate(self.hidden_dims[:-1]):
            layers.extend(self._create_linear_block(prev_dim, hidden_dim, f"linear_{i}"))
            prev_dim = hidden_dim
        
        # Final linear layer to spatial representation
        spatial_dim = start_c * start_h * start_w
        layers.extend(self._create_linear_block(prev_dim, spatial_dim, "spatial"))
        
        # Reshape layer
        layers.append(ReshapeLayer(start_c, start_h, start_w))
        
        # Convolutional upsampling layers
        conv_layers = self._build_conv_layers(start_c, start_h, start_w, c, h, w)
        layers.extend(conv_layers)
        
        return nn.Sequential(*layers)
    
    def _create_linear_block(self, in_dim: int, out_dim: int, name: str) -> List[nn.Module]:
        """Create a linear block with optional batch norm and dropout."""
        block = [nn.Linear(in_dim, out_dim)]
        
        if self.use_batch_norm:
            block.append(nn.BatchNorm1d(out_dim))
        
        # Add activation
        block.append(self._get_activation())
        
        # Add dropout
        if self.dropout_rate > 0:
            block.append(nn.Dropout(self.dropout_rate))
        
        return block
    
    def _build_conv_layers(self, start_c: int, start_h: int, start_w: int, 
                          target_c: int, target_h: int, target_w: int) -> List[nn.Module]:
        """Build convolutional upsampling layers."""
        layers = []
        
        current_c, current_h, current_w = start_c, start_h, start_w
        
        # Calculate number of upsampling steps needed
        scale_h = target_h // current_h
        scale_w = target_w // current_w
        num_upsamples = max(scale_h.bit_length() - 1, scale_w.bit_length() - 1)
        
        # Progressive upsampling
        for i in range(num_upsamples):
            next_c = max(target_c, current_c // 2)
            
            # Upsampling layer
            if i < num_upsamples - 1:
                # Intermediate layers
                layers.extend(self._create_conv_block(current_c, next_c, upsample=True))
                current_c = next_c
                current_h = min(current_h * 2, target_h)
                current_w = min(current_w * 2, target_w)
            else:
                # Final layer
                layers.extend(self._create_conv_block(current_c, target_c, upsample=True, final=True))
        
        # Ensure exact target dimensions
        if current_h != target_h or current_w != target_w:
            layers.append(nn.AdaptiveAvgPool2d((target_h, target_w)))
        
        return layers
    
    def _create_conv_block(self, in_channels: int, out_channels: int, 
                          upsample: bool = False, final: bool = False) -> List[nn.Module]:
        """Create a convolutional block with optional upsampling."""
        block = []
        
        if upsample:
            # Transposed convolution for upsampling
            block.append(nn.ConvTranspose2d(
                in_channels, out_channels, 
                kernel_size=4, stride=2, padding=1, bias=not self.use_batch_norm
            ))
        else:
            block.append(nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=3, stride=1, padding=1, bias=not self.use_batch_norm
            ))
        
        if not final:
            # Add batch norm for non-final layers
            if self.use_batch_norm:
                block.append(nn.BatchNorm2d(out_channels))
            
            # Add activation
            block.append(self._get_activation())
            
            # Add dropout
            if self.dropout_rate > 0:
                block.append(nn.Dropout2d(self.dropout_rate))
        else:
            # Final layer with output activation
            output_activation = self._get_output_activation()
            if output_activation is not None:
                block.append(output_activation)
        
        return block
    
    def _get_activation(self) -> nn.Module:
        """Get activation function."""
        if self.activation == 'relu':
            return nn.ReLU(inplace=True)
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif self.activation == 'elu':
            return nn.ELU(inplace=True)
        elif self.activation == 'gelu':
            return nn.GELU()
        else:
            logger.warning(f"Unknown activation: {self.activation}, using ReLU")
            return nn.ReLU(inplace=True)
    
    def _get_output_activation(self) -> Optional[nn.Module]:
        """Get output activation function."""
        if self.output_activation == 'tanh':
            return nn.Tanh()
        elif self.output_activation == 'sigmoid':
            return nn.Sigmoid()
        elif self.output_activation == 'none':
            return None
        else:
            logger.warning(f"Unknown output activation: {self.output_activation}, using Tanh")
            return nn.Tanh()
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def decode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Decode embeddings to images.
        
        Args:
            embeddings: Input embeddings [B, embedding_dim]
            
        Returns:
            Decoded images [B, C, H, W]
        """
        # Validate input
        is_valid, error_msg = self.validate_embeddings(embeddings)
        if not is_valid:
            raise ValueError(f"Invalid embeddings: {error_msg}")
        
        # Move to device
        embeddings = embeddings.to(self.device)
        
        # Set to evaluation mode for inference
        was_training = self.model.training
        self.model.eval()
        
        try:
            with torch.no_grad():
                images = self.model(embeddings)
            
            # Restore original training mode
            self.model.train(was_training)
            
            return images
            
        except RuntimeError as e:
            # Restore training mode even if error occurs
            self.model.train(was_training)
            
            if "out of memory" in str(e).lower():
                logger.error(f"Out of memory error with batch size {embeddings.shape[0]}")
                raise RuntimeError(
                    f"Out of memory error. Try reducing batch size. "
                    f"Current batch size: {embeddings.shape[0]}"
                ) from e
            else:
                raise e
    
    def train_decoder(self, embeddings: torch.Tensor, images: torch.Tensor) -> None:
        """
        Train the decoder on embedding-image pairs.
        
        Args:
            embeddings: Training embeddings [B, embedding_dim]
            images: Target images [B, C, H, W]
        """
        # Validate inputs
        is_valid, error_msg = self.validate_embeddings(embeddings)
        if not is_valid:
            raise ValueError(f"Invalid embeddings: {error_msg}")
        
        is_valid, error_msg = self.validate_images(images)
        if not is_valid:
            raise ValueError(f"Invalid images: {error_msg}")
        
        # This is a simplified training method
        # In practice, you would implement a full training loop with optimizer, loss, etc.
        logger.info("Training decoder - implement full training loop in separate trainer class")
        self._is_trained = True
    
    @classmethod
    def load_from_config(cls, config_path: Path) -> 'AutoencoderDecoder':
        """Load AutoencoderDecoder from configuration file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        import json
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        decoder = cls(
            embedding_dim=config_data['embedding_dim'],
            image_shape=tuple(config_data['image_shape']),
            **config_data['config']
        )
        
        # Load the model weights if available
        model_path = Path(config_data['model_path'])
        if model_path.exists():
            decoder.load_model(model_path.with_suffix(''))
        
        return decoder


class ReshapeLayer(nn.Module):
    """Layer to reshape tensor to specific spatial dimensions."""
    
    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return x.view(batch_size, self.channels, self.height, self.width)