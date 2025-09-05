"""
Variational Autoencoder (VAE) decoder implementation.
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


class VAEDecoder(BaseDecoder):
    """
    Variational Autoencoder decoder with reparameterization trick and KL divergence regularization.
    
    Features:
    - Reparameterization trick for stable training
    - KL divergence regularization
    - Probabilistic latent space sampling
    - Support for conditional and unconditional generation
    """
    
    def __init__(
        self,
        embedding_dim: int,
        image_shape: Tuple[int, int, int],
        latent_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        beta: float = 1.0,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
        activation: str = 'relu',
        output_activation: str = 'tanh',
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize VAE decoder.
        
        Args:
            embedding_dim: Dimension of input embeddings
            image_shape: Output image shape (C, H, W)
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
            beta: Beta parameter for beta-VAE (KL weight)
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
            activation: Activation function
            output_activation: Output activation function
            device: Device to run the model on
            config: Additional configuration parameters
        """
        # Set up configuration
        config = config or {}
        config.update({
            'latent_dim': latent_dim,
            'hidden_dims': hidden_dims,
            'beta': beta,
            'use_batch_norm': use_batch_norm,
            'dropout_rate': dropout_rate,
            'activation': activation,
            'output_activation': output_activation
        })
        
        super().__init__(embedding_dim, image_shape, device, config)
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or self._get_default_hidden_dims()
        self.beta = beta
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.output_activation = output_activation
        
        # Build and initialize model
        self.model = self._build_model()
        self.model = self.model.to(self.device)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized VAEDecoder with embedding_dim={embedding_dim}, latent_dim={latent_dim}")
    
    def _get_default_hidden_dims(self) -> List[int]:
        """Generate default hidden dimensions."""
        c, h, w = self.image_shape
        target_size = c * h * w
        
        # Create progressive scaling
        hidden_dims = []
        current_dim = self.latent_dim
        
        while current_dim < target_size // 4:
            current_dim = min(current_dim * 2, 1024)
            hidden_dims.append(current_dim)
        
        return hidden_dims
    
    def _build_model(self) -> nn.Module:
        """Build the VAE decoder model."""
        return VAEDecoderNetwork(
            embedding_dim=self.embedding_dim,
            latent_dim=self.latent_dim,
            image_shape=self.image_shape,
            hidden_dims=self.hidden_dims,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            output_activation=self.output_activation
        )
    
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
        Decode embeddings to images using the VAE decoder.
        
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
                # Get latent parameters
                mu, log_var = self.model.encode_to_latent(embeddings)
                
                # Sample from latent distribution
                latent_samples = self.reparameterize(mu, log_var)
                
                # Decode to images
                images = self.model.decode_from_latent(latent_samples)
            
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
    
    def encode_and_decode(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full encode-decode pass returning latent parameters and reconstructed images.
        
        Args:
            embeddings: Input embeddings [B, embedding_dim]
            
        Returns:
            Tuple of (reconstructed_images, mu, log_var)
        """
        embeddings = embeddings.to(self.device)
        
        # Get latent parameters
        mu, log_var = self.model.encode_to_latent(embeddings)
        
        # Sample from latent distribution
        latent_samples = self.reparameterize(mu, log_var)
        
        # Decode to images
        images = self.model.decode_from_latent(latent_samples)
        
        return images, mu, log_var
    
    def sample_from_latent(self, n_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Sample images from the latent space.
        
        Args:
            n_samples: Number of samples to generate
            device: Device for computation
            
        Returns:
            Generated images [n_samples, C, H, W]
        """
        device = device or self.device
        
        # Sample from standard normal distribution
        latent_samples = torch.randn(n_samples, self.latent_dim, device=device)
        
        # Decode to images
        was_training = self.model.training
        self.model.eval()
        
        with torch.no_grad():
            images = self.model.decode_from_latent(latent_samples)
        
        self.model.train(was_training)
        
        return images
    
    def interpolate_in_latent_space(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor, 
                                   n_steps: int = 10) -> torch.Tensor:
        """
        Interpolate between two embeddings in latent space.
        
        Args:
            embeddings1: First set of embeddings [B, embedding_dim]
            embeddings2: Second set of embeddings [B, embedding_dim]
            n_steps: Number of interpolation steps
            
        Returns:
            Interpolated images [B, n_steps, C, H, W]
        """
        embeddings1 = embeddings1.to(self.device)
        embeddings2 = embeddings2.to(self.device)
        
        # Encode to latent space
        mu1, _ = self.model.encode_to_latent(embeddings1)
        mu2, _ = self.model.encode_to_latent(embeddings2)
        
        # Create interpolation weights
        alphas = torch.linspace(0, 1, n_steps, device=self.device)
        
        interpolated_images = []
        
        was_training = self.model.training
        self.model.eval()
        
        with torch.no_grad():
            for alpha in alphas:
                # Linear interpolation in latent space
                interpolated_latent = (1 - alpha) * mu1 + alpha * mu2
                
                # Decode interpolated latent
                images = self.model.decode_from_latent(interpolated_latent)
                interpolated_images.append(images)
        
        self.model.train(was_training)
        
        # Stack along new dimension
        return torch.stack(interpolated_images, dim=1)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.
        
        Args:
            mu: Mean of latent distribution [B, latent_dim]
            log_var: Log variance of latent distribution [B, latent_dim]
            
        Returns:
            Sampled latent vectors [B, latent_dim]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def compute_kl_divergence(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between latent distribution and standard normal.
        
        Args:
            mu: Mean of latent distribution [B, latent_dim]
            log_var: Log variance of latent distribution [B, latent_dim]
            
        Returns:
            KL divergence [B]
        """
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    
    def train_decoder(self, embeddings: torch.Tensor, images: torch.Tensor) -> None:
        """
        Train the VAE decoder (placeholder for compatibility).
        
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
        
        # This is a placeholder - actual training should use VAETrainer
        logger.info("VAE training - use VAETrainer class for complete training")
        self._is_trained = True
    
    @classmethod
    def load_from_config(cls, config_path: Path) -> 'VAEDecoder':
        """Load VAEDecoder from configuration file."""
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


class VAEDecoderNetwork(nn.Module):
    """VAE decoder network implementation."""
    
    def __init__(
        self,
        embedding_dim: int,
        latent_dim: int,
        image_shape: Tuple[int, int, int],
        hidden_dims: List[int],
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
        activation: str = 'relu',
        output_activation: str = 'tanh'
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.hidden_dims = hidden_dims
        
        # Build encoder to latent space
        self.encoder_to_latent = self._build_encoder_to_latent(
            embedding_dim, latent_dim, hidden_dims, use_batch_norm, dropout_rate, activation
        )
        
        # Build decoder from latent space
        self.decoder_from_latent = self._build_decoder_from_latent(
            latent_dim, image_shape, hidden_dims, use_batch_norm, dropout_rate, activation, output_activation
        )
    
    def _build_encoder_to_latent(self, embedding_dim: int, latent_dim: int, hidden_dims: List[int],
                                use_batch_norm: bool, dropout_rate: float, activation: str) -> nn.Module:
        """Build encoder from embeddings to latent parameters."""
        layers = []
        
        # Hidden layers
        prev_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self._get_activation(activation))
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Final layers for mu and log_var
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.log_var_layer = nn.Linear(prev_dim, latent_dim)
        
        return nn.Sequential(*layers)
    
    def _build_decoder_from_latent(self, latent_dim: int, image_shape: Tuple[int, int, int], 
                                  hidden_dims: List[int], use_batch_norm: bool, dropout_rate: float,
                                  activation: str, output_activation: str) -> nn.Module:
        """Build decoder from latent space to images."""
        c, h, w = image_shape
        
        layers = []
        
        # Reverse hidden dimensions for decoder
        decoder_dims = [latent_dim] + list(reversed(hidden_dims))
        
        # Hidden layers
        for i in range(len(decoder_dims) - 1):
            layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(decoder_dims[i + 1]))
            
            layers.append(self._get_activation(activation))
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        # Final output layer
        layers.append(nn.Linear(decoder_dims[-1], c * h * w))
        
        # Output activation
        if output_activation == 'tanh':
            layers.append(nn.Tanh())
        elif output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        
        # Reshape to image
        layers.append(ReshapeToImage(c, h, w))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'elu':
            return nn.ELU(inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU(inplace=True)
    
    def encode_to_latent(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode embeddings to latent parameters."""
        hidden = self.encoder_to_latent(embeddings)
        mu = self.mu_layer(hidden)
        log_var = self.log_var_layer(hidden)
        return mu, log_var
    
    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to images."""
        return self.decoder_from_latent(latent)
    
    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass."""
        mu, log_var = self.encode_to_latent(embeddings)
        
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        latent = mu + eps * std
        
        # Decode
        images = self.decode_from_latent(latent)
        
        return images, mu, log_var


class ReshapeToImage(nn.Module):
    """Layer to reshape flattened tensor to image format."""
    
    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return x.view(batch_size, self.channels, self.height, self.width)