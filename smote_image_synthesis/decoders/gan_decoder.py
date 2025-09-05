"""
GAN-based image decoder implementation with progressive architecture.
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


class GANDecoder(BaseDecoder):
    """
    GAN-based decoder with progressive architecture and spectral normalization.
    
    Features:
    - Progressive GAN architecture with multiple scales
    - Spectral normalization for training stability
    - Feature matching loss for better convergence
    - Adaptive discriminator for quality control
    """
    
    def __init__(
        self,
        embedding_dim: int,
        image_shape: Tuple[int, int, int],
        latent_dim: int = 128,
        generator_dims: Optional[List[int]] = None,
        discriminator_dims: Optional[List[int]] = None,
        use_spectral_norm: bool = True,
        use_self_attention: bool = True,
        progressive_training: bool = True,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize GAN decoder.
        
        Args:
            embedding_dim: Dimension of input embeddings
            image_shape: Output image shape (C, H, W)
            latent_dim: Dimension of latent noise space
            generator_dims: Generator hidden dimensions
            discriminator_dims: Discriminator hidden dimensions
            use_spectral_norm: Whether to use spectral normalization
            use_self_attention: Whether to use self-attention layers
            progressive_training: Whether to use progressive training
            device: Device to run the model on
            config: Additional configuration parameters
        """
        # Set up configuration
        config = config or {}
        config.update({
            'latent_dim': latent_dim,
            'generator_dims': generator_dims,
            'discriminator_dims': discriminator_dims,
            'use_spectral_norm': use_spectral_norm,
            'use_self_attention': use_self_attention,
            'progressive_training': progressive_training
        })
        
        super().__init__(embedding_dim, image_shape, device, config)
        
        self.latent_dim = latent_dim
        self.generator_dims = generator_dims or self._get_default_generator_dims()
        self.discriminator_dims = discriminator_dims or self._get_default_discriminator_dims()
        self.use_spectral_norm = use_spectral_norm
        self.use_self_attention = use_self_attention
        self.progressive_training = progressive_training
        
        # Build models
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Move to device
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        
        # Progressive training state
        self.current_scale = 0
        self.max_scale = self._calculate_max_scale()
        self.training_step = 0
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized GANDecoder with embedding_dim={embedding_dim}, latent_dim={latent_dim}")
    
    def _get_default_generator_dims(self) -> List[int]:
        """Generate default generator dimensions."""
        c, h, w = self.image_shape
        
        # Start from small spatial size and work backwards
        dims = []
        current_spatial = 4  # Start with 4x4
        current_channels = 512
        
        while current_spatial < max(h, w):
            dims.append(current_channels)
            current_spatial *= 2
            current_channels = max(64, current_channels // 2)
        
        dims.append(c)  # Final output channels
        return dims
    
    def _get_default_discriminator_dims(self) -> List[int]:
        """Generate default discriminator dimensions."""
        c, h, w = self.image_shape
        
        # Start from input and work to smaller sizes
        dims = [c]
        current_channels = 64
        current_spatial = max(h, w)
        
        while current_spatial > 4:
            dims.append(current_channels)
            current_spatial //= 2
            current_channels = min(512, current_channels * 2)
        
        dims.append(1)  # Final output (real/fake)
        return dims
    
    def _calculate_max_scale(self) -> int:
        """Calculate maximum scale for progressive training."""
        c, h, w = self.image_shape
        max_dim = max(h, w)
        scale = 0
        size = 4
        
        while size < max_dim:
            scale += 1
            size *= 2
        
        return scale
    
    def _build_generator(self) -> nn.Module:
        """Build the generator network."""
        return ProgressiveGenerator(
            embedding_dim=self.embedding_dim,
            latent_dim=self.latent_dim,
            output_channels=self.image_shape[0],
            dims=self.generator_dims,
            use_spectral_norm=self.use_spectral_norm,
            use_self_attention=self.use_self_attention,
            max_scale=self.max_scale
        )
    
    def _build_discriminator(self) -> nn.Module:
        """Build the discriminator network."""
        return ProgressiveDiscriminator(
            input_channels=self.image_shape[0],
            dims=self.discriminator_dims,
            use_spectral_norm=self.use_spectral_norm,
            use_self_attention=self.use_self_attention,
            max_scale=self.max_scale
        )
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        def init_func(module):
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(module.weight, 0.0, 0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, 1.0, 0.02)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.generator.apply(init_func)
        self.discriminator.apply(init_func)
    
    def decode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Decode embeddings to images using the generator.
        
        Args:
            embeddings: Input embeddings [B, embedding_dim]
            
        Returns:
            Generated images [B, C, H, W]
        """
        # Validate input
        is_valid, error_msg = self.validate_embeddings(embeddings)
        if not is_valid:
            raise ValueError(f"Invalid embeddings: {error_msg}")
        
        # Move to device
        embeddings = embeddings.to(self.device)
        
        # Set to evaluation mode
        was_training = self.generator.training
        self.generator.eval()
        
        try:
            with torch.no_grad():
                # Generate noise
                batch_size = embeddings.shape[0]
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                
                # Generate images
                images = self.generator(embeddings, noise, scale=self.current_scale)
            
            # Restore training mode
            self.generator.train(was_training)
            
            return images
            
        except RuntimeError as e:
            # Restore training mode
            self.generator.train(was_training)
            
            if "out of memory" in str(e).lower():
                logger.error(f"Out of memory error with batch size {embeddings.shape[0]}")
                raise RuntimeError(
                    f"Out of memory error. Try reducing batch size. "
                    f"Current batch size: {embeddings.shape[0]}"
                ) from e
            else:
                raise e
    
    def generate_with_noise(self, embeddings: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Generate images with specific noise input.
        
        Args:
            embeddings: Input embeddings [B, embedding_dim]
            noise: Noise vector [B, latent_dim]
            
        Returns:
            Generated images [B, C, H, W]
        """
        embeddings = embeddings.to(self.device)
        noise = noise.to(self.device)
        
        was_training = self.generator.training
        self.generator.eval()
        
        with torch.no_grad():
            images = self.generator(embeddings, noise, scale=self.current_scale)
        
        self.generator.train(was_training)
        return images
    
    def discriminate(self, images: torch.Tensor) -> torch.Tensor:
        """
        Discriminate real vs fake images.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Discrimination scores [B, 1]
        """
        images = images.to(self.device)
        
        was_training = self.discriminator.training
        self.discriminator.eval()
        
        with torch.no_grad():
            scores = self.discriminator(images, scale=self.current_scale)
        
        self.discriminator.train(was_training)
        return scores
    
    def set_training_scale(self, scale: int):
        """Set the current training scale for progressive training."""
        self.current_scale = min(scale, self.max_scale)
        logger.info(f"Set training scale to {self.current_scale}")
    
    def get_feature_matching_loss(
        self, 
        real_images: torch.Tensor, 
        fake_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute feature matching loss between real and fake images.
        
        Args:
            real_images: Real images [B, C, H, W]
            fake_images: Generated images [B, C, H, W]
            
        Returns:
            Feature matching loss
        """
        # Get intermediate features from discriminator
        real_features = self.discriminator.get_features(real_images, scale=self.current_scale)
        fake_features = self.discriminator.get_features(fake_images, scale=self.current_scale)
        
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.l1_loss(fake_feat, real_feat.detach())
        
        return loss
    
    def train_decoder(self, embeddings: torch.Tensor, images: torch.Tensor) -> None:
        """
        Train the GAN decoder (placeholder for compatibility).
        
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
        
        # GAN training requires separate trainer class
        logger.info("GAN training - use GANTrainer class for complete training")
        self._is_trained = True
    
    @classmethod
    def load_from_config(cls, config_path: Path) -> 'GANDecoder':
        """Load GANDecoder from configuration file."""
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
        
        # Load model weights if available
        model_path = Path(config_data['model_path'])
        if model_path.exists():
            decoder.load_model(model_path.with_suffix(''))
        
        return decoder


class ProgressiveGenerator(nn.Module):
    """Progressive GAN generator with spectral normalization."""
    
    def __init__(
        self,
        embedding_dim: int,
        latent_dim: int,
        output_channels: int,
        dims: List[int],
        use_spectral_norm: bool = True,
        use_self_attention: bool = True,
        max_scale: int = 4
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.use_spectral_norm = use_spectral_norm
        self.use_self_attention = use_self_attention
        self.max_scale = max_scale
        
        # Embedding processor
        self.embedding_processor = nn.Sequential(
            nn.Linear(embedding_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Initial projection
        self.initial_projection = nn.Linear(latent_dim * 2, dims[0] * 4 * 4)
        
        # Progressive blocks
        self.blocks = nn.ModuleList()
        self.to_rgb_layers = nn.ModuleList()
        
        for i in range(max_scale + 1):
            if i == 0:
                # Initial 4x4 block
                block = nn.Sequential(
                    nn.Conv2d(dims[0], dims[0], 3, padding=1),
                    nn.BatchNorm2d(dims[0]),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            else:
                # Upsampling blocks
                in_channels = dims[max(0, i-1)]
                out_channels = dims[min(i, len(dims)-1)]
                
                layers = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True)
                ]
                
                # Add self-attention at middle scales
                if use_self_attention and i == max_scale // 2:
                    layers.append(SelfAttention(out_channels))
                
                block = nn.Sequential(*layers)
            
            if use_spectral_norm:
                block = self._apply_spectral_norm(block)
            
            self.blocks.append(block)
            
            # To RGB layers
            channels = dims[min(i, len(dims)-1)]
            to_rgb = nn.Conv2d(channels, output_channels, 1)
            if use_spectral_norm:
                to_rgb = nn.utils.spectral_norm(to_rgb)
            self.to_rgb_layers.append(to_rgb)
    
    def _apply_spectral_norm(self, module: nn.Module) -> nn.Module:
        """Apply spectral normalization to conv layers in module."""
        for child in module.children():
            if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.utils.spectral_norm(child)
            else:
                self._apply_spectral_norm(child)
        return module
    
    def forward(self, embeddings: torch.Tensor, noise: torch.Tensor, scale: int) -> torch.Tensor:
        """
        Generate images at specified scale.
        
        Args:
            embeddings: Conditioning embeddings [B, embedding_dim]
            noise: Random noise [B, latent_dim]
            scale: Training scale (0 to max_scale)
            
        Returns:
            Generated images [B, C, H, W]
        """
        batch_size = embeddings.shape[0]
        
        # Process embeddings
        processed_embeddings = self.embedding_processor(embeddings)
        
        # Combine noise and embeddings
        combined_input = torch.cat([noise, processed_embeddings], dim=1)
        
        # Initial projection
        x = self.initial_projection(combined_input)
        x = x.view(batch_size, -1, 4, 4)
        
        # Progressive generation
        for i in range(min(scale + 1, len(self.blocks))):
            x = self.blocks[i](x)
        
        # Convert to RGB
        if scale < len(self.to_rgb_layers):
            rgb = self.to_rgb_layers[scale](x)
        else:
            rgb = self.to_rgb_layers[-1](x)
        
        return torch.tanh(rgb)


class ProgressiveDiscriminator(nn.Module):
    """Progressive GAN discriminator with spectral normalization."""
    
    def __init__(
        self,
        input_channels: int,
        dims: List[int],
        use_spectral_norm: bool = True,
        use_self_attention: bool = True,
        max_scale: int = 4
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.use_spectral_norm = use_spectral_norm
        self.use_self_attention = use_self_attention
        self.max_scale = max_scale
        
        # From RGB layers
        self.from_rgb_layers = nn.ModuleList()
        
        # Progressive blocks
        self.blocks = nn.ModuleList()
        
        for i in range(max_scale + 1):
            # From RGB layer
            channels = dims[min(i, len(dims)-2)]
            from_rgb = nn.Conv2d(input_channels, channels, 1)
            if use_spectral_norm:
                from_rgb = nn.utils.spectral_norm(from_rgb)
            self.from_rgb_layers.append(from_rgb)
            
            # Discriminator block
            if i == max_scale:
                # Final block
                layers = [
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(channels, channels, 4, padding=0),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Flatten(),
                    nn.Linear(channels, 1)
                ]
            else:
                # Downsampling blocks
                out_channels = dims[min(i + 1, len(dims)-1)]
                
                layers = [
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(channels, out_channels, 3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.AvgPool2d(2)
                ]
                
                # Add self-attention at middle scales
                if use_self_attention and i == max_scale // 2:
                    layers.insert(-1, SelfAttention(out_channels))
            
            block = nn.Sequential(*layers)
            
            if use_spectral_norm:
                block = self._apply_spectral_norm(block)
            
            self.blocks.append(block)
    
    def _apply_spectral_norm(self, module: nn.Module) -> nn.Module:
        """Apply spectral normalization to conv layers in module."""
        for child in module.children():
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                nn.utils.spectral_norm(child)
            else:
                self._apply_spectral_norm(child)
        return module
    
    def forward(self, images: torch.Tensor, scale: int) -> torch.Tensor:
        """
        Discriminate images at specified scale.
        
        Args:
            images: Input images [B, C, H, W]
            scale: Training scale
            
        Returns:
            Discrimination scores [B, 1]
        """
        # Convert from RGB
        if scale < len(self.from_rgb_layers):
            x = self.from_rgb_layers[scale](images)
        else:
            x = self.from_rgb_layers[-1](images)
        
        x = F.leaky_relu(x, 0.2, inplace=True)
        
        # Progressive discrimination (reverse order)
        start_block = max(0, scale)
        for i in range(start_block, len(self.blocks)):
            x = self.blocks[i](x)
        
        return x
    
    def get_features(self, images: torch.Tensor, scale: int) -> List[torch.Tensor]:
        """Get intermediate features for feature matching loss."""
        features = []
        
        # Convert from RGB
        if scale < len(self.from_rgb_layers):
            x = self.from_rgb_layers[scale](images)
        else:
            x = self.from_rgb_layers[-1](images)
        
        x = F.leaky_relu(x, 0.2, inplace=True)
        features.append(x)
        
        # Progressive discrimination
        start_block = max(0, scale)
        for i in range(start_block, len(self.blocks) - 1):  # Exclude final layer
            x = self.blocks[i](x)
            features.append(x)
        
        return features


class SelfAttention(nn.Module):
    """Self-attention layer for improved feature learning."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.channels = channels
        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention to input tensor.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor with self-attention applied
        """
        batch_size, channels, height, width = x.shape
        
        # Generate query, key, value
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        
        # Compute attention
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Add residual connection with learnable weight
        return self.gamma * out + x