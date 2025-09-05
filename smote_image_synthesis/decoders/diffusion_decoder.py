"""
Diffusion model decoder implementation with U-Net backbone.
"""

from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import math

from .base import BaseDecoder

logger = logging.getLogger(__name__)


class DiffusionDecoder(BaseDecoder):
    """
    Diffusion model decoder with U-Net backbone for high-quality image synthesis.
    
    Features:
    - U-Net architecture with attention mechanisms
    - DDPM/DDIM sampling strategies
    - Embedding conditioning at multiple scales
    - Progressive noise scheduling
    - Memory-efficient inference
    """
    
    def __init__(
        self,
        embedding_dim: int,
        image_shape: Tuple[int, int, int],
        unet_dims: Optional[List[int]] = None,
        num_timesteps: int = 1000,
        beta_schedule: str = 'linear',
        attention_resolutions: Optional[List[int]] = None,
        num_heads: int = 8,
        use_checkpoint: bool = False,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize diffusion decoder.
        
        Args:
            embedding_dim: Dimension of input embeddings
            image_shape: Output image shape (C, H, W)
            unet_dims: U-Net channel dimensions
            num_timesteps: Number of diffusion timesteps
            beta_schedule: Noise schedule ('linear', 'cosine', 'quadratic')
            attention_resolutions: Resolutions at which to apply attention
            num_heads: Number of attention heads
            use_checkpoint: Whether to use gradient checkpointing
            device: Device to run the model on
            config: Additional configuration parameters
        """
        # Set up configuration
        config = config or {}
        config.update({
            'unet_dims': unet_dims,
            'num_timesteps': num_timesteps,
            'beta_schedule': beta_schedule,
            'attention_resolutions': attention_resolutions,
            'num_heads': num_heads,
            'use_checkpoint': use_checkpoint
        })
        
        super().__init__(embedding_dim, image_shape, device, config)
        
        self.unet_dims = unet_dims or self._get_default_unet_dims()
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.attention_resolutions = attention_resolutions or [16, 8]
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        
        # Build U-Net model
        self.unet = self._build_unet()
        self.unet = self.unet.to(self.device)
        
        # Setup noise schedule
        self.betas = self._setup_noise_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Pre-computed values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        # Move to device
        self._move_schedule_to_device()
        
        logger.info(f"Initialized DiffusionDecoder with {num_timesteps} timesteps")
    
    def _get_default_unet_dims(self) -> List[int]:
        """Generate default U-Net dimensions."""
        c, h, w = self.image_shape
        base_channels = 128
        
        # Calculate number of levels based on image size
        max_res = max(h, w)
        num_levels = int(math.log2(max_res)) - 2  # Stop at 4x4
        
        dims = []
        for i in range(num_levels):
            dims.append(base_channels * (2 ** i))
        
        return dims
    
    def _setup_noise_schedule(self) -> torch.Tensor:
        """Setup noise schedule for diffusion process."""
        if self.beta_schedule == 'linear':
            betas = torch.linspace(0.0001, 0.02, self.num_timesteps)
        elif self.beta_schedule == 'cosine':
            # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
            def cosine_beta_schedule(timesteps, s=0.008):
                steps = timesteps + 1
                x = torch.linspace(0, timesteps, steps)
                alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
                alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
                betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
                return torch.clip(betas, 0, 0.999)
            
            betas = cosine_beta_schedule(self.num_timesteps)
        elif self.beta_schedule == 'quadratic':
            betas = torch.linspace(0.0001**0.5, 0.02**0.5, self.num_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        
        return betas
    
    def _move_schedule_to_device(self):
        """Move noise schedule tensors to device."""
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)
        self.posterior_variance = self.posterior_variance.to(self.device)
    
    def _build_unet(self) -> nn.Module:
        """Build U-Net model."""
        return UNetModel(
            input_channels=self.image_shape[0],
            output_channels=self.image_shape[0],
            embedding_dim=self.embedding_dim,
            model_channels=self.unet_dims[0],
            channel_mult=tuple(dim // self.unet_dims[0] for dim in self.unet_dims),
            attention_resolutions=self.attention_resolutions,
            num_heads=self.num_heads,
            use_checkpoint=self.use_checkpoint,
            image_size=max(self.image_shape[1], self.image_shape[2])
        )
    
    def decode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Decode embeddings to images using DDPM sampling.
        
        Args:
            embeddings: Input embeddings [B, embedding_dim]
            
        Returns:
            Generated images [B, C, H, W]
        """
        # Validate input
        is_valid, error_msg = self.validate_embeddings(embeddings)
        if not is_valid:
            raise ValueError(f"Invalid embeddings: {error_msg}")
        
        return self.ddpm_sample(embeddings)
    
    def ddpm_sample(
        self,
        embeddings: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        DDPM/DDIM sampling from noise to image.
        
        Args:
            embeddings: Conditioning embeddings [B, embedding_dim]
            num_inference_steps: Number of inference steps (uses all timesteps if None)
            eta: DDIM parameter (0.0 = DDIM, 1.0 = DDPM)
            
        Returns:
            Generated images [B, C, H, W]
        """
        batch_size = embeddings.shape[0]
        embeddings = embeddings.to(self.device)
        
        # Set evaluation mode
        was_training = self.unet.training
        self.unet.eval()
        
        try:
            with torch.no_grad():
                # Start from pure noise
                x = torch.randn(
                    batch_size, 
                    self.image_shape[0], 
                    self.image_shape[1], 
                    self.image_shape[2],
                    device=self.device
                )
                
                # Determine timesteps for inference
                if num_inference_steps is None:
                    timesteps = list(range(self.num_timesteps))[::-1]
                else:
                    # Use subset of timesteps for faster inference
                    timesteps = list(range(0, self.num_timesteps, self.num_timesteps // num_inference_steps))[::-1]
                
                # Denoising loop
                for i, t in enumerate(timesteps):
                    t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                    
                    # Predict noise
                    predicted_noise = self.unet(x, t_tensor, embeddings)
                    
                    # Compute denoising step
                    if eta == 0.0:
                        # DDIM sampling
                        x = self._ddim_step(x, predicted_noise, t, timesteps, i)
                    else:
                        # DDPM sampling
                        x = self._ddpm_step(x, predicted_noise, t, eta)
                
                # Clamp to valid range
                x = torch.clamp(x, -1.0, 1.0)
                
                return x
        
        finally:
            # Restore training mode
            self.unet.train(was_training)
    
    def _ddpm_step(
        self, 
        x: torch.Tensor, 
        predicted_noise: torch.Tensor, 
        t: int, 
        eta: float
    ) -> torch.Tensor:
        """Single DDPM denoising step."""
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod_prev[t] if t > 0 else torch.tensor(1.0)
        
        # Predict x_0
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # Compute direction to x_t
        direction = torch.sqrt(1 - alpha_t_prev) * predicted_noise
        
        # Add noise if not final step
        noise = 0
        if t > 0:
            noise = eta * torch.sqrt(self.posterior_variance[t]) * torch.randn_like(x)
        
        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_t_prev) * pred_x0 + direction + noise
        
        return x_prev
    
    def _ddim_step(
        self, 
        x: torch.Tensor, 
        predicted_noise: torch.Tensor, 
        t: int,
        timesteps: List[int],
        step_idx: int
    ) -> torch.Tensor:
        """Single DDIM denoising step."""
        alpha_t = self.alphas_cumprod[t]
        
        if step_idx == len(timesteps) - 1:
            alpha_t_prev = torch.tensor(1.0)
        else:
            alpha_t_prev = self.alphas_cumprod[timesteps[step_idx + 1]]
        
        # Predict x_0
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * predicted_noise
        
        return x_prev
    
    def add_noise(
        self, 
        x0: torch.Tensor, 
        noise: torch.Tensor, 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to clean images according to noise schedule."""
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_cumprod.shape) < len(x0.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
        
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
    
    def train_decoder(self, embeddings: torch.Tensor, images: torch.Tensor) -> None:
        """
        Train the diffusion decoder (placeholder for compatibility).
        
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
        
        # Diffusion training requires separate trainer class
        logger.info("Diffusion training - use DiffusionTrainer class for complete training")
        self._is_trained = True
    
    @classmethod
    def load_from_config(cls, config_path: Path) -> 'DiffusionDecoder':
        """Load DiffusionDecoder from configuration file."""
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


class UNetModel(nn.Module):
    """
    U-Net model for diffusion with attention and embedding conditioning.
    """
    
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        embedding_dim: int,
        model_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        attention_resolutions: List[int] = [16, 8],
        num_heads: int = 8,
        use_checkpoint: bool = False,
        image_size: int = 256
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.embedding_dim = embedding_dim
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Embedding conditioning
        self.embedding_proj = nn.Sequential(
            nn.Linear(embedding_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Input projection
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                nn.Conv2d(input_channels, model_channels, 3, padding=1)
            )
        ])
        
        # Encoder blocks
        input_block_channels = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(2):  # 2 blocks per level
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        out_channels=mult * model_channels
                    )
                ]
                ch = mult * model_channels
                
                # Add attention if at specified resolution
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads, use_checkpoint=use_checkpoint)
                    )
                
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(ch)
            
            # Downsample (except last level)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch))
                )
                input_block_channels.append(ch)
                ds *= 2
        
        # Middle block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim),
            AttentionBlock(ch, num_heads=num_heads, use_checkpoint=use_checkpoint),
            ResBlock(ch, time_embed_dim)
        )
        
        # Decoder blocks
        self.output_blocks = nn.ModuleList([])
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(3):  # 3 blocks per level (including skip connection)
                ich = input_block_channels.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        out_channels=mult * model_channels
                    )
                ]
                ch = mult * model_channels
                
                # Add attention if at specified resolution
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads, use_checkpoint=use_checkpoint)
                    )
                
                # Upsample on last block of each level (except first level)
                if level and i == 2:
                    layers.append(Upsample(ch))
                    ds //= 2
                
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        
        # Output projection
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, output_channels, 3, padding=1)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        timesteps: torch.Tensor, 
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            x: Noisy input [B, C, H, W]
            timesteps: Timestep indices [B]
            embeddings: Conditioning embeddings [B, embedding_dim]
            
        Returns:
            Predicted noise [B, C, H, W]
        """
        # Embed timesteps and conditioning
        t_emb = self.time_embed(timesteps)
        cond_emb = self.embedding_proj(embeddings)
        emb = t_emb + cond_emb
        
        # Encoder
        h = x
        hs = []
        
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        
        # Middle
        h = self.middle_block(h, emb)
        
        # Decoder
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        
        # Output
        return self.out(h)


class TimestepEmbedSequential(nn.Sequential):
    """Sequential module that passes timestep embeddings to its children."""
    
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, (ResBlock, AttentionBlock)):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """Residual block with timestep conditioning."""
    
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float = 0.0,
        out_channels: Optional[int] = None,
        use_conv: bool = False
    ):
        super().__init__()
        
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1)
        )
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels)
        )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        )
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)
    
    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        h = h + emb_out
        h = self.out_layers(h)
        
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """Self-attention block."""
    
    def __init__(
        self, 
        channels: int, 
        num_heads: int = 1, 
        use_checkpoint: bool = False
    ):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x, emb=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)
    
    def _forward(self, x):
        b, c, h, w = x.shape
        
        # Reshape to sequence
        x_reshaped = x.reshape(b, c, h * w)
        
        # Compute QKV
        qkv = self.qkv(self.norm(x_reshaped))
        q, k, v = qkv.chunk(3, dim=1)
        
        # Multi-head attention
        q = q.view(b, self.num_heads, c // self.num_heads, h * w)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(c // self.num_heads)
        weight = torch.einsum('bhcq,bhck->bhqk', q * scale, k)
        weight = torch.softmax(weight, dim=-1)
        
        # Apply attention
        h_att = torch.einsum('bhqk,bhck->bhcq', weight, v)
        h_att = h_att.reshape(b, c, h * w)
        
        # Output projection
        h_att = self.proj_out(h_att)
        
        # Reshape back and add residual
        return x + h_att.reshape(b, c, h, w)


class Downsample(nn.Module):
    """Downsampling layer."""
    
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        
        self.channels = channels
        self.use_conv = use_conv
        
        if use_conv:
            self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(2, stride=2)
    
    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    """Upsampling layer."""
    
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        
        self.channels = channels
        self.use_conv = use_conv
        
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        if self.use_conv:
            x = self.conv(x)
        
        return x