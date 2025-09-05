"""
Diffusion model trainer for training diffusion-based decoders.
"""

from typing import Tuple, Optional, Dict, Any, List, Callable
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

from .diffusion_decoder import DiffusionDecoder

logger = logging.getLogger(__name__)


class DiffusionTrainer:
    """
    Trainer for diffusion-based decoders.
    
    Features:
    - DDPM training with noise prediction
    - Progressive loss weighting
    - Learning rate scheduling
    - Gradient clipping for stability
    - Comprehensive logging and visualization
    """
    
    def __init__(
        self,
        diffusion_decoder: DiffusionDecoder,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        grad_clip_norm: float = 1.0,
        loss_type: str = 'l2',
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        device: Optional[torch.device] = None
    ):
        """
        Initialize diffusion trainer.
        
        Args:
            diffusion_decoder: Diffusion decoder to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            beta1: Adam optimizer beta1 parameter
            beta2: Adam optimizer beta2 parameter
            grad_clip_norm: Gradient clipping norm
            loss_type: Loss function type ('l1', 'l2', 'huber')
            use_ema: Whether to use exponential moving average
            ema_decay: EMA decay rate
            device: Device to run training on
        """
        self.diffusion_decoder = diffusion_decoder
        self.device = device or diffusion_decoder.device
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.loss_type = loss_type
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.diffusion_decoder.unet.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
        
        # Loss function
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # EMA model
        if use_ema:
            self.ema_model = self._create_ema_model()
        else:
            self.ema_model = None
        
        # Training state
        self.current_epoch = 0
        self.training_history = {
            'loss': [],
            'learning_rate': []
        }
        
        logger.info(f"Initialized DiffusionTrainer with loss_type={loss_type}, use_ema={use_ema}")
    
    def _create_ema_model(self) -> nn.Module:
        """Create exponential moving average model."""
        ema_model = type(self.diffusion_decoder.unet)(
            **{k: v for k, v in self.diffusion_decoder.unet.__dict__.items() 
               if not k.startswith('_')}
        )
        ema_model.load_state_dict(self.diffusion_decoder.unet.state_dict())
        ema_model.eval()
        ema_model.requires_grad_(False)
        return ema_model.to(self.device)
    
    def _update_ema(self) -> None:
        """Update EMA model parameters."""
        if self.ema_model is None:
            return
        
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.diffusion_decoder.unet.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def train(
        self,
        train_embeddings: torch.Tensor,
        train_images: torch.Tensor,
        val_embeddings: Optional[torch.Tensor] = None,
        val_images: Optional[torch.Tensor] = None,
        num_epochs: int = 100,
        batch_size: int = 16,
        save_checkpoint_every: int = 10,
        checkpoint_dir: Optional[str] = None,
        visualize_progress: bool = True,
        lr_scheduler: Optional[str] = 'cosine'
    ) -> Dict[str, List[float]]:
        """
        Train the diffusion decoder.
        
        Args:
            train_embeddings: Training embeddings [N, embedding_dim]
            train_images: Training images [N, C, H, W]
            val_embeddings: Optional validation embeddings
            val_images: Optional validation images
            num_epochs: Number of training epochs
            batch_size: Training batch size
            save_checkpoint_every: Save checkpoint every N epochs
            checkpoint_dir: Directory to save checkpoints
            visualize_progress: Whether to visualize training progress
            lr_scheduler: Learning rate scheduler type
            
        Returns:
            Training history dictionary
        """
        # Validate inputs
        self._validate_training_data(train_embeddings, train_images)
        
        # Create data loader
        train_dataset = TensorDataset(train_embeddings, train_images)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Setup learning rate scheduler
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs
            )
        elif lr_scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=num_epochs // 3, gamma=0.5
            )
        else:
            scheduler = None
        
        # Setup checkpoint directory
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting diffusion training for {num_epochs} epochs")
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training step
            epoch_loss = self._train_epoch(train_loader)
            
            # Update history
            self.training_history['loss'].append(epoch_loss)
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            if scheduler:
                scheduler.step()
            
            # Validation
            if val_embeddings is not None and val_images is not None:
                val_loss = self._validate(val_embeddings, val_images, batch_size)
                logger.info(f"Epoch {epoch + 1}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch + 1}: Train Loss = {epoch_loss:.4f}")
            
            # Save checkpoint
            if checkpoint_path and (epoch + 1) % save_checkpoint_every == 0:
                self._save_checkpoint(checkpoint_path, epoch)
            
            # Visualize progress
            if visualize_progress and (epoch + 1) % 10 == 0:
                self._visualize_progress(train_embeddings[:4], train_images[:4])
        
        logger.info("Diffusion training completed")
        return self.training_history
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.diffusion_decoder.unet.train()
        
        epoch_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (embeddings, images) in enumerate(train_loader):
            embeddings = embeddings.to(self.device)
            images = images.to(self.device)
            batch_size = images.shape[0]
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, self.diffusion_decoder.num_timesteps,
                (batch_size,), device=self.device
            )
            
            # Sample noise
            noise = torch.randn_like(images)
            
            # Add noise to images
            noisy_images = self.diffusion_decoder.add_noise(images, noise, timesteps)
            
            # Predict noise
            self.optimizer.zero_grad()
            predicted_noise = self.diffusion_decoder.unet(noisy_images, timesteps, embeddings)
            
            # Compute loss
            loss = self.loss_fn(predicted_noise, noise)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.diffusion_decoder.unet.parameters(), 
                    self.grad_clip_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update EMA
            if self.use_ema:
                self._update_ema()
            
            epoch_loss += loss.item()
        
        return epoch_loss / num_batches
    
    def _validate(
        self, 
        val_embeddings: torch.Tensor, 
        val_images: torch.Tensor,
        batch_size: int
    ) -> float:
        """Validate the model."""
        self.diffusion_decoder.unet.eval()
        
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_embeddings), batch_size):
                embeddings = val_embeddings[i:i + batch_size].to(self.device)
                images = val_images[i:i + batch_size].to(self.device)
                current_batch_size = images.shape[0]
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, self.diffusion_decoder.num_timesteps,
                    (current_batch_size,), device=self.device
                )
                
                # Sample noise
                noise = torch.randn_like(images)
                
                # Add noise to images
                noisy_images = self.diffusion_decoder.add_noise(images, noise, timesteps)
                
                # Predict noise
                predicted_noise = self.diffusion_decoder.unet(noisy_images, timesteps, embeddings)
                
                # Compute loss
                loss = self.loss_fn(predicted_noise, noise)
                val_loss += loss.item()
                num_batches += 1
        
        return val_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_training_data(self, embeddings: torch.Tensor, images: torch.Tensor) -> None:
        """Validate training data format."""
        if embeddings.shape[0] != images.shape[0]:
            raise ValueError("Number of embeddings and images must match")
        
        if embeddings.shape[1] != self.diffusion_decoder.embedding_dim:
            raise ValueError(f"Expected embedding dimension {self.diffusion_decoder.embedding_dim}, "
                           f"got {embeddings.shape[1]}")
        
        expected_image_shape = (images.shape[0],) + self.diffusion_decoder.image_shape
        if images.shape != expected_image_shape:
            raise ValueError(f"Expected image shape {expected_image_shape}, got {images.shape}")
    
    def _save_checkpoint(self, checkpoint_path: Path, epoch: int) -> None:
        """Save training checkpoint."""
        checkpoint_file = checkpoint_path / f"diffusion_epoch_{epoch + 1}.pth"
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.diffusion_decoder.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': self.diffusion_decoder.config
        }
        
        if self.ema_model:
            checkpoint_data['ema_state_dict'] = self.ema_model.state_dict()
        
        torch.save(checkpoint_data, checkpoint_file)
        logger.info(f"Checkpoint saved to {checkpoint_file}")
    
    def _visualize_progress(
        self, 
        sample_embeddings: torch.Tensor, 
        sample_images: torch.Tensor
    ) -> None:
        """Visualize training progress with sample generations."""
        self.diffusion_decoder.unet.eval()
        
        with torch.no_grad():
            # Generate samples using current model
            generated_images = self.diffusion_decoder.ddpm_sample(
                sample_embeddings, 
                num_inference_steps=50  # Faster inference for visualization
            )
            
            # Convert to numpy and denormalize
            real_images = sample_images.cpu().numpy()
            generated_images = generated_images.cpu().numpy()
            
            # Create comparison plot
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            
            for i in range(4):
                # Real images
                if real_images.shape[1] == 3:  # RGB
                    axes[0, i].imshow(np.transpose(real_images[i], (1, 2, 0)))
                else:  # Grayscale
                    axes[0, i].imshow(real_images[i, 0], cmap='gray')
                axes[0, i].set_title(f"Real {i + 1}")
                axes[0, i].axis('off')
                
                # Generated images
                if generated_images.shape[1] == 3:  # RGB
                    axes[1, i].imshow(np.transpose(generated_images[i], (1, 2, 0)))
                else:  # Grayscale
                    axes[1, i].imshow(generated_images[i, 0], cmap='gray')
                axes[1, i].set_title(f"Generated {i + 1}")
                axes[1, i].axis('off')
            
            plt.suptitle(f"Training Progress - Epoch {self.current_epoch + 1}")
            plt.tight_layout()
            plt.show()
        
        self.diffusion_decoder.unet.train()
    
    def generate_samples(
        self, 
        embeddings: torch.Tensor, 
        num_inference_steps: int = 1000,
        use_ema: bool = True,
        save_path: Optional[str] = None
    ) -> torch.Tensor:
        """Generate samples using trained model."""
        # Use EMA model if available and requested
        if use_ema and self.ema_model is not None:
            original_model = self.diffusion_decoder.unet
            self.diffusion_decoder.unet = self.ema_model
        
        try:
            generated_images = self.diffusion_decoder.ddpm_sample(
                embeddings, 
                num_inference_steps=num_inference_steps
            )
            
            if save_path:
                self._save_sample_grid(generated_images, save_path)
            
            return generated_images
        
        finally:
            # Restore original model if EMA was used
            if use_ema and self.ema_model is not None:
                self.diffusion_decoder.unet = original_model
    
    def _save_sample_grid(self, images: torch.Tensor, save_path: str) -> None:
        """Save a grid of sample images."""
        from torchvision.utils import save_image
        
        # Denormalize images (assuming range [-1, 1])
        images = (images + 1) / 2.0
        images = torch.clamp(images, 0, 1)
        
        save_image(images, save_path, nrow=int(np.sqrt(images.shape[0])), padding=2)
        logger.info(f"Sample grid saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.diffusion_decoder.unet.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        
        if 'ema_state_dict' in checkpoint and self.ema_model:
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}, epoch {self.current_epoch}")
    
    def plot_training_history(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.training_history['loss'], label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax2.plot(self.training_history['learning_rate'], label='Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig