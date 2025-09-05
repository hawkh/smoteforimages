"""
GAN trainer implementation for training GAN-based decoders.
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
import seaborn as sns

from .gan_decoder import GANDecoder

logger = logging.getLogger(__name__)


class GANTrainer:
    """
    Trainer for GAN-based decoders with progressive training support.
    
    Features:
    - Progressive training with scale scheduling
    - Feature matching loss for better convergence
    - Gradient penalty for training stability
    - Learning rate scheduling and early stopping
    - Comprehensive logging and visualization
    """
    
    def __init__(
        self,
        gan_decoder: GANDecoder,
        generator_lr: float = 0.0002,
        discriminator_lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        feature_matching_weight: float = 10.0,
        gradient_penalty_weight: float = 10.0,
        use_progressive_training: bool = True,
        warmup_epochs_per_scale: int = 5,
        training_epochs_per_scale: int = 20,
        device: Optional[torch.device] = None
    ):
        """
        Initialize GAN trainer.
        
        Args:
            gan_decoder: GAN decoder to train
            generator_lr: Learning rate for generator
            discriminator_lr: Learning rate for discriminator
            beta1: Adam optimizer beta1 parameter
            beta2: Adam optimizer beta2 parameter
            feature_matching_weight: Weight for feature matching loss
            gradient_penalty_weight: Weight for gradient penalty
            use_progressive_training: Whether to use progressive training
            warmup_epochs_per_scale: Warmup epochs for each scale
            training_epochs_per_scale: Training epochs for each scale
            device: Device to run training on
        """
        self.gan_decoder = gan_decoder
        self.device = device or gan_decoder.device
        
        # Training parameters
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.feature_matching_weight = feature_matching_weight
        self.gradient_penalty_weight = gradient_penalty_weight
        
        # Progressive training parameters
        self.use_progressive_training = use_progressive_training
        self.warmup_epochs_per_scale = warmup_epochs_per_scale
        self.training_epochs_per_scale = training_epochs_per_scale
        
        # Initialize optimizers
        self.generator_optimizer = optim.Adam(
            self.gan_decoder.generator.parameters(),
            lr=generator_lr,
            betas=(beta1, beta2)
        )
        
        self.discriminator_optimizer = optim.Adam(
            self.gan_decoder.discriminator.parameters(),
            lr=discriminator_lr,
            betas=(beta1, beta2)
        )
        
        # Loss function
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        
        # Training state
        self.current_epoch = 0
        self.training_history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'feature_matching_loss': [],
            'gradient_penalty': []
        }
        
        logger.info(f"Initialized GANTrainer with progressive_training={use_progressive_training}")
    
    def train(
        self,
        train_embeddings: torch.Tensor,
        train_images: torch.Tensor,
        val_embeddings: Optional[torch.Tensor] = None,
        val_images: Optional[torch.Tensor] = None,
        num_epochs: Optional[int] = None,
        batch_size: int = 32,
        save_checkpoint_every: int = 10,
        checkpoint_dir: Optional[str] = None,
        visualize_progress: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the GAN decoder.
        
        Args:
            train_embeddings: Training embeddings [N, embedding_dim]
            train_images: Training images [N, C, H, W]
            val_embeddings: Optional validation embeddings
            val_images: Optional validation images
            num_epochs: Total number of epochs (calculated if None)
            batch_size: Training batch size
            save_checkpoint_every: Save checkpoint every N epochs
            checkpoint_dir: Directory to save checkpoints
            visualize_progress: Whether to visualize training progress
            
        Returns:
            Training history dictionary
        """
        # Validate inputs
        self._validate_training_data(train_embeddings, train_images)
        
        # Create data loader
        train_dataset = TensorDataset(train_embeddings, train_images)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Calculate total epochs if not provided
        if num_epochs is None:
            if self.use_progressive_training:
                num_epochs = (self.warmup_epochs_per_scale + self.training_epochs_per_scale) * (self.gan_decoder.max_scale + 1)
            else:
                num_epochs = 100
        
        # Setup checkpoint directory
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting GAN training for {num_epochs} epochs")
        
        # Progressive training loop
        if self.use_progressive_training:
            return self._train_progressive(
                train_loader, val_embeddings, val_images,
                save_checkpoint_every, checkpoint_path, visualize_progress
            )
        else:
            return self._train_standard(
                train_loader, num_epochs, val_embeddings, val_images,
                save_checkpoint_every, checkpoint_path, visualize_progress
            )
    
    def _train_progressive(
        self,
        train_loader: DataLoader,
        val_embeddings: Optional[torch.Tensor],
        val_images: Optional[torch.Tensor],
        save_checkpoint_every: int,
        checkpoint_path: Optional[Path],
        visualize_progress: bool
    ) -> Dict[str, List[float]]:
        """Train with progressive scaling."""
        for scale in range(self.gan_decoder.max_scale + 1):
            logger.info(f"Training scale {scale}/{self.gan_decoder.max_scale}")
            
            # Set training scale
            self.gan_decoder.set_training_scale(scale)
            
            # Warmup phase
            logger.info(f"Warmup phase for scale {scale}")
            self._train_epochs(
                train_loader, self.warmup_epochs_per_scale,
                val_embeddings, val_images, save_checkpoint_every,
                checkpoint_path, visualize_progress, scale_prefix=f"scale_{scale}_warmup"
            )
            
            # Training phase
            logger.info(f"Training phase for scale {scale}")
            self._train_epochs(
                train_loader, self.training_epochs_per_scale,
                val_embeddings, val_images, save_checkpoint_every,
                checkpoint_path, visualize_progress, scale_prefix=f"scale_{scale}_train"
            )
        
        return self.training_history
    
    def _train_standard(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_embeddings: Optional[torch.Tensor],
        val_images: Optional[torch.Tensor],
        save_checkpoint_every: int,
        checkpoint_path: Optional[Path],
        visualize_progress: bool
    ) -> Dict[str, List[float]]:
        """Train with standard (non-progressive) approach."""
        # Set to maximum scale
        self.gan_decoder.set_training_scale(self.gan_decoder.max_scale)
        
        self._train_epochs(
            train_loader, num_epochs, val_embeddings, val_images,
            save_checkpoint_every, checkpoint_path, visualize_progress
        )
        
        return self.training_history
    
    def _train_epochs(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_embeddings: Optional[torch.Tensor],
        val_images: Optional[torch.Tensor],
        save_checkpoint_every: int,
        checkpoint_path: Optional[Path],
        visualize_progress: bool,
        scale_prefix: str = ""
    ) -> None:
        """Train for specified number of epochs."""
        for epoch in range(num_epochs):
            self.current_epoch += 1
            
            # Training step
            epoch_losses = self._train_epoch(train_loader)
            
            # Update history
            for key, value in epoch_losses.items():
                self.training_history[key].append(value)
            
            # Validation
            if val_embeddings is not None and val_images is not None:
                val_loss = self._validate(val_embeddings, val_images)
                logger.info(f"Epoch {self.current_epoch}: Val Loss = {val_loss:.4f}")
            
            # Logging
            if epoch % 5 == 0:
                logger.info(f"Epoch {self.current_epoch}: "
                          f"G_Loss = {epoch_losses['generator_loss']:.4f}, "
                          f"D_Loss = {epoch_losses['discriminator_loss']:.4f}, "
                          f"FM_Loss = {epoch_losses['feature_matching_loss']:.4f}")
            
            # Save checkpoint
            if checkpoint_path and epoch % save_checkpoint_every == 0:
                self._save_checkpoint(checkpoint_path, scale_prefix)
            
            # Visualize progress
            if visualize_progress and epoch % 10 == 0:
                self._visualize_progress()
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.gan_decoder.generator.train()
        self.gan_decoder.discriminator.train()
        
        epoch_losses = {
            'generator_loss': 0.0,
            'discriminator_loss': 0.0,
            'feature_matching_loss': 0.0,
            'gradient_penalty': 0.0
        }
        
        num_batches = len(train_loader)
        
        for batch_idx, (embeddings, real_images) in enumerate(train_loader):
            embeddings = embeddings.to(self.device)
            real_images = real_images.to(self.device)
            batch_size = embeddings.shape[0]
            
            # Labels
            real_labels = torch.ones(batch_size, 1, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, device=self.device)
            
            # Train Discriminator
            self.discriminator_optimizer.zero_grad()
            
            # Real images
            real_scores = self.gan_decoder.discriminator(real_images, self.gan_decoder.current_scale)
            real_loss = self.adversarial_loss(real_scores, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, self.gan_decoder.latent_dim, device=self.device)
            fake_images = self.gan_decoder.generator(embeddings, noise, self.gan_decoder.current_scale)
            fake_scores = self.gan_decoder.discriminator(fake_images.detach(), self.gan_decoder.current_scale)
            fake_loss = self.adversarial_loss(fake_scores, fake_labels)
            
            # Gradient penalty
            gradient_penalty = self._compute_gradient_penalty(real_images, fake_images)
            
            # Total discriminator loss
            d_loss = real_loss + fake_loss + self.gradient_penalty_weight * gradient_penalty
            d_loss.backward()
            self.discriminator_optimizer.step()
            
            # Train Generator
            self.generator_optimizer.zero_grad()
            
            # Generate fake images
            noise = torch.randn(batch_size, self.gan_decoder.latent_dim, device=self.device)
            fake_images = self.gan_decoder.generator(embeddings, noise, self.gan_decoder.current_scale)
            fake_scores = self.gan_decoder.discriminator(fake_images, self.gan_decoder.current_scale)
            
            # Adversarial loss
            g_adversarial_loss = self.adversarial_loss(fake_scores, real_labels)
            
            # Feature matching loss
            feature_matching_loss = self.gan_decoder.get_feature_matching_loss(real_images, fake_images)
            
            # Total generator loss
            g_loss = g_adversarial_loss + self.feature_matching_weight * feature_matching_loss
            g_loss.backward()
            self.generator_optimizer.step()
            
            # Update epoch losses
            epoch_losses['generator_loss'] += g_loss.item()
            epoch_losses['discriminator_loss'] += d_loss.item()
            epoch_losses['feature_matching_loss'] += feature_matching_loss.item()
            epoch_losses['gradient_penalty'] += gradient_penalty.item()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def _compute_gradient_penalty(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP."""
        batch_size = real_images.shape[0]
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        interpolated.requires_grad_(True)
        
        # Get discriminator output
        d_interpolated = self.gan_decoder.discriminator(interpolated, self.gan_decoder.current_scale)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def _validate(self, val_embeddings: torch.Tensor, val_images: torch.Tensor) -> float:
        """Validate the model."""
        self.gan_decoder.generator.eval()
        self.gan_decoder.discriminator.eval()
        
        with torch.no_grad():
            val_embeddings = val_embeddings.to(self.device)
            val_images = val_images.to(self.device)
            
            # Generate fake images
            batch_size = val_embeddings.shape[0]
            noise = torch.randn(batch_size, self.gan_decoder.latent_dim, device=self.device)
            fake_images = self.gan_decoder.generator(val_embeddings, noise, self.gan_decoder.current_scale)
            
            # Compute feature matching loss
            feature_matching_loss = self.gan_decoder.get_feature_matching_loss(val_images, fake_images)
            
            return feature_matching_loss.item()
    
    def _validate_training_data(self, embeddings: torch.Tensor, images: torch.Tensor) -> None:
        """Validate training data format."""
        if embeddings.shape[0] != images.shape[0]:
            raise ValueError("Number of embeddings and images must match")
        
        if embeddings.shape[1] != self.gan_decoder.embedding_dim:
            raise ValueError(f"Expected embedding dimension {self.gan_decoder.embedding_dim}, "
                           f"got {embeddings.shape[1]}")
        
        expected_image_shape = (images.shape[0],) + self.gan_decoder.image_shape
        if images.shape != expected_image_shape:
            raise ValueError(f"Expected image shape {expected_image_shape}, got {images.shape}")
    
    def _save_checkpoint(self, checkpoint_path: Path, prefix: str = "") -> None:
        """Save training checkpoint."""
        checkpoint_name = f"{prefix}_epoch_{self.current_epoch}.pth" if prefix else f"epoch_{self.current_epoch}.pth"
        checkpoint_file = checkpoint_path / checkpoint_name
        
        torch.save({
            'epoch': self.current_epoch,
            'generator_state_dict': self.gan_decoder.generator.state_dict(),
            'discriminator_state_dict': self.gan_decoder.discriminator.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'training_history': self.training_history,
            'current_scale': self.gan_decoder.current_scale
        }, checkpoint_file)
        
        logger.info(f"Checkpoint saved to {checkpoint_file}")
    
    def _visualize_progress(self) -> None:
        """Visualize training progress."""
        if not self.training_history['generator_loss']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Generator loss
        axes[0, 0].plot(self.training_history['generator_loss'])
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        # Discriminator loss
        axes[0, 1].plot(self.training_history['discriminator_loss'])
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        
        # Feature matching loss
        axes[1, 0].plot(self.training_history['feature_matching_loss'])
        axes[1, 0].set_title('Feature Matching Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        
        # Gradient penalty
        axes[1, 1].plot(self.training_history['gradient_penalty'])
        axes[1, 1].set_title('Gradient Penalty')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Penalty')
        
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Training progress visualization updated at epoch {self.current_epoch}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.gan_decoder.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.gan_decoder.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.gan_decoder.current_scale = checkpoint['current_scale']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}, epoch {self.current_epoch}")
    
    def generate_samples(
        self, 
        embeddings: torch.Tensor, 
        num_samples: int = 8,
        save_path: Optional[str] = None
    ) -> torch.Tensor:
        """Generate and optionally save sample images."""
        self.gan_decoder.generator.eval()
        
        with torch.no_grad():
            embeddings = embeddings[:num_samples].to(self.device)
            noise = torch.randn(num_samples, self.gan_decoder.latent_dim, device=self.device)
            
            generated_images = self.gan_decoder.generator(
                embeddings, noise, self.gan_decoder.current_scale
            )
            
            if save_path:
                self._save_sample_grid(generated_images, save_path)
            
            return generated_images
    
    def _save_sample_grid(self, images: torch.Tensor, save_path: str) -> None:
        """Save a grid of sample images."""
        from torchvision.utils import save_image
        
        # Denormalize images (assuming tanh output)
        images = (images + 1) / 2.0
        images = torch.clamp(images, 0, 1)
        
        save_image(images, save_path, nrow=int(np.sqrt(images.shape[0])), padding=2)
        logger.info(f"Sample grid saved to {save_path}")