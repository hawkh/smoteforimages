"""
VAE training and inference pipeline with combined loss function.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt

from .vae_decoder import VAEDecoder

logger = logging.getLogger(__name__)


class VAETrainer:
    """
    Trainer for VAE decoder with combined loss function and latent space analysis.
    
    Features:
    - Combined reconstruction and KL divergence loss
    - Beta-VAE support for controllable disentanglement
    - Latent space interpolation and sampling
    - Training progress monitoring and visualization
    """
    
    def __init__(
        self,
        vae_decoder: VAEDecoder,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        beta: float = 1.0,
        reconstruction_loss_type: str = 'mse',
        scheduler_type: str = 'plateau',
        early_stopping_patience: int = 15,
        checkpoint_dir: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize VAE trainer.
        
        Args:
            vae_decoder: VAEDecoder to train
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            beta: Beta parameter for beta-VAE (KL weight)
            reconstruction_loss_type: Type of reconstruction loss ('mse', 'bce', 'l1')
            scheduler_type: Learning rate scheduler type
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory for saving checkpoints
            device: Device to run training on
        """
        self.vae_decoder = vae_decoder
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta = beta
        self.reconstruction_loss_type = reconstruction_loss_type
        self.scheduler_type = scheduler_type
        self.early_stopping_patience = early_stopping_patience
        self.device = device or vae_decoder.device
        
        # Set up checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('./vae_checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.vae_decoder.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize loss functions
        self.reconstruction_loss = self._get_reconstruction_loss()
        
        # Training state
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'learning_rate': [],
            'beta_values': []
        }
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0
        
        logger.info(f"Initialized VAETrainer with lr={learning_rate}, beta={beta}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        elif self.scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100, eta_min=1e-6
            )
        elif self.scheduler_type == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.95
            )
        elif self.scheduler_type == 'none':
            return None
        else:
            logger.warning(f"Unknown scheduler type: {self.scheduler_type}, using plateau")
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
    
    def _get_reconstruction_loss(self) -> nn.Module:
        """Get reconstruction loss function."""
        if self.reconstruction_loss_type == 'mse':
            return nn.MSELoss(reduction='sum')
        elif self.reconstruction_loss_type == 'bce':
            return nn.BCELoss(reduction='sum')
        elif self.reconstruction_loss_type == 'l1':
            return nn.L1Loss(reduction='sum')
        else:
            logger.warning(f"Unknown reconstruction loss: {self.reconstruction_loss_type}, using MSE")
            return nn.MSELoss(reduction='sum')
    
    def train(
        self,
        train_embeddings: torch.Tensor,
        train_images: torch.Tensor,
        val_embeddings: Optional[torch.Tensor] = None,
        val_images: Optional[torch.Tensor] = None,
        num_epochs: int = 200,
        batch_size: int = 32,
        beta_schedule: Optional[Dict[str, Any]] = None,
        save_best: bool = True,
        validate_every: int = 1,
        log_every: int = 10,
        sample_every: int = 20
    ) -> Dict[str, List[float]]:
        """
        Train the VAE decoder.
        
        Args:
            train_embeddings: Training embeddings [N, embedding_dim]
            train_images: Training target images [N, C, H, W]
            val_embeddings: Validation embeddings [M, embedding_dim]
            val_images: Validation target images [M, C, H, W]
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            beta_schedule: Schedule for beta parameter (e.g., {'type': 'linear', 'start': 0.0, 'end': 1.0})
            save_best: Whether to save the best model
            validate_every: Validate every N epochs
            log_every: Log progress every N epochs
            sample_every: Generate samples every N epochs
            
        Returns:
            Training history dictionary
        """
        # Validate inputs
        self._validate_training_data(train_embeddings, train_images)
        if val_embeddings is not None and val_images is not None:
            self._validate_training_data(val_embeddings, val_images)
        
        # Create data loaders
        train_loader = self._create_dataloader(train_embeddings, train_images, batch_size, shuffle=True)
        val_loader = None
        if val_embeddings is not None and val_images is not None:
            val_loader = self._create_dataloader(val_embeddings, val_images, batch_size, shuffle=False)
        
        # Initialize beta scheduler
        beta_scheduler = BetaScheduler(beta_schedule, num_epochs) if beta_schedule else None
        
        # Training loop
        logger.info(f"Starting VAE training for {num_epochs} epochs")
        start_time = datetime.now()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Update beta if using schedule
            if beta_scheduler:
                self.beta = beta_scheduler.get_beta(epoch)
            self.training_history['beta_values'].append(self.beta)
            
            # Training phase
            train_loss, train_recon_loss, train_kl_loss = self._train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['reconstruction_loss'].append(train_recon_loss)
            self.training_history['kl_loss'].append(train_kl_loss)
            
            # Validation phase
            if val_loader is not None and epoch % validate_every == 0:
                val_loss, val_recon_loss, val_kl_loss = self._validate_epoch(val_loader)
                self.training_history['val_loss'].append(val_loss)
                
                # Update learning rate scheduler
                if self.scheduler is not None and self.scheduler_type == 'plateau':
                    self.scheduler.step(val_loss)
                
                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    
                    if save_best:
                        self._save_checkpoint('best_model')
                else:
                    self.patience_counter += 1
                
                # Early stopping check
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            else:
                # No validation, just update scheduler
                if self.scheduler is not None and self.scheduler_type != 'plateau':
                    self.scheduler.step()
            
            # Logging
            if epoch % log_every == 0 or epoch == num_epochs - 1:
                self._log_progress(epoch, train_loss, train_recon_loss, train_kl_loss,
                                 self.training_history['val_loss'][-1] if self.training_history['val_loss'] else None)
            
            # Generate samples for monitoring
            if epoch % sample_every == 0:
                self._generate_sample_images(epoch)
            
            # Record learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.training_history['learning_rate'].append(current_lr)
        
        # Save final model
        if save_best:
            self._save_checkpoint('final_model')
        
        training_time = datetime.now() - start_time
        logger.info(f"VAE training completed in {training_time}")
        
        # Mark decoder as trained
        self.vae_decoder._is_trained = True
        
        return self.training_history
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.vae_decoder.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        for batch_idx, (embeddings, images) in enumerate(train_loader):
            embeddings = embeddings.to(self.device)
            images = images.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed, mu, log_var = self.vae_decoder.encode_and_decode(embeddings)
            
            # Compute losses
            recon_loss = self.reconstruction_loss(reconstructed, images) / embeddings.size(0)
            kl_loss = torch.mean(self.vae_decoder.compute_kl_divergence(mu, log_var))
            
            # Combined loss
            total_batch_loss = recon_loss + self.beta * kl_loss
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.vae_decoder.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
        
        return (total_loss / len(train_loader), 
                total_recon_loss / len(train_loader),
                total_kl_loss / len(train_loader))
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """Validate for one epoch."""
        self.vae_decoder.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        with torch.no_grad():
            for embeddings, images in val_loader:
                embeddings = embeddings.to(self.device)
                images = images.to(self.device)
                
                # Forward pass
                reconstructed, mu, log_var = self.vae_decoder.encode_and_decode(embeddings)
                
                # Compute losses
                recon_loss = self.reconstruction_loss(reconstructed, images) / embeddings.size(0)
                kl_loss = torch.mean(self.vae_decoder.compute_kl_divergence(mu, log_var))
                
                # Combined loss
                batch_loss = recon_loss + self.beta * kl_loss
                
                total_loss += batch_loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
        
        return (total_loss / len(val_loader),
                total_recon_loss / len(val_loader),
                total_kl_loss / len(val_loader))
    
    def _create_dataloader(self, embeddings: torch.Tensor, images: torch.Tensor, 
                          batch_size: int, shuffle: bool) -> DataLoader:
        """Create a DataLoader from embeddings and images."""
        dataset = TensorDataset(embeddings, images)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    
    def _validate_training_data(self, embeddings: torch.Tensor, images: torch.Tensor) -> None:
        """Validate training data."""
        if len(embeddings) != len(images):
            raise ValueError("Number of embeddings and images must match")
        
        is_valid, error_msg = self.vae_decoder.validate_embeddings(embeddings)
        if not is_valid:
            raise ValueError(f"Invalid embeddings: {error_msg}")
        
        is_valid, error_msg = self.vae_decoder.validate_images(images)
        if not is_valid:
            raise ValueError(f"Invalid images: {error_msg}")
    
    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{name}_epoch_{self.current_epoch}.pth"
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.vae_decoder.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'beta': self.beta,
            'config': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'beta': self.beta,
                'reconstruction_loss_type': self.reconstruction_loss_type
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"VAE checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.vae_decoder.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        self.beta = checkpoint.get('beta', self.beta)
        
        logger.info(f"VAE checkpoint loaded: {checkpoint_path}")
    
    def _generate_sample_images(self, epoch: int, n_samples: int = 16) -> None:
        """Generate sample images for monitoring."""
        try:
            samples = self.vae_decoder.sample_from_latent(n_samples)
            
            # Save samples
            sample_dir = self.checkpoint_dir / 'samples'
            sample_dir.mkdir(exist_ok=True)
            
            # Create visualization
            grid_size = int(np.sqrt(n_samples))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
            
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    if idx < n_samples:
                        img = samples[idx].cpu().detach()
                        # Convert from [-1, 1] to [0, 1] if using tanh
                        if self.vae_decoder.output_activation == 'tanh':
                            img = (img + 1) / 2
                        
                        if img.shape[0] == 1:  # Grayscale
                            axes[i, j].imshow(img.squeeze(), cmap='gray')
                        else:  # RGB
                            img = img.permute(1, 2, 0)
                            axes[i, j].imshow(img)
                        
                        axes[i, j].axis('off')
            
            plt.suptitle(f'VAE Samples - Epoch {epoch}')
            plt.tight_layout()
            
            sample_path = sample_dir / f'samples_epoch_{epoch}.png'
            plt.savefig(sample_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to generate sample images: {e}")
    
    def _log_progress(self, epoch: int, train_loss: float, recon_loss: float, 
                     kl_loss: float, val_loss: Optional[float]) -> None:
        """Log training progress."""
        log_msg = (f"Epoch {epoch}: Train Loss = {train_loss:.6f}, "
                  f"Recon = {recon_loss:.6f}, KL = {kl_loss:.6f}, "
                  f"Beta = {self.beta:.4f}")
        
        if val_loss is not None:
            log_msg += f", Val Loss = {val_loss:.6f}"
        
        current_lr = self.optimizer.param_groups[0]['lr']
        log_msg += f", LR = {current_lr:.8f}"
        
        logger.info(log_msg)
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot VAE training history."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Total loss
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        if self.training_history['val_loss']:
            axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Component losses
        axes[0, 1].plot(self.training_history['reconstruction_loss'], label='Reconstruction')
        axes[0, 1].plot(self.training_history['kl_loss'], label='KL Divergence')
        axes[0, 1].set_title('Component Losses')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Beta schedule
        axes[0, 2].plot(self.training_history['beta_values'])
        axes[0, 2].set_title('Beta Schedule')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Beta')
        axes[0, 2].grid(True)
        
        # Learning rate
        axes[1, 0].plot(self.training_history['learning_rate'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Loss ratios
        if len(self.training_history['reconstruction_loss']) > 0 and len(self.training_history['kl_loss']) > 0:
            recon = np.array(self.training_history['reconstruction_loss'])
            kl = np.array(self.training_history['kl_loss'])
            ratio = recon / (kl + 1e-8)
            axes[1, 1].plot(ratio)
            axes[1, 1].set_title('Reconstruction/KL Ratio')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Ratio')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        # Loss distribution
        if self.training_history['train_loss']:
            axes[1, 2].hist(self.training_history['train_loss'], bins=20, alpha=0.7, label='Train')
        if self.training_history['val_loss']:
            axes[1, 2].hist(self.training_history['val_loss'], bins=20, alpha=0.7, label='Val')
        axes[1, 2].set_title('Loss Distribution')
        axes[1, 2].set_xlabel('Loss Value')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"VAE training history plot saved: {save_path}")
        
        plt.show()


class BetaScheduler:
    """Scheduler for beta parameter in beta-VAE."""
    
    def __init__(self, schedule_config: Optional[Dict[str, Any]], num_epochs: int):
        """
        Initialize beta scheduler.
        
        Args:
            schedule_config: Configuration for beta schedule
            num_epochs: Total number of training epochs
        """
        self.schedule_config = schedule_config or {}
        self.num_epochs = num_epochs
        self.schedule_type = self.schedule_config.get('type', 'constant')
        
    def get_beta(self, epoch: int) -> float:
        """Get beta value for given epoch."""
        if self.schedule_type == 'constant':
            return self.schedule_config.get('value', 1.0)
        
        elif self.schedule_type == 'linear':
            start = self.schedule_config.get('start', 0.0)
            end = self.schedule_config.get('end', 1.0)
            warmup_epochs = self.schedule_config.get('warmup_epochs', self.num_epochs)
            
            if epoch >= warmup_epochs:
                return end
            else:
                return start + (end - start) * (epoch / warmup_epochs)
        
        elif self.schedule_type == 'cyclical':
            period = self.schedule_config.get('period', 50)
            min_beta = self.schedule_config.get('min', 0.0)
            max_beta = self.schedule_config.get('max', 1.0)
            
            cycle_position = (epoch % period) / period
            return min_beta + (max_beta - min_beta) * (0.5 * (1 + np.cos(np.pi * cycle_position)))
        
        else:
            return 1.0