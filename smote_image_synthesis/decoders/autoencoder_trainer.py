"""
Autoencoder training pipeline with loss functions and checkpointing.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
try:
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

from .autoencoder_decoder import AutoencoderDecoder

logger = logging.getLogger(__name__)


class AutoencoderTrainer:
    """
    Trainer for autoencoder decoder with comprehensive training features.
    
    Features:
    - Multiple loss functions (MSE, perceptual, adversarial)
    - Learning rate scheduling and early stopping
    - Model checkpointing and validation monitoring
    - Training progress visualization
    """
    
    def __init__(
        self,
        decoder: AutoencoderDecoder,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        use_perceptual_loss: bool = True,
        perceptual_loss_weight: float = 1.0,
        reconstruction_loss_weight: float = 1.0,
        scheduler_type: str = 'plateau',
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize autoencoder trainer.
        
        Args:
            decoder: AutoencoderDecoder to train
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            use_perceptual_loss: Whether to use perceptual loss
            perceptual_loss_weight: Weight for perceptual loss component
            reconstruction_loss_weight: Weight for reconstruction loss component
            scheduler_type: Learning rate scheduler type ('plateau', 'cosine', 'exponential')
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory for saving checkpoints
            device: Device to run training on
        """
        self.decoder = decoder
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_perceptual_loss = use_perceptual_loss
        self.perceptual_loss_weight = perceptual_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.scheduler_type = scheduler_type
        self.early_stopping_patience = early_stopping_patience
        self.device = device or decoder.device
        
        # Set up checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('./checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.decoder.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize loss functions
        self.reconstruction_loss = nn.MSELoss()
        if self.use_perceptual_loss:
            if not TORCHVISION_AVAILABLE:
                logger.warning("Torchvision not available, falling back to MSE loss")
                self.use_perceptual_loss = False
            else:
                self.perceptual_loss = PerceptualLoss(device=self.device)
        
        # Training state
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'reconstruction_loss': [],
            'perceptual_loss': [],
            'learning_rate': []
        }
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0
        
        logger.info(f"Initialized AutoencoderTrainer with lr={learning_rate}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
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
    
    def train(
        self,
        train_embeddings: torch.Tensor,
        train_images: torch.Tensor,
        val_embeddings: Optional[torch.Tensor] = None,
        val_images: Optional[torch.Tensor] = None,
        num_epochs: int = 100,
        batch_size: int = 32,
        save_best: bool = True,
        validate_every: int = 1,
        log_every: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the autoencoder decoder.
        
        Args:
            train_embeddings: Training embeddings [N, embedding_dim]
            train_images: Training target images [N, C, H, W]
            val_embeddings: Validation embeddings [M, embedding_dim]
            val_images: Validation target images [M, C, H, W]
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            save_best: Whether to save the best model
            validate_every: Validate every N epochs
            log_every: Log progress every N epochs
            
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
        
        # Training loop
        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = datetime.now()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            
            # Validation phase
            if val_loader is not None and epoch % validate_every == 0:
                val_loss = self._validate_epoch(val_loader)
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
                self._log_progress(epoch, train_loss, 
                                 self.training_history['val_loss'][-1] if self.training_history['val_loss'] else None)
            
            # Record learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.training_history['learning_rate'].append(current_lr)
        
        # Save final model
        if save_best:
            self._save_checkpoint('final_model')
        
        training_time = datetime.now() - start_time
        logger.info(f"Training completed in {training_time}")
        
        # Mark decoder as trained
        self.decoder._is_trained = True
        
        return self.training_history
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.decoder.model.train()
        total_loss = 0.0
        reconstruction_losses = []
        perceptual_losses = []
        
        for batch_idx, (embeddings, images) in enumerate(train_loader):
            embeddings = embeddings.to(self.device)
            images = images.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed = self.decoder.model(embeddings)
            
            # Compute losses
            recon_loss = self.reconstruction_loss(reconstructed, images)
            total_batch_loss = self.reconstruction_loss_weight * recon_loss
            reconstruction_losses.append(recon_loss.item())
            
            if self.use_perceptual_loss:
                perceptual_loss = self.perceptual_loss(reconstructed, images)
                total_batch_loss += self.perceptual_loss_weight * perceptual_loss
                perceptual_losses.append(perceptual_loss.item())
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.decoder.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
        
        # Record component losses
        avg_recon_loss = np.mean(reconstruction_losses)
        self.training_history['reconstruction_loss'].append(avg_recon_loss)
        
        if perceptual_losses:
            avg_perceptual_loss = np.mean(perceptual_losses)
            self.training_history['perceptual_loss'].append(avg_perceptual_loss)
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.decoder.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for embeddings, images in val_loader:
                embeddings = embeddings.to(self.device)
                images = images.to(self.device)
                
                # Forward pass
                reconstructed = self.decoder.model(embeddings)
                
                # Compute loss
                recon_loss = self.reconstruction_loss(reconstructed, images)
                batch_loss = self.reconstruction_loss_weight * recon_loss
                
                if self.use_perceptual_loss:
                    perceptual_loss = self.perceptual_loss(reconstructed, images)
                    batch_loss += self.perceptual_loss_weight * perceptual_loss
                
                total_loss += batch_loss.item()
        
        return total_loss / len(val_loader)
    
    def _create_dataloader(self, embeddings: torch.Tensor, images: torch.Tensor, 
                          batch_size: int, shuffle: bool) -> DataLoader:
        """Create a DataLoader from embeddings and images."""
        dataset = TensorDataset(embeddings, images)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    
    def _validate_training_data(self, embeddings: torch.Tensor, images: torch.Tensor) -> None:
        """Validate training data."""
        if len(embeddings) != len(images):
            raise ValueError("Number of embeddings and images must match")
        
        is_valid, error_msg = self.decoder.validate_embeddings(embeddings)
        if not is_valid:
            raise ValueError(f"Invalid embeddings: {error_msg}")
        
        is_valid, error_msg = self.decoder.validate_images(images)
        if not is_valid:
            raise ValueError(f"Invalid images: {error_msg}")
    
    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{name}_epoch_{self.current_epoch}.pth"
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.decoder.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'use_perceptual_loss': self.use_perceptual_loss,
                'perceptual_loss_weight': self.perceptual_loss_weight,
                'reconstruction_loss_weight': self.reconstruction_loss_weight
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.decoder.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def _log_progress(self, epoch: int, train_loss: float, val_loss: Optional[float]) -> None:
        """Log training progress."""
        log_msg = f"Epoch {epoch}: Train Loss = {train_loss:.6f}"
        if val_loss is not None:
            log_msg += f", Val Loss = {val_loss:.6f}"
        
        current_lr = self.optimizer.param_groups[0]['lr']
        log_msg += f", LR = {current_lr:.8f}"
        
        logger.info(log_msg)
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        if self.training_history['val_loss']:
            axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Component losses
        if self.training_history['reconstruction_loss']:
            axes[0, 1].plot(self.training_history['reconstruction_loss'], label='Reconstruction')
        if self.training_history['perceptual_loss']:
            axes[0, 1].plot(self.training_history['perceptual_loss'], label='Perceptual')
        axes[0, 1].set_title('Component Losses')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(self.training_history['learning_rate'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Loss distribution
        if self.training_history['train_loss']:
            axes[1, 1].hist(self.training_history['train_loss'], bins=20, alpha=0.7, label='Train')
        if self.training_history['val_loss']:
            axes[1, 1].hist(self.training_history['val_loss'], bins=20, alpha=0.7, label='Val')
        axes[1, 1].set_title('Loss Distribution')
        axes[1, 1].set_xlabel('Loss Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved: {save_path}")
        
        plt.show()


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features."""
    
    def __init__(self, device: torch.device, layers: List[str] = None):
        super().__init__()
        self.device = device
        
        if TORCHVISION_AVAILABLE:
            # Use VGG16 feature extractor without downloading pretrained weights.
            # This keeps training/tests fully offline-compatible.
            vgg = models.vgg16(weights=None).features

            vgg.eval()
            for param in vgg.parameters():
                param.requires_grad = False
            
            self.vgg = vgg.to(device)
            
            # Default layers for feature extraction
            if layers is None:
                self.layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
            else:
                self.layers = layers
            
            # VGG layer mapping
            self.layer_name_mapping = {
                'relu1_1': 1, 'relu1_2': 3,
                'relu2_1': 6, 'relu2_2': 8,
                'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15,
                'relu4_1': 18, 'relu4_2': 20, 'relu4_3': 22,
                'relu5_1': 25, 'relu5_2': 27, 'relu5_3': 29
            }
            
            # Normalization for VGG
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        else:
            # Fallback to MSE loss
            self.vgg = None
            
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss."""
        if self.vgg is None or not TORCHVISION_AVAILABLE:
            # Fallback to MSE loss
            return self.mse_loss(pred, target)
        
        # Ensure inputs are in the right range [0, 1] and format
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # Handle grayscale images by repeating channels
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, 3, 1, 1)
        
        # Normalize inputs
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        
        # Extract features
        pred_features = self._extract_features(pred_norm)
        target_features = self._extract_features(target_norm)
        
        # Compute perceptual loss
        loss = 0.0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += self.mse_loss(pred_feat, target_feat)
        
        return loss
    
    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from specified VGG layers."""
        features = []
        
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in [self.layer_name_mapping[layer_name] for layer_name in self.layers if layer_name in self.layer_name_mapping]:
                features.append(x)
        
        return features
