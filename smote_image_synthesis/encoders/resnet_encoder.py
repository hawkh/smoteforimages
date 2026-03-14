"""
ResNet-based image encoder implementation.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import logging
import json

from .base import ImageEncoder

logger = logging.getLogger(__name__)


class ResNetEncoder(ImageEncoder):
    """
    ResNet-based image encoder with configurable architecture.
    
    Supports ResNet18, ResNet50, and ResNet101 architectures with pretrained weights.
    Includes fine-tuning capabilities and memory-efficient batch processing.
    """
    
    SUPPORTED_ARCHITECTURES = {
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101
    }
    
    def __init__(
        self,
        architecture: str = 'resnet50',
        embedding_dim: int = 512,
        pretrained: bool = True,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.1
    ):
        """
        Initialize ResNet encoder.
        
        Args:
            architecture: ResNet architecture ('resnet18', 'resnet50', 'resnet101')
            embedding_dim: Dimension of output embeddings
            pretrained: Whether to use pretrained ImageNet weights
            device: Device to run the model on
            config: Additional configuration parameters
            freeze_backbone: Whether to freeze backbone weights during training
            dropout_rate: Dropout rate for the final embedding layer
        """
        if architecture not in self.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                f"Supported: {list(self.SUPPORTED_ARCHITECTURES.keys())}"
            )
        
        # Set up configuration
        config = config or {}
        config.update({
            'freeze_backbone': freeze_backbone,
            'dropout_rate': dropout_rate
        })
        
        super().__init__(
            architecture=architecture,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            device=device,
            config=config
        )
        
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout_rate
        
        # Build and initialize model
        self.model = self._build_model()
        self.model = self.model.to(self.device)
        
        # Apply backbone freezing if requested
        if self.freeze_backbone:
            self._freeze_backbone()
        
        logger.info(f"Initialized {architecture} encoder with embedding_dim={embedding_dim}")
    
    def _build_model(self) -> nn.Module:
        """
        Build the ResNet encoder model.
        
        Returns:
            Complete ResNet encoder model
        """
        # Load base ResNet model
        resnet_class = self.SUPPORTED_ARCHITECTURES[self.architecture]
        if self.pretrained:
            try:
                backbone = resnet_class(pretrained=True)
            except Exception as exc:
                logger.warning(
                    "Failed to load pretrained weights (%s). Falling back to random initialization.",
                    exc,
                )
                backbone = resnet_class(pretrained=False)
        else:
            backbone = resnet_class(pretrained=False)
        
        # Get the number of features from the last layer
        num_features = backbone.fc.in_features
        
        # Remove the original classification head
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Create custom embedding head
        embedding_head = self._create_embedding_head(num_features)
        
        # Combine backbone and embedding head
        model = nn.Sequential(
            backbone,
            nn.Flatten(),
            embedding_head
        )
        
        return model
    
    def _create_embedding_head(self, input_features: int) -> nn.Module:
        """
        Create the embedding head that maps ResNet features to target embedding dimension.
        
        Args:
            input_features: Number of input features from ResNet backbone
            
        Returns:
            Embedding head module
        """
        layers = []
        
        # Add dropout for regularization
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))
        
        # Linear projection to embedding dimension
        layers.append(nn.Linear(input_features, self.embedding_dim))
        
        # Optional batch normalization
        if self.config.get('use_batch_norm', True):
            layers.append(nn.BatchNorm1d(self.embedding_dim))
        
        # Optional activation function
        activation = self.config.get('activation', 'relu')
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'none':
            pass  # No activation
        else:
            logger.warning(f"Unknown activation: {activation}, using ReLU")
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _freeze_backbone(self) -> None:
        """Freeze the ResNet backbone parameters."""
        backbone = self.model[0]  # First part is the backbone
        for param in backbone.parameters():
            param.requires_grad = False
        
        logger.info("ResNet backbone frozen")
    
    def _unfreeze_backbone(self) -> None:
        """Unfreeze the ResNet backbone parameters."""
        backbone = self.model[0]  # First part is the backbone
        for param in backbone.parameters():
            param.requires_grad = True
        
        logger.info("ResNet backbone unfrozen")
    
    def set_backbone_frozen(self, frozen: bool) -> None:
        """
        Set whether the backbone should be frozen.
        
        Args:
            frozen: Whether to freeze the backbone
        """
        self.freeze_backbone = frozen
        if frozen:
            self._freeze_backbone()
        else:
            self._unfreeze_backbone()
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of images to embeddings.
        
        Args:
            images: Batch of images [B, C, H, W]
            
        Returns:
            Embeddings [B, embedding_dim]
        """
        # Validate input
        is_valid, error_msg = self.validate_input(images)
        if not is_valid:
            raise ValueError(f"Invalid input: {error_msg}")
        
        # Move to device
        images = images.to(self.device)
        
        # Set to evaluation mode for inference
        was_training = self.model.training
        self.model.eval()
        
        try:
            with torch.no_grad():
                embeddings = self.model(images)
            
            # Restore original training mode
            self.model.train(was_training)
            
            return embeddings
            
        except RuntimeError as e:
            # Restore training mode even if error occurs
            self.model.train(was_training)
            
            if "out of memory" in str(e).lower():
                logger.error(f"Out of memory error with batch size {images.shape[0]}")
                raise RuntimeError(
                    f"Out of memory error. Try reducing batch size. "
                    f"Current batch size: {images.shape[0]}"
                ) from e
            else:
                raise e
    
    def encode_with_memory_management(
        self, 
        images: torch.Tensor, 
        max_batch_size: int = 32
    ) -> torch.Tensor:
        """
        Encode images with automatic memory management.
        
        Args:
            images: Batch of images [B, C, H, W]
            max_batch_size: Maximum batch size to process at once
            
        Returns:
            Embeddings [B, embedding_dim]
        """
        if images.shape[0] <= max_batch_size:
            return self.encode(images)
        
        # Process in smaller batches
        all_embeddings = []
        for i in range(0, images.shape[0], max_batch_size):
            batch = images[i:i + max_batch_size]
            batch_embeddings = self.encode(batch)
            all_embeddings.append(batch_embeddings)
        
        return torch.cat(all_embeddings, dim=0)
    
    def fine_tune(
        self, 
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int = 5,
        learning_rate: float = 1e-4,
        unfreeze_after_epochs: int = 2
    ) -> Dict[str, list]:
        """
        Fine-tune the encoder on a specific dataset.
        
        Args:
            dataloader: DataLoader with (images, labels) pairs
            num_epochs: Number of training epochs
            learning_rate: Learning rate for fine-tuning
            unfreeze_after_epochs: Epoch after which to unfreeze backbone
            
        Returns:
            Training history dictionary
        """
        # Set up optimizer and loss function
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {'loss': [], 'accuracy': []}
        
        self.model.train()

        # Determine num_classes by scanning all labels before training starts
        if not hasattr(self, 'classifier'):
            all_labels = torch.cat([lbls for _, lbls in dataloader])
            num_classes = int(all_labels.max().item()) + 1
            self.classifier = nn.Linear(self.embedding_dim, num_classes).to(self.device)

        for epoch in range(num_epochs):
            # Unfreeze backbone after specified epochs
            if epoch == unfreeze_after_epochs and self.freeze_backbone:
                self._unfreeze_backbone()
                # Update optimizer to include newly unfrozen parameters
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=learning_rate * 0.1  # Lower learning rate for backbone
                )
                logger.info(f"Unfroze backbone at epoch {epoch}")
            
            epoch_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                embeddings = self.model(images)
                
                logits = self.classifier(embeddings)
                loss = criterion(logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
            
            # Calculate metrics
            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct_predictions / total_samples
            
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
        
        self._is_trained = True
        return history
    
    def get_feature_maps(self, images: torch.Tensor, layer_name: str = 'layer4') -> torch.Tensor:
        """
        Extract feature maps from a specific layer.
        
        Args:
            images: Input images [B, C, H, W]
            layer_name: Name of the layer to extract features from
            
        Returns:
            Feature maps from the specified layer
        """
        # This is a simplified implementation
        # In practice, you'd need to register hooks to extract intermediate features
        backbone = self.model[0]
        
        # Set to evaluation mode
        was_training = backbone.training
        backbone.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            
            # Forward pass through backbone layers
            x = images
            for name, layer in backbone.named_children():
                x = layer(x)
                if name == layer_name:
                    break
        
        # Restore training mode
        backbone.train(was_training)
        
        return x
    
    @classmethod
    def load_from_config(cls, config_path: Union[str, Path]) -> 'ResNetEncoder':
        """
        Load ResNet encoder from configuration file.
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            Loaded ResNet encoder instance
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Extract ResNet-specific parameters
        freeze_backbone = config_data['config'].get('freeze_backbone', False)
        dropout_rate = config_data['config'].get('dropout_rate', 0.1)
        
        encoder = cls(
            architecture=config_data['architecture'],
            embedding_dim=config_data['embedding_dim'],
            pretrained=config_data['pretrained'],
            config=config_data['config'],
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )
        
        # Load the model weights if available
        model_path = Path(config_data['model_path'])
        if model_path.exists():
            encoder.load_model(model_path.with_suffix(''))
        
        return encoder
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the ResNet encoder.
        
        Returns:
            Dictionary containing model information
        """
        info = super().get_model_info()
        
        # Add ResNet-specific information
        info.update({
            'freeze_backbone': self.freeze_backbone,
            'dropout_rate': self.dropout_rate,
            'supported_architectures': list(self.SUPPORTED_ARCHITECTURES.keys())
        })
        
        # Add layer information
        if self.model is not None:
            backbone = self.model[0]
            embedding_head = self.model[2]
            
            backbone_params = sum(p.numel() for p in backbone.parameters())
            head_params = sum(p.numel() for p in embedding_head.parameters())
            
            info.update({
                'backbone_parameters': backbone_params,
                'embedding_head_parameters': head_params,
                'backbone_frozen': not any(p.requires_grad for p in backbone.parameters())
            })
        
        return info
    
    def __repr__(self) -> str:
        """String representation of the ResNet encoder."""
        return (f"ResNetEncoder("
                f"architecture='{self.architecture}', "
                f"embedding_dim={self.embedding_dim}, "
                f"pretrained={self.pretrained}, "
                f"freeze_backbone={self.freeze_backbone}, "
                f"device='{self.device}')")