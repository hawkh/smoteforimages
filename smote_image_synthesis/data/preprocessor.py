"""
Image preprocessing utilities for the SMOTE image synthesis pipeline.
"""

import os
from typing import List, Tuple, Optional, Union, Dict, Any
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image, ImageOps
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Image preprocessing utilities for loading, resizing, normalizing, and augmenting images.
    
    Handles batch processing and provides configurable preprocessing pipelines
    for training and inference.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        augmentation_enabled: bool = False,
        augmentation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the image preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            normalize_mean: Mean values for normalization (ImageNet defaults)
            normalize_std: Standard deviation values for normalization (ImageNet defaults)
            augmentation_enabled: Whether to apply data augmentation
            augmentation_config: Configuration for data augmentation options
        """
        self.target_size = target_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.augmentation_enabled = augmentation_enabled
        self.augmentation_config = augmentation_config or {}
        
        # Build preprocessing transforms
        self._build_transforms()
        
    def _build_transforms(self) -> None:
        """Build the preprocessing transform pipeline."""
        # Base transforms (always applied)
        base_transforms = [
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ]
        
        # Augmentation transforms (optional)
        augmentation_transforms = []
        if self.augmentation_enabled:
            augmentation_transforms = self._build_augmentation_transforms()
        
        # Combine transforms
        if augmentation_transforms:
            # For training: augmentation + base transforms
            self.train_transform = transforms.Compose(
                augmentation_transforms + base_transforms
            )
            # For inference: only base transforms
            self.inference_transform = transforms.Compose(base_transforms)
        else:
            # Same transform for both training and inference
            self.train_transform = transforms.Compose(base_transforms)
            self.inference_transform = self.train_transform
            
    def _build_augmentation_transforms(self) -> List[Any]:
        """Build data augmentation transforms based on configuration."""
        augmentation_transforms = []
        
        # Random horizontal flip
        if self.augmentation_config.get('horizontal_flip', True):
            flip_prob = self.augmentation_config.get('horizontal_flip_prob', 0.5)
            augmentation_transforms.append(transforms.RandomHorizontalFlip(p=flip_prob))
        
        # Random rotation
        if self.augmentation_config.get('rotation', False):
            rotation_degrees = self.augmentation_config.get('rotation_degrees', 10)
            augmentation_transforms.append(transforms.RandomRotation(rotation_degrees))
        
        # Random crop and resize
        if self.augmentation_config.get('random_crop', False):
            crop_scale = self.augmentation_config.get('crop_scale', (0.8, 1.0))
            crop_ratio = self.augmentation_config.get('crop_ratio', (0.75, 1.33))
            augmentation_transforms.append(
                transforms.RandomResizedCrop(
                    self.target_size, 
                    scale=crop_scale, 
                    ratio=crop_ratio
                )
            )
        
        # Color jitter
        if self.augmentation_config.get('color_jitter', False):
            brightness = self.augmentation_config.get('brightness', 0.2)
            contrast = self.augmentation_config.get('contrast', 0.2)
            saturation = self.augmentation_config.get('saturation', 0.2)
            hue = self.augmentation_config.get('hue', 0.1)
            augmentation_transforms.append(
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue
                )
            )
        
        return augmentation_transforms
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load an image from file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format is not supported
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return image
            
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {str(e)}")
    
    def validate_image_format(self, image_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Validate if an image file format is supported.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        image_path = Path(image_path)
        
        # Check if file exists
        if not image_path.exists():
            return False, f"File not found: {image_path}"
        
        # Check file extension
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        if image_path.suffix.lower() not in supported_extensions:
            return False, f"Unsupported file format: {image_path.suffix}"
        
        # Try to open the image
        try:
            with Image.open(image_path) as img:
                # Check if image can be loaded
                img.verify()
            return True, "Valid image format"
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
    
    def preprocess_image(
        self, 
        image: Union[Image.Image, str, Path], 
        training: bool = False
    ) -> torch.Tensor:
        """
        Preprocess a single image.
        
        Args:
            image: PIL Image object or path to image file
            training: Whether to apply training augmentations
            
        Returns:
            Preprocessed image tensor [C, H, W]
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = self.load_image(image)
        
        # Apply appropriate transform
        transform = self.train_transform if training else self.inference_transform
        return transform(image)
    
    def preprocess_batch(
        self, 
        images: List[Union[Image.Image, str, Path]], 
        training: bool = False,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of PIL Images or image paths
            training: Whether to apply training augmentations
            batch_size: Optional batch size for memory management
            
        Returns:
            Batch of preprocessed images [B, C, H, W]
        """
        if not images:
            raise ValueError("Empty image list provided")
        
        # Process images in batches if batch_size is specified
        if batch_size is not None and len(images) > batch_size:
            return self._process_large_batch(images, training, batch_size)
        
        # Process all images at once
        processed_images = []
        for image in images:
            try:
                processed_image = self.preprocess_image(image, training=training)
                processed_images.append(processed_image)
            except Exception as e:
                logger.warning(f"Failed to process image {image}: {str(e)}")
                continue
        
        if not processed_images:
            raise ValueError("No images could be processed successfully")
        
        return torch.stack(processed_images)
    
    def _process_large_batch(
        self, 
        images: List[Union[Image.Image, str, Path]], 
        training: bool, 
        batch_size: int
    ) -> torch.Tensor:
        """
        Process large batches of images with memory management.
        
        Args:
            images: List of images to process
            training: Whether to apply training augmentations
            batch_size: Size of each processing batch
            
        Returns:
            Combined batch tensor
        """
        all_batches = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_tensor = self.preprocess_batch(batch_images, training=training)
            all_batches.append(batch_tensor)
        
        return torch.cat(all_batches, dim=0)
    
    def load_images_from_directory(
        self, 
        directory: Union[str, Path], 
        recursive: bool = False,
        max_images: Optional[int] = None
    ) -> List[Path]:
        """
        Load image paths from a directory.
        
        Args:
            directory: Directory containing images
            recursive: Whether to search subdirectories
            max_images: Maximum number of images to load
            
        Returns:
            List of image file paths
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Supported image extensions
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Find image files
        image_paths = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory.glob(pattern):
            if (file_path.is_file() and 
                file_path.suffix.lower() in supported_extensions):
                image_paths.append(file_path)
                
                # Check max_images limit
                if max_images is not None and len(image_paths) >= max_images:
                    break
        
        return sorted(image_paths)
    
    def get_image_stats(self, images: List[Union[Image.Image, str, Path]]) -> Dict[str, Any]:
        """
        Compute statistics for a collection of images.
        
        Args:
            images: List of images or image paths
            
        Returns:
            Dictionary containing image statistics
        """
        if not images:
            return {}
        
        sizes = []
        modes = []
        
        for image in images:
            try:
                if isinstance(image, (str, Path)):
                    with Image.open(image) as img:
                        sizes.append(img.size)
                        modes.append(img.mode)
                else:
                    sizes.append(image.size)
                    modes.append(image.mode)
            except Exception as e:
                logger.warning(f"Failed to get stats for image {image}: {str(e)}")
                continue
        
        if not sizes:
            return {}
        
        # Calculate statistics
        widths, heights = zip(*sizes)
        
        stats = {
            'count': len(sizes),
            'width': {
                'min': min(widths),
                'max': max(widths),
                'mean': sum(widths) / len(widths)
            },
            'height': {
                'min': min(heights),
                'max': max(heights),
                'mean': sum(heights) / len(heights)
            },
            'modes': list(set(modes)),
            'target_size': self.target_size
        }
        
        return stats
    
    def denormalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a tensor for visualization.
        
        Args:
            tensor: Normalized tensor [C, H, W] or [B, C, H, W]
            
        Returns:
            Denormalized tensor
        """
        # Convert to numpy for easier manipulation
        if tensor.dim() == 4:  # Batch
            # Process each image in batch
            denormalized = []
            for i in range(tensor.shape[0]):
                img = self._denormalize_single(tensor[i])
                denormalized.append(img)
            return torch.stack(denormalized)
        else:  # Single image
            return self._denormalize_single(tensor)
    
    def _denormalize_single(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize a single image tensor."""
        mean = torch.tensor(self.normalize_mean).view(3, 1, 1)
        std = torch.tensor(self.normalize_std).view(3, 1, 1)
        
        # Denormalize: x = (x_norm * std) + mean
        denormalized = tensor * std + mean
        
        # Clamp to valid range [0, 1]
        return torch.clamp(denormalized, 0, 1)
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert a tensor to PIL Image.
        
        Args:
            tensor: Image tensor [C, H, W]
            
        Returns:
            PIL Image
        """
        # Denormalize if needed
        if tensor.min() < 0 or tensor.max() > 1:
            tensor = self.denormalize_tensor(tensor)
        
        # Convert to PIL
        return F.to_pil_image(tensor)