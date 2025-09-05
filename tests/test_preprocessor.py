"""
Unit tests for the image preprocessor.
"""

import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image
import torch
import numpy as np

from smote_image_synthesis.data.preprocessor import ImagePreprocessor


class TestImagePreprocessor:
    """Test cases for ImagePreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a basic preprocessor instance."""
        return ImagePreprocessor(
            target_size=(224, 224),
            augmentation_enabled=False
        )
    
    @pytest.fixture
    def augmented_preprocessor(self):
        """Create a preprocessor with augmentation enabled."""
        augmentation_config = {
            'horizontal_flip': True,
            'horizontal_flip_prob': 0.5,
            'rotation': True,
            'rotation_degrees': 10,
            'color_jitter': True,
            'brightness': 0.2,
            'contrast': 0.2
        }
        return ImagePreprocessor(
            target_size=(224, 224),
            augmentation_enabled=True,
            augmentation_config=augmentation_config
        )
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample PIL image for testing."""
        # Create a simple RGB image
        image = Image.new('RGB', (256, 256), color='red')
        return image
    
    @pytest.fixture
    def temp_image_file(self, sample_image):
        """Create a temporary image file."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            sample_image.save(tmp.name)
            yield tmp.name
        os.unlink(tmp.name)
    
    @pytest.fixture
    def temp_directory_with_images(self):
        """Create a temporary directory with sample images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create sample images
            for i in range(5):
                image = Image.new('RGB', (100 + i * 10, 100 + i * 10), 
                                color=(i * 50, i * 30, i * 20))
                image.save(temp_path / f'image_{i}.jpg')
            
            # Create a non-image file
            (temp_path / 'not_an_image.txt').write_text('test')
            
            yield temp_path
    
    def test_initialization_default(self):
        """Test preprocessor initialization with default parameters."""
        preprocessor = ImagePreprocessor()
        
        assert preprocessor.target_size == (224, 224)
        assert preprocessor.normalize_mean == (0.485, 0.456, 0.406)
        assert preprocessor.normalize_std == (0.229, 0.224, 0.225)
        assert not preprocessor.augmentation_enabled
        assert preprocessor.augmentation_config == {}
    
    def test_initialization_custom(self):
        """Test preprocessor initialization with custom parameters."""
        target_size = (256, 256)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        augmentation_config = {'horizontal_flip': True}
        
        preprocessor = ImagePreprocessor(
            target_size=target_size,
            normalize_mean=mean,
            normalize_std=std,
            augmentation_enabled=True,
            augmentation_config=augmentation_config
        )
        
        assert preprocessor.target_size == target_size
        assert preprocessor.normalize_mean == mean
        assert preprocessor.normalize_std == std
        assert preprocessor.augmentation_enabled
        assert preprocessor.augmentation_config == augmentation_config
    
    def test_load_image_success(self, preprocessor, temp_image_file):
        """Test successful image loading."""
        image = preprocessor.load_image(temp_image_file)
        
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
    
    def test_load_image_file_not_found(self, preprocessor):
        """Test loading non-existent image file."""
        with pytest.raises(FileNotFoundError):
            preprocessor.load_image('non_existent_file.jpg')
    
    def test_validate_image_format_valid(self, preprocessor, temp_image_file):
        """Test validation of valid image format."""
        is_valid, message = preprocessor.validate_image_format(temp_image_file)
        
        assert is_valid
        assert "Valid image format" in message
    
    def test_validate_image_format_invalid_extension(self, preprocessor):
        """Test validation of invalid file extension."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b'not an image')
            tmp.flush()
            
            is_valid, message = preprocessor.validate_image_format(tmp.name)
            
            assert not is_valid
            assert "Unsupported file format" in message
            
        os.unlink(tmp.name)
    
    def test_validate_image_format_file_not_found(self, preprocessor):
        """Test validation of non-existent file."""
        is_valid, message = preprocessor.validate_image_format('non_existent.jpg')
        
        assert not is_valid
        assert "File not found" in message
    
    def test_preprocess_image_pil(self, preprocessor, sample_image):
        """Test preprocessing PIL image."""
        tensor = preprocessor.preprocess_image(sample_image, training=False)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)  # [C, H, W]
        assert tensor.dtype == torch.float32
    
    def test_preprocess_image_path(self, preprocessor, temp_image_file):
        """Test preprocessing image from file path."""
        tensor = preprocessor.preprocess_image(temp_image_file, training=False)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32
    
    def test_preprocess_image_training_mode(self, augmented_preprocessor, sample_image):
        """Test preprocessing with training augmentations."""
        tensor = augmented_preprocessor.preprocess_image(sample_image, training=True)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32
    
    def test_preprocess_batch_success(self, preprocessor, sample_image):
        """Test successful batch preprocessing."""
        images = [sample_image] * 3
        batch_tensor = preprocessor.preprocess_batch(images, training=False)
        
        assert isinstance(batch_tensor, torch.Tensor)
        assert batch_tensor.shape == (3, 3, 224, 224)  # [B, C, H, W]
        assert batch_tensor.dtype == torch.float32
    
    def test_preprocess_batch_empty_list(self, preprocessor):
        """Test batch preprocessing with empty list."""
        with pytest.raises(ValueError, match="Empty image list provided"):
            preprocessor.preprocess_batch([])
    
    def test_preprocess_batch_with_batch_size(self, preprocessor, sample_image):
        """Test batch preprocessing with specified batch size."""
        images = [sample_image] * 5
        batch_tensor = preprocessor.preprocess_batch(
            images, training=False, batch_size=2
        )
        
        assert isinstance(batch_tensor, torch.Tensor)
        assert batch_tensor.shape == (5, 3, 224, 224)
        assert batch_tensor.dtype == torch.float32
    
    def test_load_images_from_directory(self, preprocessor, temp_directory_with_images):
        """Test loading images from directory."""
        image_paths = preprocessor.load_images_from_directory(temp_directory_with_images)
        
        assert len(image_paths) == 5  # Should find 5 image files
        assert all(path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'] 
                  for path in image_paths)
        assert all(path.exists() for path in image_paths)
    
    def test_load_images_from_directory_max_images(self, preprocessor, temp_directory_with_images):
        """Test loading limited number of images from directory."""
        image_paths = preprocessor.load_images_from_directory(
            temp_directory_with_images, max_images=3
        )
        
        assert len(image_paths) == 3
    
    def test_load_images_from_directory_not_found(self, preprocessor):
        """Test loading from non-existent directory."""
        with pytest.raises(FileNotFoundError):
            preprocessor.load_images_from_directory('non_existent_directory')
    
    def test_get_image_stats(self, preprocessor, sample_image):
        """Test computing image statistics."""
        # Create images with different sizes
        images = [
            Image.new('RGB', (100, 100), color='red'),
            Image.new('RGB', (200, 150), color='green'),
            Image.new('RGB', (300, 200), color='blue')
        ]
        
        stats = preprocessor.get_image_stats(images)
        
        assert stats['count'] == 3
        assert stats['width']['min'] == 100
        assert stats['width']['max'] == 300
        assert stats['height']['min'] == 100
        assert stats['height']['max'] == 200
        assert 'RGB' in stats['modes']
        assert stats['target_size'] == (224, 224)
    
    def test_get_image_stats_empty_list(self, preprocessor):
        """Test computing stats for empty image list."""
        stats = preprocessor.get_image_stats([])
        assert stats == {}
    
    def test_denormalize_tensor_single_image(self, preprocessor):
        """Test denormalizing a single image tensor."""
        # Create a normalized tensor
        tensor = torch.randn(3, 224, 224)
        denormalized = preprocessor.denormalize_tensor(tensor)
        
        assert denormalized.shape == (3, 224, 224)
        assert denormalized.min() >= 0
        assert denormalized.max() <= 1
    
    def test_denormalize_tensor_batch(self, preprocessor):
        """Test denormalizing a batch of image tensors."""
        # Create a batch of normalized tensors
        tensor = torch.randn(4, 3, 224, 224)
        denormalized = preprocessor.denormalize_tensor(tensor)
        
        assert denormalized.shape == (4, 3, 224, 224)
        assert denormalized.min() >= 0
        assert denormalized.max() <= 1
    
    def test_tensor_to_pil(self, preprocessor):
        """Test converting tensor to PIL image."""
        # Create a valid tensor (values in [0, 1])
        tensor = torch.rand(3, 224, 224)
        pil_image = preprocessor.tensor_to_pil(tensor)
        
        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (224, 224)
        assert pil_image.mode == 'RGB'
    
    def test_augmentation_transforms_built(self, augmented_preprocessor):
        """Test that augmentation transforms are properly built."""
        # The augmented preprocessor should have different transforms for train/inference
        assert augmented_preprocessor.train_transform is not None
        assert augmented_preprocessor.inference_transform is not None
        
        # Train transform should have more steps than inference
        train_steps = len(augmented_preprocessor.train_transform.transforms)
        inference_steps = len(augmented_preprocessor.inference_transform.transforms)
        assert train_steps > inference_steps
    
    def test_different_target_sizes(self):
        """Test preprocessor with different target sizes."""
        sizes = [(128, 128), (256, 256), (512, 512)]
        
        for size in sizes:
            preprocessor = ImagePreprocessor(target_size=size)
            sample_image = Image.new('RGB', (100, 100), color='red')
            
            tensor = preprocessor.preprocess_image(sample_image)
            assert tensor.shape == (3, size[0], size[1])
    
    def test_grayscale_to_rgb_conversion(self, preprocessor):
        """Test conversion of grayscale images to RGB."""
        # Create a grayscale image
        grayscale_image = Image.new('L', (100, 100), color=128)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            grayscale_image.save(tmp.name)
            
            # Load and check conversion
            loaded_image = preprocessor.load_image(tmp.name)
            assert loaded_image.mode == 'RGB'
            
        os.unlink(tmp.name)


if __name__ == '__main__':
    pytest.main([__file__])