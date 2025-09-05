"""
Unit tests for the base image encoder interface.
"""

import pytest
import tempfile
import json
from pathlib import Path
import torch
import torch.nn as nn

from smote_image_synthesis.encoders.base import ImageEncoder


class MockEncoder(ImageEncoder):
    """Mock encoder implementation for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self._build_model()
        self.model = self.model.to(self.device)
    
    def _build_model(self) -> nn.Module:
        """Build a simple mock model."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, self.embedding_dim)  # Assuming 3 input channels
        )
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using the mock model."""
        if not self.validate_input(images)[0]:
            raise ValueError("Invalid input tensor")
        
        self.model.eval()
        with torch.no_grad():
            return self.model(images.to(self.device))
    
    @classmethod
    def load_from_config(cls, config_path: Path) -> 'MockEncoder':
        """Load mock encoder from config."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        encoder = cls(
            architecture=config_data['architecture'],
            embedding_dim=config_data['embedding_dim'],
            pretrained=config_data['pretrained'],
            config=config_data['config']
        )
        
        # Load the model
        model_path = Path(config_data['model_path'])
        if model_path.exists():
            encoder.load_model(model_path.with_suffix(''))
        
        return encoder


class TestImageEncoder:
    """Test cases for ImageEncoder base class."""
    
    @pytest.fixture
    def mock_encoder(self):
        """Create a mock encoder instance."""
        return MockEncoder(
            architecture='mock_cnn',
            embedding_dim=128,
            pretrained=False
        )
    
    @pytest.fixture
    def sample_images(self):
        """Create sample image tensors."""
        return torch.randn(4, 3, 224, 224)  # Batch of 4 RGB images
    
    def test_initialization_default(self):
        """Test encoder initialization with default parameters."""
        encoder = MockEncoder(
            architecture='test_arch',
            embedding_dim=256
        )
        
        assert encoder.architecture == 'test_arch'
        assert encoder.embedding_dim == 256
        assert encoder.pretrained is True  # Default
        assert encoder.device.type in ['cpu', 'cuda']
        assert encoder.config == {}
        assert encoder.model is not None
    
    def test_initialization_custom(self):
        """Test encoder initialization with custom parameters."""
        config = {'param1': 'value1', 'param2': 42}
        device = torch.device('cpu')
        
        encoder = MockEncoder(
            architecture='custom_arch',
            embedding_dim=512,
            pretrained=False,
            device=device,
            config=config
        )
        
        assert encoder.architecture == 'custom_arch'
        assert encoder.embedding_dim == 512
        assert encoder.pretrained is False
        assert encoder.device == device
        assert encoder.config == config
    
    def test_invalid_embedding_dim(self):
        """Test initialization with invalid embedding dimension."""
        with pytest.raises(ValueError, match="Embedding dimension must be a positive integer"):
            MockEncoder(architecture='test', embedding_dim=0)
        
        with pytest.raises(ValueError, match="Embedding dimension must be a positive integer"):
            MockEncoder(architecture='test', embedding_dim=-10)
    
    def test_large_embedding_dim_warning(self, caplog):
        """Test warning for large embedding dimensions."""
        MockEncoder(architecture='test', embedding_dim=5000)
        assert "Large embedding dimension" in caplog.text
    
    def test_encode_success(self, mock_encoder, sample_images):
        """Test successful encoding."""
        embeddings = mock_encoder.encode(sample_images)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (4, 128)  # [batch_size, embedding_dim]
        assert embeddings.dtype == torch.float32
    
    def test_encode_batch_success(self, mock_encoder):
        """Test successful batch encoding."""
        batch1 = torch.randn(2, 3, 224, 224)
        batch2 = torch.randn(3, 3, 224, 224)
        image_batch = [batch1, batch2]
        
        embeddings = mock_encoder.encode_batch(image_batch)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (5, 128)  # Total 5 images
    
    def test_encode_batch_empty(self, mock_encoder):
        """Test encoding empty batch."""
        with pytest.raises(ValueError, match="Empty image batch provided"):
            mock_encoder.encode_batch([])
    
    def test_encode_batch_with_batch_size(self, mock_encoder):
        """Test batch encoding with specified batch size."""
        batches = [torch.randn(2, 3, 224, 224) for _ in range(5)]
        
        embeddings = mock_encoder.encode_batch(batches, batch_size=2)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (10, 128)  # 5 batches * 2 images each
    
    def test_get_embedding_dim(self, mock_encoder):
        """Test getting embedding dimension."""
        assert mock_encoder.get_embedding_dim() == 128
    
    def test_get_architecture(self, mock_encoder):
        """Test getting architecture name."""
        assert mock_encoder.get_architecture() == 'mock_cnn'
    
    def test_get_device(self, mock_encoder):
        """Test getting device."""
        device = mock_encoder.get_device()
        assert isinstance(device, torch.device)
    
    def test_to_device(self, mock_encoder):
        """Test moving encoder to different device."""
        original_device = mock_encoder.device
        new_device = torch.device('cpu')
        
        result = mock_encoder.to_device(new_device)
        
        assert result is mock_encoder  # Should return self
        assert mock_encoder.device == new_device
    
    def test_validate_input_success(self, mock_encoder, sample_images):
        """Test successful input validation."""
        is_valid, message = mock_encoder.validate_input(sample_images)
        
        assert is_valid
        assert "Valid input" in message
    
    def test_validate_input_wrong_type(self, mock_encoder):
        """Test input validation with wrong type."""
        is_valid, message = mock_encoder.validate_input("not a tensor")
        
        assert not is_valid
        assert "Expected torch.Tensor" in message
    
    def test_validate_input_wrong_dimensions(self, mock_encoder):
        """Test input validation with wrong dimensions."""
        # 3D tensor instead of 4D
        wrong_tensor = torch.randn(3, 224, 224)
        is_valid, message = mock_encoder.validate_input(wrong_tensor)
        
        assert not is_valid
        assert "Expected 4D tensor" in message
    
    def test_validate_input_wrong_channels(self, mock_encoder):
        """Test input validation with wrong number of channels."""
        # 4 channels instead of 1 or 3
        wrong_tensor = torch.randn(2, 4, 224, 224)
        is_valid, message = mock_encoder.validate_input(wrong_tensor)
        
        assert not is_valid
        assert "Expected 1 or 3 channels" in message
    
    def test_validate_input_zero_batch_size(self, mock_encoder):
        """Test input validation with zero batch size."""
        wrong_tensor = torch.randn(0, 3, 224, 224)
        is_valid, message = mock_encoder.validate_input(wrong_tensor)
        
        assert not is_valid
        assert "Batch size cannot be zero" in message
    
    def test_validate_input_nan_values(self, mock_encoder):
        """Test input validation with NaN values."""
        nan_tensor = torch.randn(2, 3, 224, 224)
        nan_tensor[0, 0, 0, 0] = float('nan')
        
        is_valid, message = mock_encoder.validate_input(nan_tensor)
        
        assert not is_valid
        assert "NaN values" in message
    
    def test_validate_input_inf_values(self, mock_encoder):
        """Test input validation with infinite values."""
        inf_tensor = torch.randn(2, 3, 224, 224)
        inf_tensor[0, 0, 0, 0] = float('inf')
        
        is_valid, message = mock_encoder.validate_input(inf_tensor)
        
        assert not is_valid
        assert "infinite values" in message
    
    def test_validate_input_format_success(self, mock_encoder):
        """Test successful image format validation."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(b'fake image data')
            tmp.flush()
            
            is_valid, message = mock_encoder.validate_input_format(tmp.name)
            
            assert is_valid
            assert "Valid image format" in message
        
        Path(tmp.name).unlink()
    
    def test_validate_input_format_file_not_found(self, mock_encoder):
        """Test image format validation with non-existent file."""
        is_valid, message = mock_encoder.validate_input_format('non_existent.jpg')
        
        assert not is_valid
        assert "File not found" in message
    
    def test_validate_input_format_unsupported_extension(self, mock_encoder):
        """Test image format validation with unsupported extension."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b'not an image')
            tmp.flush()
            
            is_valid, message = mock_encoder.validate_input_format(tmp.name)
            
            assert not is_valid
            assert "Unsupported file format" in message
        
        Path(tmp.name).unlink()
    
    def test_save_and_load_model(self, mock_encoder):
        """Test saving and loading model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / 'test_model'
            
            # Save model
            mock_encoder.save_model(save_path)
            
            # Check files exist
            assert (save_path.with_suffix('.pth')).exists()
            assert (save_path.with_suffix('.json')).exists()
            
            # Create new encoder and load
            new_encoder = MockEncoder(
                architecture='mock_cnn',
                embedding_dim=128,
                pretrained=False
            )
            new_encoder.load_model(save_path)
            
            # Verify loaded correctly
            assert new_encoder.architecture == mock_encoder.architecture
            assert new_encoder.embedding_dim == mock_encoder.embedding_dim
    
    def test_load_model_file_not_found(self, mock_encoder):
        """Test loading non-existent model."""
        with pytest.raises(FileNotFoundError):
            mock_encoder.load_model('non_existent_model')
    
    def test_load_model_architecture_mismatch(self, mock_encoder):
        """Test loading model with mismatched architecture."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / 'test_model'
            
            # Save model
            mock_encoder.save_model(save_path)
            
            # Try to load with different architecture
            different_encoder = MockEncoder(
                architecture='different_arch',
                embedding_dim=128,
                pretrained=False
            )
            
            with pytest.raises(ValueError, match="Architecture mismatch"):
                different_encoder.load_model(save_path)
    
    def test_load_model_embedding_dim_mismatch(self, mock_encoder):
        """Test loading model with mismatched embedding dimension."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / 'test_model'
            
            # Save model
            mock_encoder.save_model(save_path)
            
            # Try to load with different embedding dimension
            different_encoder = MockEncoder(
                architecture='mock_cnn',
                embedding_dim=256,  # Different dimension
                pretrained=False
            )
            
            with pytest.raises(ValueError, match="Embedding dimension mismatch"):
                different_encoder.load_model(save_path)
    
    def test_get_model_info(self, mock_encoder):
        """Test getting model information."""
        info = mock_encoder.get_model_info()
        
        assert info['architecture'] == 'mock_cnn'
        assert info['embedding_dim'] == 128
        assert info['pretrained'] is False
        assert 'device' in info
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert 'model_size_mb' in info
    
    def test_training_mode_setting(self, mock_encoder):
        """Test setting training/evaluation mode."""
        # Test training mode
        mock_encoder.train()
        assert mock_encoder.model.training
        
        # Test evaluation mode
        mock_encoder.eval()
        assert not mock_encoder.model.training
        
        # Test explicit setting
        mock_encoder.set_training_mode(True)
        assert mock_encoder.model.training
        
        mock_encoder.set_training_mode(False)
        assert not mock_encoder.model.training
    
    def test_string_representation(self, mock_encoder):
        """Test string representation methods."""
        repr_str = repr(mock_encoder)
        str_str = str(mock_encoder)
        
        assert 'MockEncoder' in repr_str
        assert 'mock_cnn' in repr_str
        assert '128' in repr_str
        assert repr_str == str_str
    
    def test_save_model_no_model(self):
        """Test saving when no model is built."""
        encoder = MockEncoder.__new__(MockEncoder)  # Create without calling __init__
        encoder.model = None
        
        with pytest.raises(ValueError, match="No model to save"):
            encoder.save_model('test_path')
    
    def test_load_from_config_not_implemented(self):
        """Test that base class load_from_config raises NotImplementedError."""
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as tmp:
            json.dump({'test': 'config'}, tmp)
            tmp.flush()
            
            with pytest.raises(NotImplementedError):
                ImageEncoder.load_from_config(tmp.name)
        
        Path(tmp.name).unlink()


if __name__ == '__main__':
    pytest.main([__file__])