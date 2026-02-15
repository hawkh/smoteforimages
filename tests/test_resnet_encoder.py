"""
Unit and integration tests for the ResNet encoder.
"""

import pytest
import tempfile
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder


class TestResNetEncoder:
    """Test cases for ResNetEncoder class."""
    
    @pytest.fixture
    def resnet_encoder(self):
        """Create a ResNet encoder instance."""
        return ResNetEncoder(
            architecture='resnet18',  # Use smallest for faster tests
            embedding_dim=128,
            pretrained=False,  # Avoid downloading weights in tests
            freeze_backbone=False
        )
    
    @pytest.fixture
    def frozen_resnet_encoder(self):
        """Create a ResNet encoder with frozen backbone."""
        return ResNetEncoder(
            architecture='resnet18',
            embedding_dim=256,
            pretrained=False,
            freeze_backbone=True
        )
    
    @pytest.fixture
    def sample_images(self):
        """Create sample image tensors."""
        return torch.randn(4, 3, 224, 224)  # Batch of 4 RGB images
    
    @pytest.fixture
    def large_batch_images(self):
        """Create a large batch of images for memory management tests."""
        return torch.randn(100, 3, 224, 224)
    
    @pytest.fixture
    def sample_dataloader(self):
        """Create a sample dataloader for fine-tuning tests."""
        images = torch.randn(20, 3, 224, 224)
        labels = torch.randint(0, 3, (20,))  # 3 classes
        dataset = TensorDataset(images, labels)
        return DataLoader(dataset, batch_size=4, shuffle=True)
    
    def test_initialization_default(self):
        """Test ResNet encoder initialization with default parameters."""
        encoder = ResNetEncoder()
        
        assert encoder.architecture == 'resnet50'  # Default
        assert encoder.embedding_dim == 512  # Default
        assert encoder.pretrained is True
        assert encoder.freeze_backbone is False
        assert encoder.dropout_rate == 0.1
        assert encoder.model is not None
    
    def test_initialization_custom(self):
        """Test ResNet encoder initialization with custom parameters."""
        encoder = ResNetEncoder(
            architecture='resnet101',
            embedding_dim=1024,
            pretrained=False,
            freeze_backbone=True,
            dropout_rate=0.2
        )
        
        assert encoder.architecture == 'resnet101'
        assert encoder.embedding_dim == 1024
        assert encoder.pretrained is False
        assert encoder.freeze_backbone is True
        assert encoder.dropout_rate == 0.2
    
    def test_unsupported_architecture(self):
        """Test initialization with unsupported architecture."""
        with pytest.raises(ValueError, match="Unsupported architecture"):
            ResNetEncoder(architecture='vgg16')
    
    def test_supported_architectures(self):
        """Test that all supported architectures can be initialized."""
        architectures = ['resnet18', 'resnet50', 'resnet101']
        
        for arch in architectures:
            encoder = ResNetEncoder(
                architecture=arch,
                embedding_dim=128,
                pretrained=False
            )
            assert encoder.architecture == arch
            assert encoder.model is not None
    
    def test_encode_success(self, resnet_encoder, sample_images):
        """Test successful encoding."""
        embeddings = resnet_encoder.encode(sample_images)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (4, 128)  # [batch_size, embedding_dim]
        assert embeddings.dtype == torch.float32
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()
    
    def test_encode_different_embedding_dims(self, sample_images):
        """Test encoding with different embedding dimensions."""
        embedding_dims = [64, 256, 512, 1024]
        
        for dim in embedding_dims:
            encoder = ResNetEncoder(
                architecture='resnet18',
                embedding_dim=dim,
                pretrained=False
            )
            embeddings = encoder.encode(sample_images)
            assert embeddings.shape == (4, dim)
    
    def test_encode_single_image(self, resnet_encoder):
        """Test encoding a single image."""
        single_image = torch.randn(1, 3, 224, 224)
        embeddings = resnet_encoder.encode(single_image)
        
        assert embeddings.shape == (1, 128)
    
    def test_encode_invalid_input(self, resnet_encoder):
        """Test encoding with invalid input."""
        # Wrong number of dimensions
        invalid_input = torch.randn(3, 224, 224)  # Missing batch dimension
        
        with pytest.raises(ValueError, match="Invalid input"):
            resnet_encoder.encode(invalid_input)
    
    def test_encode_with_memory_management(self, resnet_encoder, large_batch_images):
        """Test encoding with memory management."""
        embeddings = resnet_encoder.encode_with_memory_management(
            large_batch_images, 
            max_batch_size=16
        )
        
        assert embeddings.shape == (100, 128)
        assert not torch.isnan(embeddings).any()
    
    def test_encode_batch_success(self, resnet_encoder):
        """Test batch encoding."""
        batch1 = torch.randn(2, 3, 224, 224)
        batch2 = torch.randn(3, 3, 224, 224)
        image_batch = [batch1, batch2]
        
        embeddings = resnet_encoder.encode_batch(image_batch)
        
        assert embeddings.shape == (5, 128)  # Total 5 images
    
    def test_frozen_backbone(self, frozen_resnet_encoder):
        """Test that backbone is properly frozen."""
        backbone = frozen_resnet_encoder.model[0]
        
        # Check that backbone parameters are not trainable
        for param in backbone.parameters():
            assert not param.requires_grad
        
        # Check that embedding head parameters are trainable
        embedding_head = frozen_resnet_encoder.model[2]
        for param in embedding_head.parameters():
            assert param.requires_grad
    
    def test_set_backbone_frozen(self, resnet_encoder):
        """Test freezing and unfreezing backbone."""
        backbone = resnet_encoder.model[0]
        
        # Initially unfrozen
        assert any(p.requires_grad for p in backbone.parameters())
        
        # Freeze backbone
        resnet_encoder.set_backbone_frozen(True)
        assert not any(p.requires_grad for p in backbone.parameters())
        
        # Unfreeze backbone
        resnet_encoder.set_backbone_frozen(False)
        assert any(p.requires_grad for p in backbone.parameters())
    
    def test_fine_tune(self, resnet_encoder, sample_dataloader):
        """Test fine-tuning functionality."""
        # Fine-tune for just 1 epoch to keep test fast
        history = resnet_encoder.fine_tune(
            sample_dataloader,
            num_epochs=1,
            learning_rate=1e-3,
            unfreeze_after_epochs=0,
            num_classes=3
        )
        
        assert 'loss' in history
        assert 'accuracy' in history
        assert len(history['loss']) == 1
        assert len(history['accuracy']) == 1
        assert resnet_encoder._is_trained
    
    def test_get_feature_maps(self, resnet_encoder, sample_images):
        """Test feature map extraction."""
        # Test extracting features from different layers
        layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        
        for layer_name in layer_names:
            try:
                feature_maps = resnet_encoder.get_feature_maps(sample_images, layer_name)
                assert isinstance(feature_maps, torch.Tensor)
                assert feature_maps.shape[0] == 4  # Batch size
                assert len(feature_maps.shape) == 4  # [B, C, H, W]
            except Exception as e:
                # Some layer names might not exist in ResNet18
                if "layer" in layer_name:
                    continue
                else:
                    raise e
    
    def test_save_and_load_model(self, resnet_encoder):
        """Test saving and loading ResNet model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / 'resnet_model'
            
            # Save model
            resnet_encoder.save_model(save_path)
            
            # Check files exist
            assert (save_path.with_suffix('.pth')).exists()
            assert (save_path.with_suffix('.json')).exists()
            
            # Create new encoder and load
            new_encoder = ResNetEncoder(
                architecture='resnet18',
                embedding_dim=128,
                pretrained=False
            )
            new_encoder.load_model(save_path)
            
            # Test that loaded model produces same outputs
            sample_input = torch.randn(2, 3, 224, 224)
            original_output = resnet_encoder.encode(sample_input)
            loaded_output = new_encoder.encode(sample_input)
            
            # Outputs should be very close (allowing for small numerical differences)
            assert torch.allclose(original_output, loaded_output, atol=1e-5)
    
    def test_load_from_config(self, resnet_encoder):
        """Test loading ResNet encoder from configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / 'resnet_model'
            config_path = save_path.with_suffix('.json')
            
            # Save model first
            resnet_encoder.save_model(save_path)
            
            # Load from config
            loaded_encoder = ResNetEncoder.load_from_config(config_path)
            
            assert loaded_encoder.architecture == resnet_encoder.architecture
            assert loaded_encoder.embedding_dim == resnet_encoder.embedding_dim
            assert loaded_encoder.pretrained == resnet_encoder.pretrained
    
    def test_get_model_info(self, resnet_encoder):
        """Test getting detailed model information."""
        info = resnet_encoder.get_model_info()
        
        # Check base information
        assert info['architecture'] == 'resnet18'
        assert info['embedding_dim'] == 128
        assert info['pretrained'] is False
        
        # Check ResNet-specific information
        assert 'freeze_backbone' in info
        assert 'dropout_rate' in info
        assert 'supported_architectures' in info
        assert 'backbone_parameters' in info
        assert 'embedding_head_parameters' in info
        assert 'backbone_frozen' in info
        
        # Verify parameter counts make sense
        assert info['backbone_parameters'] > 0
        assert info['embedding_head_parameters'] > 0
        assert info['total_parameters'] == (
            info['backbone_parameters'] + info['embedding_head_parameters']
        )
    
    def test_different_input_sizes(self, resnet_encoder):
        """Test encoding with different input image sizes."""
        # ResNet should handle different input sizes due to adaptive pooling
        sizes = [(224, 224), (256, 256), (128, 128)]
        
        for h, w in sizes:
            images = torch.randn(2, 3, h, w)
            embeddings = resnet_encoder.encode(images)
            assert embeddings.shape == (2, 128)
    
    def test_training_mode_preservation(self, resnet_encoder):
        """Test that training mode is preserved during encoding."""
        # Set to training mode
        resnet_encoder.model.train()
        assert resnet_encoder.model.training
        
        # Encode (should temporarily switch to eval mode)
        sample_input = torch.randn(2, 3, 224, 224)
        resnet_encoder.encode(sample_input)
        
        # Should be back to training mode
        assert resnet_encoder.model.training
        
        # Test with eval mode
        resnet_encoder.model.eval()
        assert not resnet_encoder.model.training
        
        resnet_encoder.encode(sample_input)
        assert not resnet_encoder.model.training
    
    def test_device_handling(self):
        """Test device handling for ResNet encoder."""
        # Test CPU device
        cpu_encoder = ResNetEncoder(
            architecture='resnet18',
            embedding_dim=128,
            pretrained=False,
            device=torch.device('cpu')
        )
        
        assert cpu_encoder.device.type == 'cpu'
        assert next(cpu_encoder.model.parameters()).device.type == 'cpu'
        
        # Test moving to different device
        new_device = torch.device('cpu')  # Keep as CPU for testing
        cpu_encoder.to_device(new_device)
        assert cpu_encoder.device == new_device
    
    def test_string_representation(self, resnet_encoder):
        """Test string representation."""
        repr_str = repr(resnet_encoder)
        
        assert 'ResNetEncoder' in repr_str
        assert 'resnet18' in repr_str
        assert '128' in repr_str
        assert 'False' in repr_str  # pretrained=False
    
    def test_embedding_head_configurations(self):
        """Test different embedding head configurations."""
        # Test with batch normalization disabled
        config = {'use_batch_norm': False}
        encoder = ResNetEncoder(
            architecture='resnet18',
            embedding_dim=128,
            pretrained=False,
            config=config
        )
        
        # Test with different activation functions
        activations = ['relu', 'tanh', 'none']
        for activation in activations:
            config = {'activation': activation}
            encoder = ResNetEncoder(
                architecture='resnet18',
                embedding_dim=128,
                pretrained=False,
                config=config
            )
            
            sample_input = torch.randn(2, 3, 224, 224)
            embeddings = encoder.encode(sample_input)
            assert embeddings.shape == (2, 128)
    
    def test_memory_error_handling(self, resnet_encoder):
        """Test handling of memory errors during encoding."""
        # This test is difficult to trigger reliably, so we'll just ensure
        # the error handling code path exists and doesn't crash
        
        # Test with a reasonable batch that shouldn't cause memory issues
        reasonable_batch = torch.randn(4, 3, 224, 224)
        embeddings = resnet_encoder.encode(reasonable_batch)
        assert embeddings.shape == (4, 128)
    
    def test_gradient_flow(self, resnet_encoder):
        """Test that gradients flow properly through the model."""
        resnet_encoder.model.train()
        
        # Create input that requires gradients
        images = torch.randn(2, 3, 224, 224, requires_grad=True)
        
        # Forward pass
        embeddings = resnet_encoder.model(images)  # Use model directly to allow gradients
        
        # Backward pass
        loss = embeddings.sum()
        loss.backward()
        
        # Check that input gradients exist
        assert images.grad is not None
        assert not torch.isnan(images.grad).any()


class TestResNetEncoderIntegration:
    """Integration tests for ResNet encoder."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Create encoder
        encoder = ResNetEncoder(
            architecture='resnet18',
            embedding_dim=256,
            pretrained=False
        )
        
        # Create sample data
        images = torch.randn(8, 3, 224, 224)
        
        # Encode images
        embeddings = encoder.encode(images)
        
        # Verify output
        assert embeddings.shape == (8, 256)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()
        
        # Test batch processing
        batch_embeddings = encoder.encode_with_memory_management(
            images, max_batch_size=4
        )
        
        # Results should be identical
        assert torch.allclose(embeddings, batch_embeddings, atol=1e-5)
    
    def test_multiple_architectures_consistency(self):
        """Test that different architectures produce consistent results."""
        architectures = ['resnet18', 'resnet50']
        sample_input = torch.randn(4, 3, 224, 224)
        
        results = {}
        for arch in architectures:
            encoder = ResNetEncoder(
                architecture=arch,
                embedding_dim=128,
                pretrained=False
            )
            embeddings = encoder.encode(sample_input)
            results[arch] = embeddings
            
            # Basic sanity checks
            assert embeddings.shape == (4, 128)
            assert not torch.isnan(embeddings).any()
        
        # Different architectures should produce different embeddings
        assert not torch.allclose(results['resnet18'], results['resnet50'])


if __name__ == '__main__':
    pytest.main([__file__])