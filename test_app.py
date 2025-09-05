#!/usr/bin/env python3
"""
Simple test script for SMOTE Image Synthesis application.
"""

import torch
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required imports work."""
    logger.info("Testing imports...")
    
    try:
        from smote_image_synthesis.data.models import PipelineConfig
        from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder
        from smote_image_synthesis.decoders.autoencoder_decoder import AutoencoderDecoder
        from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE
        logger.info("✓ All imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_basic_pipeline():
    """Test basic pipeline functionality."""
    logger.info("Testing basic pipeline setup...")
    
    try:
        # Set device
        device = torch.device('cpu')  # Use CPU for testing
        
        # Create simple configuration
        config = PipelineConfig(
            config_name="test_config",
            output_dir="./test_output",
            seed=42
        )
        
        # Adjust for simple test
        config.encoder_config.architecture = 'resnet18'
        config.encoder_config.embedding_dim = 64
        config.encoder_config.pretrained = False
        config.decoder_config.image_shape = (3, 32, 32)
        
        logger.info("✓ Configuration created")
        
        # Create small test data
        batch_size = 4
        test_images = torch.rand(batch_size, 3, 32, 32)
        test_labels = np.array([0, 0, 1, 1])
        
        logger.info(f"✓ Test data created: {test_images.shape}")
        
        # Test encoder
        encoder = ResNetEncoder(
            architecture=config.encoder_config.architecture,
            embedding_dim=config.encoder_config.embedding_dim,
            pretrained=False,
            device=device
        )
        
        embeddings = encoder.encode(test_images)
        logger.info(f"✓ Encoder working: {embeddings.shape}")
        
        # Test decoder
        decoder = AutoencoderDecoder(
            embedding_dim=config.encoder_config.embedding_dim,
            image_shape=config.decoder_config.image_shape,
            device=device
        )
        
        reconstructed = decoder.decode(embeddings[:2])
        logger.info(f"✓ Decoder working: {reconstructed.shape}")
        
        # Test SMOTE
        smote = ConstrainedSMOTE(
            k_neighbors=2,  # Small for test
            sampling_strategy='auto',
            random_state=42
        )
        
        # Fit and transform
        embeddings_np = embeddings.detach().cpu().numpy()
        X_resampled, y_resampled = smote.fit_resample(embeddings_np, test_labels)
        
        logger.info(f"✓ SMOTE working: {X_resampled.shape} samples generated")
        logger.info(f"  Original distribution: {np.bincount(test_labels)}")
        logger.info(f"  Resampled distribution: {np.bincount(y_resampled)}")
        
        logger.info("✓ Basic pipeline test successful!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    logger.info("Starting SMOTE Image Synthesis Application Test")
    logger.info("=" * 50)
    
    # Test imports
    if not test_imports():
        logger.error("Import test failed - cannot proceed")
        return 1
    
    # Test basic pipeline
    if not test_basic_pipeline():
        logger.error("Pipeline test failed")
        return 1
    
    logger.info("=" * 50)
    logger.info("✓ All tests passed! The application is working correctly.")
    logger.info("You can now run the full demo with:")
    logger.info("  python demo_pipeline.py --n-samples 30 --cpu")
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())