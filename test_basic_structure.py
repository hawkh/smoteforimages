#!/usr/bin/env python3
"""
Basic test to check if the SMOTE Image Synthesis modules can be imported
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test importing the main modules."""
    print("Testing imports...")
    
    try:
        # Test basic imports
        from smote_image_synthesis.data.models import PipelineConfig
        print("✓ PipelineConfig imported successfully")
        
        from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder
        print("✓ ResNetEncoder imported successfully")
        
        from smote_image_synthesis.decoders.autoencoder_decoder import AutoencoderDecoder
        print("✓ AutoencoderDecoder imported successfully")
        
        from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE
        print("✓ ConstrainedSMOTE imported successfully")
        
        from smote_image_synthesis.quality.assessor import QualityAssessor
        print("✓ QualityAssessor imported successfully")
        
        from smote_image_synthesis.pipeline import SynthesisPipeline
        print("✓ SynthesisPipeline imported successfully")
        
        print("\n🎉 All core modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without heavy dependencies."""
    print("\nTesting basic functionality...")
    
    try:
        # Create a simple configuration
        config = PipelineConfig()
        print(f"✓ Created PipelineConfig: {config.config_name}")
        
        # Test encoder config
        encoder_config = config.encoder_config
        print(f"✓ Encoder config: {encoder_config.architecture}, dim={encoder_config.embedding_dim}")
        
        # Test SMOTE config
        smote_config = config.smote_config
        print(f"✓ SMOTE config: k={smote_config.k_neighbors}, clustering={smote_config.use_clustering}")
        
        print("✓ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("[SEARCH] SMOTE Image Synthesis - Basic Test")
    print("=" * 50)

    success1 = test_imports()
    success2 = test_basic_functionality()

    print("\n" + "=" * 50)
    if success1 and success2:
        print("[SUCCESS] All tests passed! The system structure is intact.")
    else:
        print("[ERROR] Some tests failed, but core structure exists.")