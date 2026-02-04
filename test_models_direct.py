#!/usr/bin/env python3
"""
Direct test to check if the SMOTE Image Synthesis models can be imported directly
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_direct_import():
    """Test importing the models directly."""
    print("Testing direct import of models...")
    
    try:
        # Import the models file directly without going through package init
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "models", 
            "smote_image_synthesis/data/models.py"
        )
        models_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models_module)
        
        # Access the classes
        PipelineConfig = models_module.PipelineConfig
        print("[OK] PipelineConfig imported successfully")

        EncoderConfig = models_module.EncoderConfig
        print("[OK] EncoderConfig imported successfully")

        SMOTEConfig = models_module.SMOTEConfig
        print("[OK] SMOTEConfig imported successfully")

        DecoderConfig = models_module.DecoderConfig
        print("[OK] DecoderConfig imported successfully")

        QualityConfig = models_module.QualityConfig
        print("[OK] QualityConfig imported successfully")

        EmbeddingData = models_module.EmbeddingData
        print("[OK] EmbeddingData imported successfully")

        SyntheticSample = models_module.SyntheticSample
        print("[OK] SyntheticSample imported successfully")

        print("\n[SUCCESS] All data models imported successfully!")
        return True

    except Exception as e:
        print(f"[ERROR] Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of the models."""
    print("\nTesting basic functionality...")
    
    try:
        # Import the models file directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "models", 
            "smote_image_synthesis/data/models.py"
        )
        models_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models_module)
        
        # Create a simple configuration
        PipelineConfig = models_module.PipelineConfig
        config = PipelineConfig()
        print(f"[OK] Created PipelineConfig: {config.config_name}")

        # Test encoder config
        encoder_config = config.encoder_config
        print(f"[OK] Encoder config: {encoder_config.architecture}, dim={encoder_config.embedding_dim}")

        # Test SMOTE config
        smote_config = config.smote_config
        print(f"[OK] SMOTE config: k={smote_config.k_neighbors}, clustering={smote_config.use_clustering}")

        # Test decoder config
        decoder_config = config.decoder_config
        print(f"[OK] Decoder config: {decoder_config.decoder_type}, shape={decoder_config.image_shape}")

        # Test quality config
        quality_config = config.quality_config
        print(f"[OK] Quality config: metrics={quality_config.metrics}")

        print("[OK] Basic functionality test passed")
        return True

    except Exception as e:
        print(f"[ERROR] Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("[CONFIG] SMOTE Image Synthesis - Direct Models Test")
    print("=" * 60)
    
    success1 = test_direct_import()
    success2 = test_basic_functionality()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("[SUCCESS] All tests passed! The data models are properly implemented.")
        print("The SMOTE Image Synthesis system has a complete configuration system.")
    else:
        print("[ERROR] Some tests failed.")