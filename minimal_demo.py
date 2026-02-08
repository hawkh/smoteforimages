#!/usr/bin/env python3
"""
Minimal SMOTE Image Synthesis Demo
"""

import torch
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def minimal_demo():
    """Run a minimal demo of the SMOTE image synthesis pipeline."""
    
    print("=" * 60)
    print("          SMOTE IMAGE SYNTHESIS - MINIMAL DEMO")
    print("=" * 60)
    print()
    
    try:
        # Test basic imports
        print("1. Testing imports...")
        from smote_image_synthesis.data.models import PipelineConfig
        from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder
        from smote_image_synthesis.decoders.autoencoder_decoder import AutoencoderDecoder
        from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE
        print("   ✓ All imports successful")
        
        # Create simple test data
        print("2. Creating test data...")
        device = torch.device('cpu')
        batch_size = 8
        image_size = 32
        
        # Create simple synthetic images with patterns
        test_images = []
        test_labels = []
        
        # Class 0: Mostly zeros
        for _ in range(3):
            img = torch.zeros(3, image_size, image_size)
            img[:, :8, :] = 0.5  # Top stripe
            test_images.append(img)
            test_labels.append(0)
        
        # Class 1: Mostly ones  
        for _ in range(2):
            img = torch.ones(3, image_size, image_size) * 0.5
            img[:, :, :8] = 0.8  # Left stripe
            test_images.append(img)
            test_labels.append(1)
        
        # Class 2: Mixed pattern
        for _ in range(3):
            img = torch.rand(3, image_size, image_size) * 0.3
            img[:, 8:16, 8:16] = 1.0  # Center square
            test_images.append(img)
            test_labels.append(2)
        
        test_images = torch.stack(test_images)
        test_labels = np.array(test_labels)
        
        print(f"   ✓ Created {len(test_images)} test images")
        print(f"   ✓ Class distribution: {np.bincount(test_labels)}")
        
        # Test encoder
        print("3. Testing encoder...")
        encoder = ResNetEncoder(
            architecture='resnet18',
            embedding_dim=32,
            pretrained=False,
            device=device
        )
        
        with torch.no_grad():
            embeddings = encoder.encode(test_images)
        print(f"   ✓ Encoded to shape: {embeddings.shape}")
        
        # Test decoder
        print("4. Testing decoder...")
        decoder = AutoencoderDecoder(
            embedding_dim=32,
            image_shape=(3, image_size, image_size),
            device=device
        )
        
        with torch.no_grad():
            reconstructed = decoder.decode(embeddings[:3])
        print(f"   ✓ Decoded to shape: {reconstructed.shape}")
        
        # Test SMOTE
        print("5. Testing SMOTE...")
        smote = ConstrainedSMOTE(
            k_neighbors=1,
            sampling_strategy='auto',
            random_state=42
        )
        
        # Convert to numpy for SMOTE
        embeddings_np = embeddings.detach().cpu().numpy()
        # ConstrainedSMOTE uses fit/generate_synthetic API, not fit_resample
        smote.fit(embeddings_np, test_labels)
        synthetic_embeddings, synthetic_labels = smote.generate_synthetic()
        
        # Combine original and synthetic for demo purposes
        if len(synthetic_embeddings) > 0:
            X_resampled = np.vstack([embeddings_np, synthetic_embeddings])
            y_resampled = np.concatenate([test_labels, synthetic_labels])
        else:
            X_resampled = embeddings_np
            y_resampled = test_labels

        print(f"   ✓ Original samples: {len(embeddings_np)}")
        print(f"   ✓ Resampled samples: {len(X_resampled)}")
        print(f"   ✓ Original distribution: {np.bincount(test_labels)}")
        print(f"   ✓ Resampled distribution: {np.bincount(y_resampled)}")
        
        # Generate synthetic images from new embeddings
        print("6. Generating synthetic images...")
        n_new = len(X_resampled) - len(embeddings_np)
        if n_new > 0:
            new_embeddings = torch.tensor(X_resampled[-n_new:], dtype=torch.float32)
            
            with torch.no_grad():
                synthetic_images = decoder.decode(new_embeddings)
            
            print(f"   ✓ Generated {len(synthetic_images)} synthetic images")
            print(f"   ✓ Synthetic image shape: {synthetic_images.shape}")
        else:
            print("   ! No new images generated (dataset already balanced)")
        
        # Simple quality check
        print("7. Basic quality check...")
        
        # Compute basic statistics
        original_mean = test_images.mean().item()
        if n_new > 0:
            synthetic_mean = synthetic_images.mean().item()
            print(f"   ✓ Original images mean: {original_mean:.4f}")
            print(f"   ✓ Synthetic images mean: {synthetic_mean:.4f}")
            print(f"   ✓ Mean difference: {abs(original_mean - synthetic_mean):.4f}")
        else:
            print(f"   ✓ Original images mean: {original_mean:.4f}")
        
        print()
        print("=" * 60)
        print("✅ MINIMAL DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("The SMOTE Image Synthesis pipeline is working correctly.")
        print("Key components tested:")
        print("  • Image encoding (ResNet)")
        print("  • SMOTE oversampling") 
        print("  • Image decoding (Autoencoder)")
        print("  • Synthetic image generation")
        print()
        
        if n_new > 0:
            print(f"Generated {n_new} new synthetic images to balance the dataset.")
        print("You can now run the full demo with more features:")
        print("  python demo_pipeline.py --n-samples 50 --cpu")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = minimal_demo()
    if not success:
        exit(1)