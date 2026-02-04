#!/usr/bin/env python3
"""
Image-to-Image SMOTE Synthesis Tool (Dependency-Free Version)

This script takes input images and generates synthetic images using the SMOTE technique.
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path
from PIL import Image
import argparse
import importlib.util

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_pipeline():
    """Set up the SMOTE image synthesis pipeline."""
    # Import models directly to avoid problematic __init__.py imports
    spec = importlib.util.spec_from_file_location(
        "models", 
        "smote_image_synthesis/data/models.py"
    )
    models_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models_module)
    
    # Import other modules directly
    encoder_spec = importlib.util.spec_from_file_location(
        "resnet_encoder",
        "smote_image_synthesis/encoders/resnet_encoder.py"
    )
    encoder_module = importlib.util.module_from_spec(encoder_spec)
    encoder_spec.loader.exec_module(encoder_module)
    
    decoder_spec = importlib.util.spec_from_file_location(
        "autoencoder_decoder",
        "smote_image_synthesis/decoders/autoencoder_decoder.py"
    )
    decoder_module = importlib.util.module_from_spec(decoder_spec)
    decoder_spec.loader.exec_module(decoder_module)
    
    smote_spec = importlib.util.spec_from_file_location(
        "constrained_smote",
        "smote_image_synthesis/smote/constrained_smote.py"
    )
    smote_module = importlib.util.module_from_spec(smote_spec)
    smote_spec.loader.exec_module(smote_module)
    
    quality_spec = importlib.util.spec_from_file_location(
        "assessor",
        "smote_image_synthesis/quality/assessor.py"
    )
    quality_module = importlib.util.module_from_spec(quality_spec)
    quality_spec.loader.exec_module(quality_module)
    
    pipeline_spec = importlib.util.spec_from_file_location(
        "pipeline",
        "smote_image_synthesis/pipeline.py"
    )
    pipeline_module = importlib.util.module_from_spec(pipeline_spec)
    pipeline_spec.loader.exec_module(pipeline_module)
    
    # Get classes from modules
    PipelineConfig = models_module.PipelineConfig
    ResNetEncoder = encoder_module.ResNetEncoder
    AutoencoderDecoder = decoder_module.AutoencoderDecoder
    ConstrainedSMOTE = smote_module.ConstrainedSMOTE
    QualityAssessor = quality_module.QualityAssessor
    SynthesisPipeline = pipeline_module.SynthesisPipeline
    
    # Create configuration
    config = PipelineConfig(
        config_name="image_synthesis_tool",
        output_dir="./output"
    )
    
    # Adjust configuration for image synthesis
    config.encoder_config.architecture = 'resnet18'
    config.encoder_config.embedding_dim = 128
    config.encoder_config.pretrained = False  # Use non-pretrained for faster execution
    config.decoder_config.image_shape = (3, 64, 64)  # (channels, height, width)
    config.decoder_config.decoder_type = 'autoencoder'
    config.smote_config.k_neighbors = 3
    config.quality_config.metrics = ['mse', 'mae']  # Use simple metrics
    
    # Create components
    encoder = ResNetEncoder(
        architecture=config.encoder_config.architecture,
        embedding_dim=config.encoder_config.embedding_dim,
        pretrained=config.encoder_config.pretrained,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    decoder = AutoencoderDecoder(
        embedding_dim=config.encoder_config.embedding_dim,
        image_shape=config.decoder_config.image_shape,
        device=encoder.device
    )
    
    smote = ConstrainedSMOTE(
        k_neighbors=config.smote_config.k_neighbors,
        use_clustering=config.smote_config.use_clustering,
        clustering_method=config.smote_config.cluster_method,
        random_state=config.seed
    )
    
    quality_assessor = QualityAssessor(
        metrics=config.quality_config.metrics,
        device=encoder.device
    )
    
    # Create pipeline
    pipeline = SynthesisPipeline(
        encoder=encoder,
        decoder=decoder,
        smote=smote,
        quality_assessor=quality_assessor
    )
    
    return pipeline, config

def load_and_preprocess_images(image_paths, target_size=(64, 64)):
    """Load and preprocess images from file paths."""
    images = []
    labels = []
    
    for i, img_path in enumerate(image_paths):
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Resize to target size
        img = img.resize(target_size)
        
        # Convert to tensor
        img_array = np.array(img).astype(np.float32)  # Shape: (H, W, C)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # Shape: (C, H, W)
        
        # Normalize using ImageNet stats
        normalize_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        normalize_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor / 255.0 - normalize_mean) / normalize_std
        
        images.append(img_tensor)
        # Assign a simple label (in a real scenario, you'd have actual labels)
        labels.append(0)  # All images belong to class 0 for this example
    
    if not images:
        raise ValueError("No images could be loaded")
    
    images_tensor = torch.stack(images)
    labels_array = np.array(labels)
    
    return images_tensor, labels_array

def save_images(images_tensor, output_dir, prefix="synthetic"):
    """Save tensor images to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Denormalize images
    denormalize_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    denormalize_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for i, img_tensor in enumerate(images_tensor):
        # Denormalize
        img_denorm = (img_tensor * denormalize_std + denormalize_mean).clamp(0, 1)
        
        # Convert to PIL Image
        img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        # Save image
        img_pil.save(output_path / f"{prefix}_{i:04d}.png")
    
    print(f"Saved {len(images_tensor)} images to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="SMOTE Image Synthesis Tool")
    parser.add_argument("--input", "-i", required=True, help="Input image file or directory")
    parser.add_argument("--output", "-o", default="./output", help="Output directory for synthetic images")
    parser.add_argument("--num-samples", "-n", type=int, default=10, help="Number of synthetic samples to generate")
    parser.add_argument("--target-size", "-s", type=int, nargs=2, default=[64, 64], help="Target image size (height width)")
    
    args = parser.parse_args()
    
    # Parse input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image file
        image_paths = [input_path]
    elif input_path.is_dir():
        # Directory of images
        image_paths = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")) + list(input_path.glob("*.jpeg"))
        if not image_paths:
            raise ValueError(f"No image files found in directory: {input_path}")
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    
    print(f"Found {len(image_paths)} input images")
    
    # Load and preprocess images
    print("Loading and preprocessing images...")
    images, labels = load_and_preprocess_images(image_paths, target_size=tuple(args.target_size))
    print(f"Loaded images with shape: {images.shape}")
    
    # Set up pipeline
    print("Setting up SMOTE pipeline...")
    pipeline, config = setup_pipeline()
    
    # Fit pipeline on input images
    print("Fitting pipeline on input images...")
    pipeline.fit(images, labels, train_decoder=False)  # Skip decoder training for speed
    
    # Generate synthetic images
    print(f"Generating {args.num_samples} synthetic images...")
    synthetic_images, synthetic_labels = pipeline.generate_synthetic_images(args.num_samples)
    
    if len(synthetic_images) == 0:
        print("No synthetic images generated")
        return
    
    print(f"Generated {len(synthetic_images)} synthetic images")
    
    # Save synthetic images
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Denormalize images for saving
    denormalize_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    denormalize_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for i, img_tensor in enumerate(synthetic_images):
        # Denormalize
        img_denorm = (img_tensor * denormalize_std + denormalize_mean).clamp(0, 1)
        
        # Convert to PIL Image
        img_np = (img_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        # Save image
        img_pil.save(output_dir / f"synthetic_{i:04d}.png")
    
    print(f"Synthetic images saved to: {output_dir}")
    
    # Optionally evaluate quality
    if len(images) > 0 and len(synthetic_images) > 0:
        print("Evaluating quality of synthetic images...")
        n_eval = min(5, len(images), len(synthetic_images))
        eval_real = images[:n_eval]
        eval_synthetic = synthetic_images[:n_eval]
        
        quality_results = pipeline.evaluate_quality(eval_synthetic, eval_real)
        
        print("Quality Results:")
        for metric, value in quality_results.items():
            if isinstance(value, dict):
                for sub_metric, sub_value in value.items():
                    print(f"  {metric}.{sub_metric}: {sub_value:.6f}")
            else:
                print(f"  {metric}: {value:.6f}")

if __name__ == "__main__":
    main()