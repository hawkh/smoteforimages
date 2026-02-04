# SMOTE Image Synthesis - Real Image Usage Guide

## Overview

This guide explains how to use the SMOTE Image Synthesis system with real images to generate synthetic data for addressing class imbalance in image datasets.

## Prerequisites

Before using the system with real images, you need to ensure all dependencies are properly installed:

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## Input/Output Workflow

The system follows this workflow:

1. **Input**: Real images (single image or directory of images)
2. **Processing**: 
   - Encode images to embedding space using neural networks (ResNet, etc.)
   - Apply SMOTE algorithm in the embedding space to generate synthetic embeddings
   - Decode synthetic embeddings back to image space using decoder networks
   - Assess quality of synthetic images
3. **Output**: Synthetic images that maintain semantic properties of the input

## Usage Examples

### Command Line Interface

Once dependencies are installed, you can use the system with:

```bash
# Generate synthetic images from a single input
python image_synthesis_tool.py --input my_image.jpg --output ./synthetic --num-samples 10

# Generate synthetic images from a directory of images
python image_synthesis_tool.py --input ./input_images --output ./synthetic --num-samples 20

# Use specific parameters
python image_synthesis_tool.py --input ./input_images --output ./synthetic --num-samples 50 --target-size 128 128
```

### Python API

```python
from smote_image_synthesis.pipeline import SynthesisPipeline
from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder
from smote_image_synthesis.decoders.autoencoder_decoder import AutoencoderDecoder
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE
from smote_image_synthesis.quality.assessor import QualityAssessor

# Create pipeline
pipeline = SynthesisPipeline(
    encoder=ResNetEncoder(architecture='resnet50', embedding_dim=512),
    decoder=AutoencoderDecoder(embedding_dim=512, image_shape=(3, 224, 224)),
    smote=ConstrainedSMOTE(k_neighbors=5, use_clustering=True),
    quality_assessor=QualityAssessor(metrics=['fid', 'ssim'])
)

# Load and preprocess your images
# images = load_your_images()  # Shape: [N, C, H, W]
# labels = your_labels  # Shape: [N]

# Fit pipeline on your data
# pipeline.fit(images, labels)

# Generate synthetic images
# synthetic_images, synthetic_labels = pipeline.generate_synthetic_images(100)

# Evaluate quality
# quality_results = pipeline.evaluate_quality(synthetic_images, images[:100])
```

## Supported Image Formats

The system supports common image formats:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Image Preprocessing

The system automatically:
- Resizes images to the target size (default 64x64, configurable)
- Normalizes images using ImageNet statistics
- Converts images to RGB format
- Applies any configured data augmentation

## Decoder Architecture Options

The system supports multiple decoder architectures:

1. **Autoencoder**: Good for reconstruction quality
2. **VAE (Variational Autoencoder)**: Probabilistic approach with sampling variations
3. **GAN (Generative Adversarial Network)**: Potentially highest quality but harder to train
4. **Diffusion Model**: State-of-the-art generative model

## Quality Assessment

The system evaluates synthetic images using:
- **FID (Fréchet Inception Distance)**: Measures similarity between real and synthetic distributions
- **SSIM (Structural Similarity Index)**: Measures structural similarity
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Perceptual similarity measure
- **Diversity Metrics**: Ensures synthetic images are varied

## Handling Class Imbalance

The system is specifically designed to address class imbalance by:
- Identifying underrepresented classes in your dataset
- Generating synthetic samples for minority classes
- Preserving semantic properties of the original class
- Validating that synthetic samples are realistic

## Configuration Options

You can configure:
- Encoder architecture (ResNet18/50/101)
- Decoder type (autoencoder, VAE, GAN, diffusion)
- Embedding dimension size
- SMOTE parameters (k-neighbors, clustering method)
- Quality assessment metrics
- Training parameters for decoder fine-tuning

## Example Use Cases

1. **Medical Imaging**: Generate additional examples for rare conditions
2. **Object Detection**: Balance datasets with rare object categories
3. **Face Recognition**: Augment datasets with underrepresented demographics
4. **Satellite Imagery**: Generate examples for rare geographic features

## Troubleshooting

If you encounter dependency issues:
1. Make sure PyTorch is installed with the correct CUDA version
2. Install torchvision separately if needed: `pip install torchvision`
3. Check that all requirements from requirements.txt are installed

## Performance Tips

- Use GPU acceleration when available for faster processing
- Start with smaller image sizes (64x64) for prototyping
- Use autoencoder decoder for faster results, GAN for highest quality
- Monitor memory usage with large datasets and adjust batch sizes accordingly

## Expected Output

The system will generate:
- Synthetic images in the specified output directory
- Quality assessment reports
- Configuration files for reproducibility
- Log files with processing information

The synthetic images will maintain the visual characteristics of your input while introducing appropriate variations to enhance dataset diversity.