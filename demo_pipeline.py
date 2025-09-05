#!/usr/bin/env python3
"""
Comprehensive example demonstrating the SMOTE image synthesis pipeline.

This script shows how to:
1. Set up the complete pipeline with encoder, decoder, SMOTE, and quality assessment
2. Load and preprocess image data
3. Train the pipeline components
4. Generate synthetic images
5. Evaluate quality and generate reports
"""

import torch
import numpy as np
from pathlib import Path
import logging
import argparse
from typing import Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import pipeline components
from smote_image_synthesis.data.models import PipelineConfig
from smote_image_synthesis.data.preprocessor import ImagePreprocessor
from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder
from smote_image_synthesis.decoders.autoencoder_decoder import AutoencoderDecoder
from smote_image_synthesis.decoders.vae_decoder import VAEDecoder
from smote_image_synthesis.decoders.autoencoder_trainer import AutoencoderTrainer
from smote_image_synthesis.decoders.vae_trainer import VAETrainer
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE
from smote_image_synthesis.quality.assessor import QualityAssessor
from smote_image_synthesis.quality.reporter import QualityReporter
from smote_image_synthesis.pipeline import SynthesisPipeline


def create_synthetic_dataset(n_samples: int = 200, image_size: int = 64) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Create a synthetic dataset for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        image_size: Size of square images
        
    Returns:
        Tuple of (images, labels)
    """
    logger.info(f"Creating synthetic dataset with {n_samples} samples")
    
    # Create different classes with distinct patterns
    images = []
    labels = []
    
    n_per_class = n_samples // 3
    
    # Class 0: Horizontal stripes
    for i in range(n_per_class):
        img = torch.zeros(3, image_size, image_size)
        stripe_width = 8
        for y in range(0, image_size, stripe_width * 2):
            img[:, y:y+stripe_width, :] = torch.rand(3, 1, 1) * 0.8 + 0.2
        
        # Add noise
        img += torch.randn_like(img) * 0.1
        img = torch.clamp(img, 0, 1)
        
        images.append(img)
        labels.append(0)
    
    # Class 1: Vertical stripes
    for i in range(n_per_class):
        img = torch.zeros(3, image_size, image_size)
        stripe_width = 8
        for x in range(0, image_size, stripe_width * 2):
            img[:, :, x:x+stripe_width] = torch.rand(3, 1, 1) * 0.8 + 0.2
        
        # Add noise
        img += torch.randn_like(img) * 0.1
        img = torch.clamp(img, 0, 1)
        
        images.append(img)
        labels.append(1)
    
    # Class 2: Checkerboard pattern
    for i in range(n_samples - 2 * n_per_class):
        img = torch.zeros(3, image_size, image_size)
        square_size = 16
        
        for y in range(0, image_size, square_size):
            for x in range(0, image_size, square_size):
                if (y // square_size + x // square_size) % 2 == 0:
                    color = torch.rand(3) * 0.8 + 0.2
                    img[:, y:y+square_size, x:x+square_size] = color.view(3, 1, 1)
        
        # Add noise
        img += torch.randn_like(img) * 0.1
        img = torch.clamp(img, 0, 1)
        
        images.append(img)
        labels.append(2)
    
    images_tensor = torch.stack(images)
    labels_array = np.array(labels)
    
    logger.info(f"Dataset created with shape {images_tensor.shape}")
    logger.info(f"Class distribution: {np.bincount(labels_array)}")
    
    return images_tensor, labels_array


def setup_pipeline(config: PipelineConfig, device: torch.device) -> SynthesisPipeline:
    """
    Set up the complete SMOTE image synthesis pipeline.
    
    Args:
        config: Pipeline configuration
        device: PyTorch device
        
    Returns:
        Configured pipeline
    """
    logger.info("Setting up pipeline components")
    
    # Create encoder
    encoder = ResNetEncoder(
        architecture=config.encoder_config.architecture,
        embedding_dim=config.encoder_config.embedding_dim,
        pretrained=config.encoder_config.pretrained,
        device=device,
        freeze_backbone=config.encoder_config.freeze_backbone,
        dropout_rate=config.encoder_config.dropout_rate
    )
    
    # Create decoder based on configuration
    if config.decoder_config.decoder_type == 'autoencoder':
        decoder = AutoencoderDecoder(
            embedding_dim=config.encoder_config.embedding_dim,
            image_shape=config.decoder_config.image_shape,
            device=device
        )
    elif config.decoder_config.decoder_type == 'vae':
        decoder = VAEDecoder(
            embedding_dim=config.encoder_config.embedding_dim,
            image_shape=config.decoder_config.image_shape,
            latent_dim=128,
            device=device
        )
    else:
        raise ValueError(f"Unsupported decoder type: {config.decoder_config.decoder_type}")
    
    # Create SMOTE
    smote = ConstrainedSMOTE(
        k_neighbors=config.smote_config.k_neighbors,
        sampling_strategy=config.smote_config.sampling_strategy,
        use_clustering=config.smote_config.use_clustering,
        clustering_method=config.smote_config.cluster_method,
        max_distance_threshold=config.smote_config.distance_threshold,
        random_state=config.seed
    )
    
    # Create quality assessor
    quality_assessor = QualityAssessor(
        metrics=config.quality_config.metrics,
        fid_batch_size=config.quality_config.fid_batch_size,
        compute_diversity=config.quality_config.compute_diversity,
        device=device
    )
    
    # Create pipeline
    pipeline = SynthesisPipeline(
        encoder=encoder,
        decoder=decoder,
        smote=smote,
        quality_assessor=quality_assessor
    )
    
    logger.info("Pipeline setup complete")
    return pipeline


def train_decoder(decoder, images: torch.Tensor, embeddings: torch.Tensor, config: PipelineConfig):
    """Train the decoder component."""
    logger.info("Training decoder")
    
    # Split data for training and validation
    n_train = int(0.8 * len(images))
    train_images = images[:n_train]
    train_embeddings = embeddings[:n_train]
    val_images = images[n_train:]
    val_embeddings = embeddings[n_train:]
    
    if isinstance(decoder, AutoencoderDecoder):
        trainer = AutoencoderTrainer(
            decoder=decoder,
            learning_rate=config.decoder_config.learning_rate,
            use_perceptual_loss=config.decoder_config.use_perceptual_loss,
            early_stopping_patience=config.decoder_config.early_stopping_patience
        )
        
        history = trainer.train(
            train_embeddings=train_embeddings,
            train_images=train_images,
            val_embeddings=val_embeddings if len(val_embeddings) > 0 else None,
            val_images=val_images if len(val_images) > 0 else None,
            num_epochs=min(config.decoder_config.num_epochs, 20),  # Reduced for demo
            batch_size=config.decoder_config.batch_size
        )
        
    elif isinstance(decoder, VAEDecoder):
        trainer = VAETrainer(
            vae_decoder=decoder,
            learning_rate=config.decoder_config.learning_rate,
            early_stopping_patience=config.decoder_config.early_stopping_patience
        )
        
        history = trainer.train(
            train_embeddings=train_embeddings,
            train_images=train_images,
            val_embeddings=val_embeddings if len(val_embeddings) > 0 else None,
            val_images=val_images if len(val_images) > 0 else None,
            num_epochs=min(config.decoder_config.num_epochs, 20),  # Reduced for demo
            batch_size=config.decoder_config.batch_size
        )
    
    logger.info("Decoder training complete")


def run_pipeline_demo(args):
    """Run the complete pipeline demonstration."""
    logger.info("Starting SMOTE Image Synthesis Pipeline Demo")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = PipelineConfig(
        config_name="demo_config",
        output_dir=str(output_dir),
        seed=42
    )
    
    # Adjust configuration for demo
    config.encoder_config.architecture = 'resnet18'  # Smaller for demo
    config.encoder_config.embedding_dim = 128
    config.encoder_config.pretrained = False  # Avoid downloading pretrained weights
    config.decoder_config.image_shape = (3, 64, 64)
    config.decoder_config.decoder_type = args.decoder_type
    config.decoder_config.num_epochs = 10  # Reduced for demo
    config.smote_config.k_neighbors = 3
    config.quality_config.metrics = ['mse', 'mae']  # Simple metrics for demo
    
    # Save configuration
    config_path = output_dir / "pipeline_config.json"
    config.save_config(str(config_path))
    logger.info(f"Configuration saved to {config_path}")
    
    # Create synthetic dataset
    images, labels = create_synthetic_dataset(n_samples=args.n_samples, image_size=64)
    
    # Set up pipeline
    pipeline = setup_pipeline(config, device)
    
    # Fit pipeline (this encodes images and fits SMOTE)
    logger.info("Fitting pipeline on training data")
    pipeline.fit(images, labels)
    
    # Train decoder if requested
    if args.train_decoder:
        embeddings = pipeline.encoder.encode(images)
        train_decoder(pipeline.decoder, images, embeddings, config)
    
    # Generate synthetic images
    logger.info("Generating synthetic images")
    n_synthetic = args.n_synthetic or len(images) // 2
    synthetic_images, synthetic_labels = pipeline.generate_synthetic_images(n_synthetic)
    
    if len(synthetic_images) == 0:
        logger.warning("No synthetic images generated")
        return
    
    logger.info(f"Generated {len(synthetic_images)} synthetic images")
    logger.info(f"Synthetic class distribution: {np.bincount(synthetic_labels)}")
    
    # Evaluate quality
    logger.info("Evaluating quality of synthetic images")
    
    # Use a subset for evaluation to avoid memory issues
    n_eval = min(50, len(images), len(synthetic_images))
    eval_real = images[:n_eval]
    eval_synthetic = synthetic_images[:n_eval]
    
    quality_results = pipeline.evaluate_quality(eval_synthetic, eval_real)
    
    logger.info("Quality Results:")
    for metric, value in quality_results.items():
        if isinstance(value, dict):
            for sub_metric, sub_value in value.items():
                logger.info(f"  {metric}.{sub_metric}: {sub_value:.6f}")
        else:
            logger.info(f"  {metric}: {value:.6f}")
    
    # Generate quality report
    if args.generate_report:
        logger.info("Generating quality assessment report")
        
        reporter = QualityReporter(
            output_dir=str(output_dir),
            report_format='html',
            save_plots=True
        )
        
        report_path = reporter.generate_comprehensive_report(
            quality_results=quality_results,
            synthetic_images=eval_synthetic,
            real_images=eval_real,
            report_name="demo_quality_report"
        )
        
        logger.info(f"Quality report generated: {report_path}")
        
        # Export metrics to CSV
        csv_path = reporter.export_metrics_csv(quality_results, "demo_metrics")
        logger.info(f"Metrics exported to CSV: {csv_path}")
    
    # Save some sample images
    logger.info("Saving sample images")
    
    import matplotlib.pyplot as plt
    
    # Create comparison plot
    n_display = min(8, len(synthetic_images))
    fig, axes = plt.subplots(2, n_display, figsize=(2*n_display, 4))
    
    for i in range(n_display):
        # Real images (top row)
        real_img = images[i].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(real_img)
        axes[0, i].set_title(f'Real (Class {labels[i]})')
        axes[0, i].axis('off')
        
        # Synthetic images (bottom row)
        synth_img = synthetic_images[i].permute(1, 2, 0).cpu().detach().numpy()
        if synth_img.min() < 0:  # Normalize if needed
            synth_img = (synth_img + 1) / 2
        synth_img = np.clip(synth_img, 0, 1)
        
        axes[1, i].imshow(synth_img)
        axes[1, i].set_title(f'Synthetic (Class {synthetic_labels[i]})')
        axes[1, i].axis('off')
    
    plt.suptitle('Real vs Synthetic Images Comparison')
    plt.tight_layout()
    
    sample_path = output_dir / "sample_comparison.png"
    plt.savefig(sample_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Sample comparison saved to {sample_path}")
    
    logger.info("Pipeline demonstration complete!")
    logger.info(f"Results saved to: {output_dir}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="SMOTE Image Synthesis Pipeline Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='./demo_output',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--n-samples', 
        type=int, 
        default=150,
        help='Number of samples in synthetic dataset'
    )
    
    parser.add_argument(
        '--n-synthetic', 
        type=int, 
        default=None,
        help='Number of synthetic images to generate (default: half of dataset)'
    )
    
    parser.add_argument(
        '--decoder-type', 
        type=str, 
        choices=['autoencoder', 'vae'], 
        default='autoencoder',
        help='Type of decoder to use'
    )
    
    parser.add_argument(
        '--train-decoder', 
        action='store_true',
        help='Whether to train the decoder (takes longer but better quality)'
    )
    
    parser.add_argument(
        '--generate-report', 
        action='store_true',
        help='Whether to generate comprehensive quality report'
    )
    
    parser.add_argument(
        '--cpu', 
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )
    
    args = parser.parse_args()
    
    try:
        run_pipeline_demo(args)
    except Exception as e:
        logger.error(f"Demo failed with error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
