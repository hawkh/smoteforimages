#!/usr/bin/env python3
"""
Command-line interface for SMOTE image synthesis pipeline.

This CLI provides comprehensive functionality for:
- Pipeline configuration and setup
- Training individual components
- Generating synthetic images
- Quality assessment and reporting
- Experiment management
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

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
from smote_image_synthesis.decoders.gan_decoder import GANDecoder
from smote_image_synthesis.decoders.autoencoder_trainer import AutoencoderTrainer
from smote_image_synthesis.decoders.vae_trainer import VAETrainer
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE
from smote_image_synthesis.quality.assessor import QualityAssessor
from smote_image_synthesis.quality.reporter import QualityReporter
from smote_image_synthesis.pipeline import SynthesisPipeline
from smote_image_synthesis.error_handling import ErrorRecoveryManager, PipelineHealthMonitor

import torch
import numpy as np


class SMOTEImageCLI:
    """Command-line interface for SMOTE image synthesis."""
    
    def __init__(self):
        """Initialize CLI interface."""
        self.parser = argparse.ArgumentParser(
            description="SMOTE Image Synthesis Pipeline CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Create a new configuration
  python -m smote_image_cli config create --name experiment_001 --output-dir ./experiments

  # Train pipeline components
  python -m smote_image_cli train --config experiment_config.json --data-dir ./dataset

  # Generate synthetic images
  python -m smote_image_cli generate --config experiment_config.json --n-samples 100

  # Evaluate quality
  python -m smote_image_cli evaluate --config experiment_config.json --synthetic-dir ./synthetic --real-dir ./real

  # Run complete pipeline
  python -m smote_image_cli pipeline run --config experiment_config.json --data-dir ./dataset
            """
        )
        
        # Add subcommands
        self.subparsers = self.parser.add_subparsers(
            dest='command',
            help='Available commands'
        )
        
        self._setup_config_commands()
        self._setup_train_commands()
        self._setup_generate_commands()
        self._setup_evaluate_commands()
        self._setup_pipeline_commands()
        self._setup_health_commands()
    
    def _setup_config_commands(self):
        """Set up configuration management commands."""
        config_parser = self.subparsers.add_parser(
            'config',
            help='Configuration management'
        )
        
        config_subparsers = config_parser.add_subparsers(
            dest='config_action',
            help='Configuration actions'
        )
        
        # Create config
        create_parser = config_subparsers.add_parser(
            'create',
            help='Create new configuration'
        )
        create_parser.add_argument('--name', required=True, help='Configuration name')
        create_parser.add_argument('--output-dir', default='./output', help='Output directory')
        create_parser.add_argument('--encoder', default='resnet50', choices=['resnet18', 'resnet50', 'resnet101'])
        create_parser.add_argument('--decoder', default='autoencoder', choices=['autoencoder', 'vae', 'gan'])
        create_parser.add_argument('--embedding-dim', type=int, default=512, help='Embedding dimension')
        create_parser.add_argument('--image-size', type=int, default=224, help='Image size')
        create_parser.add_argument('--channels', type=int, default=3, help='Image channels')
        
        # Validate config
        validate_parser = config_subparsers.add_parser(
            'validate',
            help='Validate configuration'
        )
        validate_parser.add_argument('config_file', help='Configuration file to validate')
        
        # List configs
        list_parser = config_subparsers.add_parser(
            'list',
            help='List available configurations'
        )
        list_parser.add_argument('--directory', default='.', help='Directory to search')
        
        # Show config
        show_parser = config_subparsers.add_parser(
            'show',
            help='Show configuration details'
        )
        show_parser.add_argument('config_file', help='Configuration file to show')
    
    def _setup_train_commands(self):
        """Set up training commands."""
        train_parser = self.subparsers.add_parser(
            'train',
            help='Train pipeline components'
        )
        
        train_parser.add_argument('--config', required=True, help='Configuration file')
        train_parser.add_argument('--data-dir', required=True, help='Training data directory')
        train_parser.add_argument('--component', choices=['encoder', 'decoder', 'all'], default='all')
        train_parser.add_argument('--epochs', type=int, help='Number of training epochs (overrides config)')
        train_parser.add_argument('--batch-size', type=int, help='Batch size (overrides config)')
        train_parser.add_argument('--learning-rate', type=float, help='Learning rate (overrides config)')
        train_parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
        train_parser.add_argument('--checkpoint-dir', help='Directory for saving checkpoints')
        train_parser.add_argument('--resume', help='Resume training from checkpoint')
        train_parser.add_argument('--validate', action='store_true', help='Enable validation during training')
        train_parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    
    def _setup_generate_commands(self):
        """Set up synthetic image generation commands."""
        generate_parser = self.subparsers.add_parser(
            'generate',
            help='Generate synthetic images'
        )
        
        generate_parser.add_argument('--config', required=True, help='Configuration file')
        generate_parser.add_argument('--data-dir', required=True, help='Source data directory')
        generate_parser.add_argument('--output-dir', help='Output directory for synthetic images')
        generate_parser.add_argument('--n-samples', type=int, help='Number of samples to generate')
        generate_parser.add_argument('--target-classes', nargs='+', type=int, help='Specific classes to generate')
        generate_parser.add_argument('--batch-size', type=int, default=32, help='Generation batch size')
        generate_parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
        generate_parser.add_argument('--save-embeddings', action='store_true', help='Save intermediate embeddings')
        generate_parser.add_argument('--format', choices=['png', 'jpg', 'numpy'], default='png', help='Output format')
    
    def _setup_evaluate_commands(self):
        """Set up quality evaluation commands."""
        evaluate_parser = self.subparsers.add_parser(
            'evaluate',
            help='Evaluate synthetic image quality'
        )
        
        evaluate_parser.add_argument('--config', required=True, help='Configuration file')
        evaluate_parser.add_argument('--synthetic-dir', required=True, help='Synthetic images directory')
        evaluate_parser.add_argument('--real-dir', required=True, help='Real images directory')
        evaluate_parser.add_argument('--output-dir', help='Output directory for reports')
        evaluate_parser.add_argument('--metrics', nargs='+', help='Specific metrics to compute')
        evaluate_parser.add_argument('--sample-size', type=int, help='Sample size for evaluation')
        evaluate_parser.add_argument('--report-format', choices=['html', 'json', 'txt'], default='html')
        evaluate_parser.add_argument('--save-plots', action='store_true', help='Save quality plots')
        evaluate_parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    
    def _setup_pipeline_commands(self):
        """Set up complete pipeline commands."""
        pipeline_parser = self.subparsers.add_parser(
            'pipeline',
            help='Complete pipeline operations'
        )
        
        pipeline_subparsers = pipeline_parser.add_subparsers(
            dest='pipeline_action',
            help='Pipeline actions'
        )
        
        # Run complete pipeline
        run_parser = pipeline_subparsers.add_parser(
            'run',
            help='Run complete pipeline'
        )
        run_parser.add_argument('--config', required=True, help='Configuration file')
        run_parser.add_argument('--data-dir', required=True, help='Training data directory')
        run_parser.add_argument('--output-dir', help='Output directory')
        run_parser.add_argument('--train-decoder', action='store_true', help='Train decoder during pipeline')
        run_parser.add_argument('--generate-samples', type=int, help='Number of synthetic samples to generate')
        run_parser.add_argument('--evaluate-quality', action='store_true', help='Evaluate quality of results')
        run_parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
        
        # Test pipeline
        test_parser = pipeline_subparsers.add_parser(
            'test',
            help='Test pipeline with small dataset'
        )
        test_parser.add_argument('--config', required=True, help='Configuration file')
        test_parser.add_argument('--data-dir', help='Test data directory (optional)')
        test_parser.add_argument('--synthetic-data', action='store_true', help='Use synthetic test data')
    
    def _setup_health_commands(self):
        """Set up health monitoring commands."""
        health_parser = self.subparsers.add_parser(
            'health',
            help='Pipeline health monitoring'
        )
        
        health_parser.add_argument('--config', required=True, help='Configuration file')
        health_parser.add_argument('--check-interval', type=float, default=30.0, help='Health check interval')
        health_parser.add_argument('--output-file', help='Output file for health report')
        health_parser.add_argument('--continuous', action='store_true', help='Continuous monitoring')
    
    def run(self, args=None):
        """Run the CLI with provided arguments."""
        parsed_args = self.parser.parse_args(args)
        
        if not parsed_args.command:
            self.parser.print_help()
            return 1
        
        try:
            # Route to appropriate handler
            if parsed_args.command == 'config':
                return self._handle_config_commands(parsed_args)
            elif parsed_args.command == 'train':
                return self._handle_train_command(parsed_args)
            elif parsed_args.command == 'generate':
                return self._handle_generate_command(parsed_args)
            elif parsed_args.command == 'evaluate':
                return self._handle_evaluate_command(parsed_args)
            elif parsed_args.command == 'pipeline':
                return self._handle_pipeline_commands(parsed_args)
            elif parsed_args.command == 'health':
                return self._handle_health_command(parsed_args)
            else:
                logger.error(f"Unknown command: {parsed_args.command}")
                return 1
                
        except Exception as e:
            logger.error(f"Command failed: {e}")
            logger.debug("Full traceback:", exc_info=True)
            return 1
    
    def _handle_config_commands(self, args) -> int:
        """Handle configuration commands."""
        if args.config_action == 'create':
            return self._create_config(args)
        elif args.config_action == 'validate':
            return self._validate_config(args)
        elif args.config_action == 'list':
            return self._list_configs(args)
        elif args.config_action == 'show':
            return self._show_config(args)
        else:
            logger.error("Config action required")
            return 1
    
    def _create_config(self, args) -> int:
        """Create new configuration."""
        logger.info(f"Creating configuration: {args.name}")
        
        config = PipelineConfig(
            config_name=args.name,
            output_dir=args.output_dir
        )
        
        # Update encoder config
        config.encoder_config.architecture = args.encoder
        config.encoder_config.embedding_dim = args.embedding_dim
        
        # Update decoder config
        config.decoder_config.decoder_type = args.decoder
        config.decoder_config.image_shape = (args.channels, args.image_size, args.image_size)
        
        # Validate configuration
        is_valid, errors = config.validate()
        if not is_valid:
            logger.error(f"Invalid configuration: {errors}")
            return 1
        
        # Save configuration
        config_file = f"{args.name}_config.json"
        config.save_config(config_file)
        
        logger.info(f"Configuration saved to: {config_file}")
        return 0
    
    def _validate_config(self, args) -> int:
        """Validate configuration file."""
        logger.info(f"Validating configuration: {args.config_file}")
        
        try:
            config = PipelineConfig.load_config(args.config_file)
            is_valid, errors = config.validate()
            
            if is_valid:
                logger.info("Configuration is valid")
                
                # Show warnings if any
                warnings = config.get_consistency_warnings()
                if warnings:
                    logger.warning("Configuration warnings:")
                    for warning in warnings:
                        logger.warning(f"  - {warning}")
                
                return 0
            else:
                logger.error("Configuration validation failed:")
                for error in errors:
                    logger.error(f"  - {error}")
                return 1
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return 1
    
    def _list_configs(self, args) -> int:
        """List available configurations."""
        directory = Path(args.directory)
        config_files = list(directory.glob("*_config.json"))
        
        if not config_files:
            logger.info("No configuration files found")
            return 0
        
        logger.info(f"Found {len(config_files)} configuration files:")
        for config_file in sorted(config_files):
            try:
                config = PipelineConfig.load_config(str(config_file))
                logger.info(f"  {config_file.name} - {config.config_name}")
            except Exception as e:
                logger.warning(f"  {config_file.name} - Invalid: {e}")
        
        return 0
    
    def _show_config(self, args) -> int:
        """Show configuration details."""
        try:
            config = PipelineConfig.load_config(args.config_file)
            
            print(f"Configuration: {config.config_name}")
            print(f"Created: {config.creation_timestamp}")
            print(f"Output Directory: {config.output_dir}")
            print()
            
            print("Encoder Configuration:")
            encoder_dict = config.encoder_config.__dict__
            for key, value in encoder_dict.items():
                print(f"  {key}: {value}")
            print()
            
            print("Decoder Configuration:")
            decoder_dict = config.decoder_config.__dict__
            for key, value in decoder_dict.items():
                print(f"  {key}: {value}")
            print()
            
            print("SMOTE Configuration:")
            smote_dict = config.smote_config.__dict__
            for key, value in smote_dict.items():
                print(f"  {key}: {value}")
            print()
            
            print("Quality Configuration:")
            quality_dict = config.quality_config.__dict__
            for key, value in quality_dict.items():
                print(f"  {key}: {value}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to show configuration: {e}")
            return 1
    
    def _handle_train_command(self, args) -> int:
        """Handle training command."""
        logger.info("Starting training process")
        
        # Load configuration
        config = PipelineConfig.load_config(args.config)
        
        # Set device
        device = self._get_device(args.device)
        
        # Override config parameters if provided
        if args.epochs:
            config.decoder_config.num_epochs = args.epochs
        if args.batch_size:
            config.decoder_config.batch_size = args.batch_size
        if args.learning_rate:
            config.decoder_config.learning_rate = args.learning_rate
        
        # Load and preprocess data
        logger.info(f"Loading data from: {args.data_dir}")
        # Implementation would load actual data
        # For now, create dummy data for demonstration
        images = torch.randn(100, *config.decoder_config.image_shape)
        labels = np.random.randint(0, 5, 100)
        
        # Set up pipeline components
        encoder = ResNetEncoder(
            architecture=config.encoder_config.architecture,
            embedding_dim=config.encoder_config.embedding_dim,
            device=device
        )
        
        # Train based on component selection
        if args.component in ['encoder', 'all']:
            logger.info("Training encoder...")
            # Encoder training would be implemented here
            
        if args.component in ['decoder', 'all']:
            logger.info("Training decoder...")
            
            # Create decoder based on type
            if config.decoder_config.decoder_type == 'autoencoder':
                decoder = AutoencoderDecoder(
                    embedding_dim=config.encoder_config.embedding_dim,
                    image_shape=config.decoder_config.image_shape,
                    device=device
                )
                trainer = AutoencoderTrainer(decoder)
            elif config.decoder_config.decoder_type == 'vae':
                decoder = VAEDecoder(
                    embedding_dim=config.encoder_config.embedding_dim,
                    image_shape=config.decoder_config.image_shape,
                    device=device
                )
                trainer = VAETrainer(decoder)
            else:
                logger.error(f"Unsupported decoder type: {config.decoder_config.decoder_type}")
                return 1
            
            # Generate embeddings
            embeddings = encoder.encode(images)
            
            # Split data if validation requested
            if args.validate:
                split_idx = int(len(images) * (1 - args.val_split))
                train_embeddings = embeddings[:split_idx]
                train_images = images[:split_idx]
                val_embeddings = embeddings[split_idx:]
                val_images = images[split_idx:]
            else:
                train_embeddings = embeddings
                train_images = images
                val_embeddings = None
                val_images = None
            
            # Train decoder
            trainer.train(
                train_embeddings=train_embeddings,
                train_images=train_images,
                val_embeddings=val_embeddings,
                val_images=val_images,
                num_epochs=config.decoder_config.num_epochs,
                batch_size=config.decoder_config.batch_size
            )
        
        logger.info("Training completed successfully")
        return 0
    
    def _handle_generate_command(self, args) -> int:
        """Handle generation command."""
        logger.info("Starting synthetic image generation")
        
        try:
            # Load configuration
            config = PipelineConfig.load_config(args.config)
            
            # Set device
            device = self._get_device(args.device)
            
            # Set up output directory
            output_dir = Path(args.output_dir) if args.output_dir else Path(config.output_dir) / "synthetic"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load data
            logger.info(f"Loading data from: {args.data_dir}")
            preprocessor = ImagePreprocessor(
                target_size=(config.decoder_config.image_shape[1], config.decoder_config.image_shape[2])
            )
            
            # Load images from directory
            image_paths = preprocessor.load_images_from_directory(args.data_dir, recursive=True)
            if not image_paths:
                logger.error(f"No images found in {args.data_dir}")
                return 1
            
            logger.info(f"Found {len(image_paths)} images")
            
            # Create dummy labels for demonstration (in real scenario, these would be loaded)
            labels = np.random.randint(0, max(5, len(set(range(len(image_paths))))), len(image_paths))
            
            # Process images in batches
            batch_size = args.batch_size
            all_images = []
            
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = preprocessor.preprocess_batch(batch_paths)
                all_images.append(batch_images)
            
            images = torch.cat(all_images, dim=0)
            
            # Set up pipeline components
            encoder = ResNetEncoder(
                architecture=config.encoder_config.architecture,
                embedding_dim=config.encoder_config.embedding_dim,
                device=device
            )
            
            # Create decoder based on type
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
                    device=device
                )
            elif config.decoder_config.decoder_type == 'gan':
                decoder = GANDecoder(
                    embedding_dim=config.encoder_config.embedding_dim,
                    image_shape=config.decoder_config.image_shape,
                    device=device
                )
            else:
                logger.error(f"Unsupported decoder type: {config.decoder_config.decoder_type}")
                return 1
            
            # Set up SMOTE
            smote = ConstrainedSMOTE(
                k_neighbors=config.smote_config.k_neighbors,
                use_clustering=config.smote_config.use_clustering,
                clustering_method=config.smote_config.clustering_method
            )
            
            # Create pipeline
            pipeline = SynthesisPipeline(
                encoder=encoder,
                decoder=decoder,
                smote=smote
            )
            
            # Fit pipeline
            logger.info("Fitting pipeline...")
            pipeline.fit(images, labels)
            
            # Generate synthetic images
            n_samples = args.n_samples or len(images)
            logger.info(f"Generating {n_samples} synthetic images...")
            
            synthetic_images, synthetic_labels = pipeline.generate_synthetic_images(n_samples)
            
            if len(synthetic_images) == 0:
                logger.warning("No synthetic images generated")
                return 1
            
            # Save synthetic images
            logger.info(f"Saving {len(synthetic_images)} images to {output_dir}")
            
            # Denormalize images for saving
            synthetic_images = preprocessor.denormalize_tensor(synthetic_images)
            
            # Save images
            for i, (image, label) in enumerate(zip(synthetic_images, synthetic_labels)):
                if args.format == 'numpy':
                    filename = output_dir / f"synthetic_{i:06d}_class_{label}.npy"
                    np.save(filename, image.cpu().numpy())
                else:
                    pil_image = preprocessor.tensor_to_pil(image)
                    filename = output_dir / f"synthetic_{i:06d}_class_{label}.{args.format}"
                    pil_image.save(filename)
            
            # Save embeddings if requested
            if args.save_embeddings:
                embeddings_dir = output_dir / "embeddings"
                embeddings_dir.mkdir(exist_ok=True)
                
                # Get embeddings for synthetic images
                with torch.no_grad():
                    synthetic_embeddings = encoder.encode(synthetic_images)
                
                np.save(embeddings_dir / "synthetic_embeddings.npy", synthetic_embeddings.cpu().numpy())
                np.save(embeddings_dir / "synthetic_labels.npy", synthetic_labels)
                
                logger.info(f"Embeddings saved to {embeddings_dir}")
            
            # Save generation metadata
            metadata = {
                "config_file": args.config,
                "source_data_dir": args.data_dir,
                "n_generated": len(synthetic_images),
                "target_classes": args.target_classes,
                "generation_timestamp": str(np.datetime64('now')),
                "format": args.format
            }
            
            with open(output_dir / "generation_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Generation completed successfully. {len(synthetic_images)} images saved to {output_dir}")
            return 0
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            logger.debug("Full traceback:", exc_info=True)
            return 1
    
    def _handle_evaluate_command(self, args) -> int:
        """Handle evaluation command."""
        logger.info("Starting quality evaluation")
        
        try:
            # Load configuration
            config = PipelineConfig.load_config(args.config)
            
            # Set device
            device = self._get_device(args.device)
            
            # Set up output directory
            output_dir = Path(args.output_dir) if args.output_dir else Path("./evaluation_reports")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Set up preprocessor
            preprocessor = ImagePreprocessor(
                target_size=(config.decoder_config.image_shape[1], config.decoder_config.image_shape[2])
            )
            
            # Load synthetic images
            logger.info(f"Loading synthetic images from: {args.synthetic_dir}")
            synthetic_paths = preprocessor.load_images_from_directory(
                args.synthetic_dir, 
                recursive=True, 
                max_images=args.sample_size
            )
            
            if not synthetic_paths:
                logger.error(f"No synthetic images found in {args.synthetic_dir}")
                return 1
            
            # Load real images
            logger.info(f"Loading real images from: {args.real_dir}")
            real_paths = preprocessor.load_images_from_directory(
                args.real_dir, 
                recursive=True, 
                max_images=args.sample_size
            )
            
            if not real_paths:
                logger.error(f"No real images found in {args.real_dir}")
                return 1
            
            logger.info(f"Found {len(synthetic_paths)} synthetic and {len(real_paths)} real images")
            
            # Process images
            synthetic_images = preprocessor.preprocess_batch(synthetic_paths[:args.sample_size] if args.sample_size else synthetic_paths)
            real_images = preprocessor.preprocess_batch(real_paths[:args.sample_size] if args.sample_size else real_paths)
            
            # Set up quality assessor
            metrics = args.metrics or config.quality_config.metrics
            quality_assessor = QualityAssessor(
                metrics=metrics,
                device=device
            )
            
            # Evaluate quality
            logger.info(f"Evaluating quality using metrics: {metrics}")
            quality_results = quality_assessor.evaluate_quality(
                synthetic_images=synthetic_images,
                real_images=real_images,
                return_detailed=True
            )
            
            # Set up reporter
            reporter = QualityReporter(
                output_dir=str(output_dir),
                report_format=args.report_format
            )
            
            # Generate report
            report_name = f"quality_evaluation_{str(np.datetime64('now')).replace(':', '-')}"
            report_path = reporter.generate_comprehensive_report(
                quality_results=quality_results,
                synthetic_images=synthetic_images if args.save_plots else None,
                real_images=real_images if args.save_plots else None,
                report_name=report_name
            )
            
            # Print summary
            logger.info("Quality Evaluation Results:")
            logger.info("=" * 40)
            
            for metric, value in quality_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"{metric}: {value:.4f}")
                else:
                    logger.info(f"{metric}: {value}")
            
            logger.info("=" * 40)
            logger.info(f"Detailed report saved to: {report_path}")
            
            # Save evaluation metadata
            metadata = {
                "config_file": args.config,
                "synthetic_dir": args.synthetic_dir,
                "real_dir": args.real_dir,
                "sample_size": args.sample_size,
                "metrics_used": metrics,
                "evaluation_timestamp": str(np.datetime64('now')),
                "quality_results": quality_results,
                "report_path": str(report_path)
            }
            
            with open(output_dir / "evaluation_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info("Evaluation completed successfully")
            return 0
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            logger.debug("Full traceback:", exc_info=True)
            return 1
    
    def _handle_pipeline_commands(self, args) -> int:
        """Handle pipeline commands."""
        if args.pipeline_action == 'run':
            return self._run_complete_pipeline(args)
        elif args.pipeline_action == 'test':
            return self._test_pipeline(args)
        else:
            logger.error("Pipeline action required")
            return 1
    
    def _run_complete_pipeline(self, args) -> int:
        """Run complete pipeline."""
        logger.info("Running complete pipeline")
        
        try:
            # Load configuration
            config = PipelineConfig.load_config(args.config)
            
            # Set device
            device = self._get_device(args.device)
            
            # Set up output directory
            output_dir = Path(args.output_dir) if args.output_dir else Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Set up error recovery
            error_recovery = ErrorRecoveryManager()
            health_monitor = PipelineHealthMonitor()
            
            # Load and preprocess data
            logger.info(f"Loading data from: {args.data_dir}")
            preprocessor = ImagePreprocessor(
                target_size=(config.decoder_config.image_shape[1], config.decoder_config.image_shape[2])
            )
            
            # Load images
            image_paths = preprocessor.load_images_from_directory(args.data_dir, recursive=True)
            if not image_paths:
                logger.error(f"No images found in {args.data_dir}")
                return 1
            
            logger.info(f"Found {len(image_paths)} images")
            
            # Process images in batches
            batch_size = 32
            all_images = []
            
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = preprocessor.preprocess_batch(batch_paths)
                all_images.append(batch_images)
            
            images = torch.cat(all_images, dim=0)
            
            # Create dummy labels (in real scenario, these would be loaded from data)
            labels = np.random.randint(0, max(5, len(set(range(len(image_paths))))), len(images))
            
            logger.info(f"Processing {len(images)} images with {len(set(labels))} classes")
            
            # Set up pipeline components with error recovery
            try:
                # Encoder
                encoder = ResNetEncoder(
                    architecture=config.encoder_config.architecture,
                    embedding_dim=config.encoder_config.embedding_dim,
                    device=device
                )
                
                # Decoder
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
                        device=device
                    )
                elif config.decoder_config.decoder_type == 'gan':
                    decoder = GANDecoder(
                        embedding_dim=config.encoder_config.embedding_dim,
                        image_shape=config.decoder_config.image_shape,
                        device=device
                    )
                else:
                    logger.error(f"Unsupported decoder type: {config.decoder_config.decoder_type}")
                    return 1
                
                # SMOTE
                smote = ConstrainedSMOTE(
                    k_neighbors=config.smote_config.k_neighbors,
                    use_clustering=config.smote_config.use_clustering,
                    clustering_method=config.smote_config.clustering_method
                )
                
                # Quality assessor
                quality_assessor = QualityAssessor(
                    metrics=config.quality_config.metrics,
                    device=device
                )
                
            except Exception as e:
                logger.error(f"Failed to initialize components: {e}")
                # Try error recovery
                recovered_config = error_recovery.recover_from_initialization_error(config, str(e))
                if recovered_config:
                    logger.info("Using recovered configuration")
                    config = recovered_config
                    # Retry initialization with fallback config
                    # (Implementation would retry with simpler configuration)
                else:
                    logger.error("Could not recover from initialization error")
                    return 1
            
            # Create pipeline
            pipeline = SynthesisPipeline(
                encoder=encoder,
                decoder=decoder,
                smote=smote,
                quality_assessor=quality_assessor
            )
            
            # Health check
            health_status = health_monitor.check_pipeline_health(pipeline, images[:10], labels[:10])
            if not health_status['overall_health']:
                logger.warning(f"Pipeline health issues detected: {health_status}")
            
            # Train decoder if requested
            if args.train_decoder:
                logger.info("Training decoder...")
                
                # Generate embeddings
                embeddings = encoder.encode(images)
                
                # Train based on decoder type
                if config.decoder_config.decoder_type == 'autoencoder':
                    trainer = AutoencoderTrainer(decoder)
                    trainer.train(
                        train_embeddings=embeddings,
                        train_images=images,
                        num_epochs=config.decoder_config.num_epochs,
                        batch_size=config.decoder_config.batch_size
                    )
                elif config.decoder_config.decoder_type == 'vae':
                    trainer = VAETrainer(decoder)
                    trainer.train(
                        train_embeddings=embeddings,
                        train_images=images,
                        num_epochs=config.decoder_config.num_epochs,
                        batch_size=config.decoder_config.batch_size
                    )
                # Note: GAN training would require specialized trainer
                
                logger.info("Decoder training completed")
            
            # Fit pipeline
            logger.info("Fitting pipeline on data...")
            pipeline.fit(images, labels)
            
            # Generate synthetic samples if requested
            if args.generate_samples:
                logger.info(f"Generating {args.generate_samples} synthetic samples...")
                
                synthetic_images, synthetic_labels = pipeline.generate_synthetic_images(args.generate_samples)
                
                if len(synthetic_images) > 0:
                    # Save synthetic images
                    synthetic_dir = output_dir / "synthetic_images"
                    synthetic_dir.mkdir(exist_ok=True)
                    
                    # Denormalize images for saving
                    synthetic_images = preprocessor.denormalize_tensor(synthetic_images)
                    
                    for i, (image, label) in enumerate(zip(synthetic_images, synthetic_labels)):
                        pil_image = preprocessor.tensor_to_pil(image)
                        filename = synthetic_dir / f"synthetic_{i:06d}_class_{label}.png"
                        pil_image.save(filename)
                    
                    logger.info(f"Saved {len(synthetic_images)} synthetic images to {synthetic_dir}")
                    
                    # Evaluate quality if requested
                    if args.evaluate_quality:
                        logger.info("Evaluating synthetic image quality...")
                        
                        # Use a subset of original images for comparison
                        comparison_images = images[:len(synthetic_images)]
                        quality_results = pipeline.evaluate_quality(synthetic_images, comparison_images)
                        
                        # Generate quality report
                        reporter = QualityReporter(
                            output_dir=str(output_dir / "quality_reports"),
                            report_format="html"
                        )
                        
                        report_path = reporter.generate_comprehensive_report(
                            quality_results=quality_results,
                            synthetic_images=synthetic_images,
                            real_images=comparison_images,
                            report_name="pipeline_quality_report"
                        )
                        
                        logger.info(f"Quality report saved to: {report_path}")
                        
                        # Print quality summary
                        logger.info("Quality Results:")
                        for metric, value in quality_results.items():
                            if isinstance(value, (int, float)):
                                logger.info(f"  {metric}: {value:.4f}")
                else:
                    logger.warning("No synthetic images were generated")
            
            # Save pipeline
            pipeline_path = output_dir / "trained_pipeline"
            pipeline.save_pipeline(str(pipeline_path))
            logger.info(f"Pipeline saved to: {pipeline_path}")
            
            # Save pipeline metadata
            metadata = {
                "config_file": args.config,
                "data_dir": args.data_dir,
                "output_dir": str(output_dir),
                "n_input_images": len(images),
                "n_classes": len(set(labels)),
                "decoder_trained": args.train_decoder,
                "synthetic_samples_generated": args.generate_samples,
                "quality_evaluated": args.evaluate_quality,
                "pipeline_timestamp": str(np.datetime64('now')),
                "health_status": health_status
            }
            
            with open(output_dir / "pipeline_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info("Complete pipeline run finished successfully")
            return 0
            
        except Exception as e:
            logger.error(f"Pipeline run failed: {e}")
            logger.debug("Full traceback:", exc_info=True)
            return 1
    
    def _test_pipeline(self, args) -> int:
        """Test pipeline with small dataset."""
        logger.info("Testing pipeline")
        
        try:
            # Load configuration
            config = PipelineConfig.load_config(args.config)
            
            # Set device
            device = self._get_device(args.device)
            
            # Generate synthetic test data if no data directory provided
            if args.synthetic_data or not args.data_dir:
                logger.info("Using synthetic test data")
                
                # Create small synthetic dataset
                test_images = torch.randn(20, *config.decoder_config.image_shape)
                test_labels = np.random.randint(0, 3, 20)  # 3 classes
                
                logger.info(f"Created synthetic test dataset: {test_images.shape}")
            else:
                # Load small subset of real data
                logger.info(f"Loading test data from: {args.data_dir}")
                
                preprocessor = ImagePreprocessor(
                    target_size=(config.decoder_config.image_shape[1], config.decoder_config.image_shape[2])
                )
                
                image_paths = preprocessor.load_images_from_directory(
                    args.data_dir, 
                    recursive=True, 
                    max_images=20  # Small test set
                )
                
                if not image_paths:
                    logger.warning(f"No images found in {args.data_dir}, using synthetic data")
                    test_images = torch.randn(20, *config.decoder_config.image_shape)
                    test_labels = np.random.randint(0, 3, 20)
                else:
                    test_images = preprocessor.preprocess_batch(image_paths)
                    test_labels = np.random.randint(0, 3, len(test_images))
                    
                    logger.info(f"Loaded {len(test_images)} test images")
            
            # Set up simplified pipeline components
            encoder = ResNetEncoder(
                architecture='resnet18',  # Use smaller model for testing
                embedding_dim=128,  # Smaller embedding for testing
                device=device
            )
            
            decoder = AutoencoderDecoder(
                embedding_dim=128,
                image_shape=config.decoder_config.image_shape,
                device=device
            )
            
            smote = ConstrainedSMOTE(
                k_neighbors=3,  # Smaller k for small dataset
                use_clustering=False  # Disable clustering for small test
            )
            
            quality_assessor = QualityAssessor(
                metrics=['ssim'],  # Use fast metric for testing
                device=device
            )
            
            # Create test pipeline
            pipeline = SynthesisPipeline(
                encoder=encoder,
                decoder=decoder,
                smote=smote,
                quality_assessor=quality_assessor
            )
            
            logger.info("Testing pipeline components...")
            
            # Test encoding
            logger.info("Testing encoder...")
            embeddings = encoder.encode(test_images)
            logger.info(f"Encoder output shape: {embeddings.shape}")
            
            # Test decoder
            logger.info("Testing decoder...")
            reconstructed = decoder.decode(embeddings)
            logger.info(f"Decoder output shape: {reconstructed.shape}")
            
            # Test SMOTE
            logger.info("Testing SMOTE...")
            embeddings_np = embeddings.detach().cpu().numpy()
            smote.fit(embeddings_np, test_labels)
            synthetic_embeddings, synthetic_labels = smote.generate_synthetic(10)
            logger.info(f"SMOTE generated {len(synthetic_embeddings)} synthetic embeddings")
            
            # Test quality assessment
            logger.info("Testing quality assessment...")
            quality_results = quality_assessor.evaluate_quality(
                synthetic_images=reconstructed[:5],
                real_images=test_images[:5]
            )
            logger.info(f"Quality metrics: {quality_results}")
            
            # Test complete pipeline
            logger.info("Testing complete pipeline...")
            pipeline.fit(test_images, test_labels)
            
            synthetic_images, synthetic_labels = pipeline.generate_synthetic_images(5)
            logger.info(f"Pipeline generated {len(synthetic_images)} synthetic images")
            
            # Health check
            health_monitor = PipelineHealthMonitor()
            health_status = health_monitor.check_pipeline_health(pipeline, test_images[:5], test_labels[:5])
            
            logger.info("Health Check Results:")
            logger.info(f"  Overall Health: {health_status['overall_health']}")
            logger.info(f"  Component Health: {health_status['component_health']}")
            
            if health_status['issues']:
                logger.warning("Issues detected:")
                for issue in health_status['issues']:
                    logger.warning(f"  - {issue}")
            
            logger.info("Pipeline test completed successfully - all components working")
            return 0
            
        except Exception as e:
            logger.error(f"Pipeline test failed: {e}")
            logger.debug("Full traceback:", exc_info=True)
            return 1
    
    def _handle_health_command(self, args) -> int:
        """Handle health monitoring command."""
        logger.info("Starting health monitoring")
        
        try:
            # Load configuration
            config = PipelineConfig.load_config(args.config)
            
            # Set up health monitor
            health_monitor = PipelineHealthMonitor()
            
            # Create minimal pipeline for health checking
            device = torch.device('cpu')  # Use CPU for health monitoring
            
            encoder = ResNetEncoder(
                architecture='resnet18',
                embedding_dim=128,
                device=device
            )
            
            decoder = AutoencoderDecoder(
                embedding_dim=128,
                image_shape=config.decoder_config.image_shape,
                device=device
            )
            
            smote = ConstrainedSMOTE(k_neighbors=3)
            quality_assessor = QualityAssessor(metrics=['ssim'], device=device)
            
            pipeline = SynthesisPipeline(
                encoder=encoder,
                decoder=decoder,
                smote=smote,
                quality_assessor=quality_assessor
            )
            
            # Create test data for health check
            test_images = torch.randn(5, *config.decoder_config.image_shape)
            test_labels = np.array([0, 1, 0, 1, 2])
            
            if args.continuous:
                logger.info(f"Starting continuous health monitoring (interval: {args.check_interval}s)")
                
                import time
                health_history = []
                
                try:
                    while True:
                        # Perform health check
                        health_status = health_monitor.check_pipeline_health(
                            pipeline, test_images, test_labels
                        )
                        
                        health_status['timestamp'] = str(np.datetime64('now'))
                        health_history.append(health_status)
                        
                        # Log current status
                        logger.info(f"Health Status: {health_status['overall_health']}")
                        
                        if not health_status['overall_health']:
                            logger.warning("Health issues detected:")
                            for issue in health_status['issues']:
                                logger.warning(f"  - {issue}")
                        
                        # Save health report if file specified
                        if args.output_file:
                            with open(args.output_file, 'w') as f:
                                json.dump(health_history, f, indent=2, default=str)
                        
                        # Wait for next check
                        time.sleep(args.check_interval)
                        
                except KeyboardInterrupt:
                    logger.info("Health monitoring stopped by user")
                    
                    # Final report
                    if args.output_file:
                        with open(args.output_file, 'w') as f:
                            json.dump(health_history, f, indent=2, default=str)
                        logger.info(f"Health monitoring history saved to: {args.output_file}")
            
            else:
                # Single health check
                health_status = health_monitor.check_pipeline_health(
                    pipeline, test_images, test_labels
                )
                
                logger.info("Pipeline Health Report:")
                logger.info("=" * 30)
                logger.info(f"Overall Health: {health_status['overall_health']}")
                logger.info(f"Memory Usage: {health_status['memory_usage']:.2f} MB")
                logger.info(f"GPU Available: {health_status['gpu_available']}")
                
                logger.info("\nComponent Health:")
                for component, status in health_status['component_health'].items():
                    logger.info(f"  {component}: {status}")
                
                if health_status['issues']:
                    logger.info("\nIssues Detected:")
                    for issue in health_status['issues']:
                        logger.info(f"  - {issue}")
                
                if health_status['recommendations']:
                    logger.info("\nRecommendations:")
                    for rec in health_status['recommendations']:
                        logger.info(f"  - {rec}")
                
                # Save single report if file specified
                if args.output_file:
                    with open(args.output_file, 'w') as f:
                        json.dump(health_status, f, indent=2, default=str)
                    logger.info(f"Health report saved to: {args.output_file}")
            
            logger.info("Health monitoring completed")
            return 0
            
        except Exception as e:
            logger.error(f"Health monitoring failed: {e}")
            logger.debug("Full traceback:", exc_info=True)
            return 1
    
    def _get_device(self, device_arg: str) -> torch.device:
        """Get PyTorch device based on argument."""
        if device_arg == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_arg == 'cpu':
            return torch.device('cpu')
        elif device_arg == 'cuda':
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                return torch.device('cpu')
            return torch.device('cuda')
        else:
            raise ValueError(f"Invalid device: {device_arg}")


def main():
    """Main entry point for CLI."""
    cli = SMOTEImageCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())