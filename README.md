# SMOTE for Images - Advanced Synthetic Image Generation

A comprehensive implementation of SMOTE (Synthetic Minority Over-sampling Technique) extended for image data generation. This project addresses class imbalance in image datasets by generating high-quality synthetic images in the embedding space.

## 🚀 Features

### Core Components
- **Advanced Image Encoders**: ResNet-based encoders with fine-tuning capabilities
- **Multiple Decoder Architectures**: Autoencoder, VAE, GAN, and Diffusion decoders with progressive upsampling
- **Constrained SMOTE**: Enhanced SMOTE with semantic clustering and validation
- **Comprehensive Quality Assessment**: Multiple metrics including FID, LPIPS, SSIM, and diversity measures
- **Pipeline Orchestration**: End-to-end synthesis pipeline with memory management

### Key Capabilities
- ✅ **Semantic Clustering**: Maintains class structure during synthetic generation
- ✅ **Quality Validation**: Multi-metric assessment with automated reporting
- ✅ **Memory Efficient**: Batch processing with automatic memory management
- ✅ **Configurable Pipeline**: Flexible configuration system for all components
- ✅ **Multiple Architectures**: Support for various encoder/decoder combinations
- ✅ **Training Pipelines**: Complete training systems for autoencoder, VAE, GAN, and Diffusion models
- ✅ **Visualization**: Comprehensive quality reports with plots and analysis

## 📋 Requirements

```bash
pip install -r requirements.txt
```

### Dependencies
- PyTorch >= 1.9.0
- TorchVision >= 0.10.0
- NumPy >= 1.21.0
- Scikit-learn >= 1.0.0
- Imbalanced-learn >= 0.8.0
- Pillow >= 8.3.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0

## 🔧 Installation

```bash
# Clone the repository
git clone <repository-url>
cd smote-for-images

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## 🎯 Quick Start

### Basic Pipeline Usage

```python
import torch
import numpy as np
from smote_image_synthesis.pipeline import SynthesisPipeline
from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder
from smote_image_synthesis.decoders.autoencoder_decoder import AutoencoderDecoder
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE
from smote_image_synthesis.quality.assessor import QualityAssessor

# Set up components
encoder = ResNetEncoder(
    architecture='resnet50',
    embedding_dim=512,
    pretrained=True
)

decoder = AutoencoderDecoder(
    embedding_dim=512,
    image_shape=(3, 224, 224)
)

smote = ConstrainedSMOTE(
    k_neighbors=5,
    use_clustering=True,
    clustering_method='kmeans'
)

quality_assessor = QualityAssessor(
    metrics=['fid', 'lpips', 'ssim']
)

# Create pipeline
pipeline = SynthesisPipeline(
    encoder=encoder,
    decoder=decoder,
    smote=smote,
    quality_assessor=quality_assessor
)

# Fit on your data
images = torch.randn(100, 3, 224, 224)  # Your image data
labels = np.random.randint(0, 5, 100)   # Your labels

pipeline.fit(images, labels)

# Generate synthetic images
synthetic_images, synthetic_labels = pipeline.generate_synthetic_images(50)

# Evaluate quality
quality_results = pipeline.evaluate_quality(synthetic_images, images[:50])
print(quality_results)
```

### Using the Demo Script

```bash
# Run basic demo
python demo_pipeline.py

# Run with custom parameters
python demo_pipeline.py --n-samples 200 --decoder-type vae --train-decoder --generate-report

# See all options
python demo_pipeline.py --help
```

### Jupyter Notebook Examples

The repository includes three example notebooks:

1. `01_basic_pipeline_usage.ipynb` - Basic pipeline usage
2. `02_decoder_architectures.ipynb` - Comparison of different decoder architectures
3. `03_custom_dataset_integration.ipynb` - Guide for integrating custom datasets

## 🏗️ Architecture

### Pipeline Components

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Images    │───▶│   Encoder    │───▶│ Embeddings  │
└─────────────┘    └──────────────┘    └─────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Synthetic   │◀───│   Decoder    │◀───│ Constrained │
│   Images    │    └──────────────┘    │    SMOTE    │
└─────────────┘                        └─────────────┘
      │                                       │
      ▼                                       │
┌─────────────┐    ┌──────────────┐         │
│  Quality    │───▶│   Reports    │         │
│ Assessment  │    │ & Analysis   │         │
└─────────────┘    └──────────────┘         │
                                             │
                   ┌─────────────┐         │
                   │ Clustering  │◀────────┘
                   │ Validation  │
                   └─────────────┘
```

### Supported Architectures

#### Encoders
- **ResNet Encoder**: ResNet18/50/101 with configurable embedding dimensions
  - Pretrained ImageNet weights
  - Fine-tuning capabilities
  - Backbone freezing options

#### Decoders
- **Autoencoder Decoder**: Progressive upsampling with skip connections
  - Configurable hidden dimensions
  - Perceptual loss integration
  - Memory-efficient training

- **VAE Decoder**: Variational autoencoder with reparameterization trick
  - KL divergence regularization
  - Latent space interpolation
  - Beta-VAE support

- **GAN Decoder**: Generative Adversarial Network with spectral normalization
  - Feature matching loss
  - Progressive training
  - Self-attention mechanisms

- **Diffusion Decoder**: Denoising diffusion probabilistic model
  - U-Net architecture
  - DDPM/DDIM sampling
  - Exponential moving average

#### SMOTE Enhancements
- **Semantic Clustering**: K-means, DBSCAN, hierarchical clustering
- **Distance Thresholding**: Ensures valid interpolation regions
- **Outlier Detection**: Filters invalid synthetic samples
- **Boundary Detection**: Maintains class separation

## 📊 Quality Assessment

### Supported Metrics
- **FID**: Fréchet Inception Distance
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **SSIM**: Structural Similarity Index
- **PSNR**: Peak Signal-to-Noise Ratio
- **MSE/MAE**: Mean Squared/Absolute Error

### Diversity Metrics
- Pairwise distance analysis
- Intra-class diversity
- Distribution similarity tests

### Reporting
- HTML reports with visualizations
- CSV export for metrics
- Statistical analysis plots
- Image comparison grids

## ⚙️ Configuration

### Pipeline Configuration

```python
from smote_image_synthesis.data.models import PipelineConfig

config = PipelineConfig(
    config_name="my_experiment",
    encoder_config={
        'architecture': 'resnet50',
        'embedding_dim': 512,
        'pretrained': True,
        'freeze_backbone': False
    },
    decoder_config={
        'decoder_type': 'autoencoder',
        'learning_rate': 0.001,
        'num_epochs': 100,
        'use_perceptual_loss': True
    },
    smote_config={
        'k_neighbors': 5,
        'use_clustering': True,
        'clustering_method': 'kmeans',
        'distance_threshold': 0.5
    },
    quality_config={
        'metrics': ['fid', 'lpips', 'ssim'],
        'compute_diversity': True
    }
)

# Save configuration
config.save_config('experiment_config.json')

# Load configuration
loaded_config = PipelineConfig.load_config('experiment_config.json')
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_integration.py -v
python -m pytest tests/test_data_models.py -v

# Run with coverage
python -m pytest tests/ --cov=smote_image_synthesis --cov-report=html
```

## 📚 Examples

### Training Custom Decoders

```python
from smote_image_synthesis.decoders.autoencoder_trainer import AutoencoderTrainer
from smote_image_synthesis.decoders.vae_trainer import VAETrainer
from smote_image_synthesis.decoders.gan_trainer import GANTrainer
from smote_image_synthesis.decoders.diffusion_trainer import DiffusionTrainer

# Train Autoencoder
trainer = AutoencoderTrainer(
    decoder=autoencoder_decoder,
    learning_rate=0.001,
    use_perceptual_loss=True
)

history = trainer.train(
    train_embeddings=train_embeddings,
    train_images=train_images,
    val_embeddings=val_embeddings,
    val_images=val_images,
    num_epochs=100
)

# Train VAE
vae_trainer = VAETrainer(
    vae_decoder=vae_decoder,
    beta=1.0,  # Beta-VAE parameter
    learning_rate=0.001
)

vae_history = vae_trainer.train(
    train_embeddings=train_embeddings,
    train_images=train_images,
    num_epochs=200
)

# Train GAN
gan_trainer = GANTrainer(
    gan_decoder=gan_decoder,
    generator_lr=0.0002,
    discriminator_lr=0.0002
)

gan_history = gan_trainer.train(
    train_embeddings=train_embeddings,
    train_images=train_images,
    num_epochs=100
)

# Train Diffusion Model
diffusion_trainer = DiffusionTrainer(
    diffusion_decoder=diffusion_decoder,
    learning_rate=1e-4
)

diffusion_history = diffusion_trainer.train(
    train_embeddings=train_embeddings,
    train_images=train_images,
    num_epochs=100
)
```

### Advanced SMOTE Configuration

```python
smote = ConstrainedSMOTE(
    k_neighbors=5,
    sampling_strategy='auto',
    use_clustering=True,
    clustering_method='kmeans',
    cluster_validation_threshold=0.7,
    semantic_coherence_threshold=0.8,
    boundary_detection_method='density',
    outlier_detection_threshold=0.1,
    manifold_validation=True,
    normalize_embeddings=True
)

# Fit and generate with metadata
smote.fit(embeddings, labels)
synthetic_embeddings, synthetic_labels, metadata = smote.generate_synthetic(
    n_samples=100,
    return_metadata=True
)

# Validate embedding space
is_valid, report = smote.validate_embedding_space(test_embeddings)
print(f"Validation: {is_valid}")
print(f"Report: {report}")
```

### Quality Assessment and Reporting

```python
from smote_image_synthesis.quality.reporter import QualityReporter

# Comprehensive quality assessment
quality_results = assessor.evaluate_quality(
    synthetic_images=synthetic_images,
    real_images=real_images,
    return_detailed=True
)

# Generate comprehensive report
reporter = QualityReporter(
    output_dir='./quality_reports',
    report_format='html'
)

report_path = reporter.generate_comprehensive_report(
    quality_results=quality_results,
    synthetic_images=synthetic_images,
    real_images=real_images,
    report_name='experiment_001'
)

print(f"Report saved to: {report_path}")
```

## 📈 Performance Tips

### Memory Management
- Use batch processing for large datasets
- Enable memory management in decoders
- Use CPU for preprocessing when GPU memory is limited

### Training Optimization
- Start with smaller architectures (ResNet18) for prototyping
- Use mixed precision training for larger models
- Implement gradient accumulation for large batch sizes

### Quality vs Speed Trade-offs
- Use fewer quality metrics for faster evaluation
- Reduce FID batch size for memory constraints
- Disable perceptual loss for faster decoder training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run tests and ensure they pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r test_requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests before committing
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of PyTorch and TorchVision
- Uses imbalanced-learn for SMOTE implementation
- Inspired by research in synthetic data generation
- Quality metrics implementation based on established computer vision metrics

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in the `docs/` directory
- Review the example scripts and tests

---

**Note**: This implementation is designed for research and educational purposes. For production use, please validate the quality and appropriateness of synthetic images for your specific use case.