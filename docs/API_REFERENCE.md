# SMOTE for Images - API Reference

This document provides comprehensive API documentation for all components of the SMOTE for Images library.

## Table of Contents

- [Pipeline](#pipeline)
- [Encoders](#encoders)
- [Decoders](#decoders)
- [SMOTE](#smote)
- [Quality Assessment](#quality-assessment)
- [Data Models](#data-models)
- [Utilities](#utilities)

---

## Pipeline

### SynthesisPipeline

Main orchestrator class that coordinates all pipeline components.

```python
from smote_image_synthesis.pipeline import SynthesisPipeline
```

#### Constructor

```python
SynthesisPipeline(
    encoder: ImageEncoder,
    decoder: BaseDecoder,
    smote: ConstrainedSMOTE,
    quality_assessor: Optional[QualityAssessor] = None,
    device: Optional[torch.device] = None,
    memory_efficient: bool = True
)
```

**Parameters:**
- `encoder`: Image encoder for converting images to embeddings
- `decoder`: Decoder for converting embeddings back to images
- `smote`: SMOTE instance for synthetic embedding generation
- `quality_assessor`: Optional quality assessment component
- `device`: Computation device (auto-detected if None)
- `memory_efficient`: Enable memory management features

#### Methods

##### fit()

```python
fit(
    images: torch.Tensor,
    labels: np.ndarray,
    train_decoder: bool = True,
    decoder_epochs: int = 100,
    validation_data: Optional[Tuple[torch.Tensor, np.ndarray]] = None,
    batch_size: int = 32
) -> None
```

Fit the pipeline on training data.

##### generate_synthetic_images()

```python
generate_synthetic_images(
    n_samples: int,
    target_classes: Optional[List[int]] = None,
    return_metadata: bool = False
) -> Union[Tuple[torch.Tensor, np.ndarray], 
           Tuple[torch.Tensor, np.ndarray, Dict]]
```

Generate synthetic images using SMOTE.

##### evaluate_quality()

```python
evaluate_quality(
    synthetic_images: torch.Tensor,
    real_images: torch.Tensor,
    return_detailed: bool = False
) -> Dict[str, Any]
```

Evaluate quality of synthetic images.

---

## Encoders

### ResNetEncoder

ResNet-based image encoder with configurable architecture.

```python
from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder
```

#### Constructor

```python
ResNetEncoder(
    architecture: str = 'resnet50',
    embedding_dim: int = 512,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.1,
    use_batch_norm: bool = True
)
```

---

## Decoders

### AutoencoderDecoder

Simple autoencoder decoder with progressive upsampling.

```python
from smote_image_synthesis.decoders.autoencoder_decoder import AutoencoderDecoder
```

### VAEDecoder

Variational Autoencoder decoder with probabilistic latent space.

```python
from smote_image_synthesis.decoders.vae_decoder import VAEDecoder
```

### GANDecoder

Generative Adversarial Network decoder.

```python
from smote_image_synthesis.decoders.gan_decoder import GANDecoder
```

### DiffusionDecoder

Diffusion model decoder for high-quality generation.

```python
from smote_image_synthesis.decoders.diffusion_decoder import DiffusionDecoder
```

---

## SMOTE

### ConstrainedSMOTE

Enhanced SMOTE with semantic clustering and validation.

```python
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE
```

---

## Quality Assessment

### QualityAssessor

Comprehensive quality assessment for synthetic images.

```python
from smote_image_synthesis.quality.assessor import QualityAssessor
```

**Supported Metrics:**
- `fid`: Fréchet Inception Distance
- `lpips`: Learned Perceptual Image Patch Similarity
- `ssim`: Structural Similarity Index
- `ms_ssim`: Multi-Scale SSIM
- `psnr`: Peak Signal-to-Noise Ratio
- `mse`: Mean Squared Error
- `mae`: Mean Absolute Error

---

## Data Models

### PipelineConfig

Configuration management for the entire pipeline.

```python
from smote_image_synthesis.data.models import PipelineConfig
```

### EmbeddingData

Data model for embedding storage and validation.

```python
from smote_image_synthesis.data.models import EmbeddingData
```

### SyntheticSample

Data model for synthetic sample tracking.

```python
from smote_image_synthesis.data.models import SyntheticSample
```

---

## Example Usage

### Basic Pipeline

```python
import torch
import numpy as np
from smote_image_synthesis.pipeline import SynthesisPipeline
from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder
from smote_image_synthesis.decoders.autoencoder_decoder import AutoencoderDecoder
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE
from smote_image_synthesis.quality.assessor import QualityAssessor

# Create components
encoder = ResNetEncoder(architecture='resnet50', embedding_dim=512)
decoder = AutoencoderDecoder(embedding_dim=512, image_shape=(3, 224, 224))
smote = ConstrainedSMOTE(k_neighbors=5, use_clustering=True)
quality_assessor = QualityAssessor(metrics=['fid', 'ssim'])

# Create pipeline
pipeline = SynthesisPipeline(
    encoder=encoder,
    decoder=decoder,
    smote=smote,
    quality_assessor=quality_assessor
)

# Fit and generate
images = torch.randn(100, 3, 224, 224)
labels = np.random.randint(0, 5, 100)

pipeline.fit(images, labels)
synthetic_images, synthetic_labels = pipeline.generate_synthetic_images(50)
```