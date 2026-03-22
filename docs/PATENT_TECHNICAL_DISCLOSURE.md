# TECHNICAL DISCLOSURE DOCUMENT
## System and Method for Class-Conditional Synthetic Image Generation via Spherical Interpolation in a Jointly Trained Adversarial Embedding Space

**Document Type:** Invention Disclosure / Patent Application Support
**Date:** 2026-03-15
**Status:** Confidential — Prior to Patent Filing

---

## TABLE OF CONTENTS

1. [Field of the Invention](#1-field-of-the-invention)
2. [Background and Problem Statement](#2-background-and-problem-statement)
3. [Summary of the Invention](#3-summary-of-the-invention)
4. [Detailed Description of the Architecture](#4-detailed-description-of-the-architecture)
   - 4.1 System Overview
   - 4.2 Image Encoder Module
   - 4.3 Spherical SMOTE Oversampling Module
   - 4.4 Class-Conditional DCGAN Decoder
   - 4.5 End-to-End Adversarial Training Pipeline
   - 4.6 Exponential Moving Average Inference Mechanism
   - 4.7 Discriminator with Feature Matching
   - 4.8 Perceptual Loss Module
   - 4.9 Quality Assessment Module
   - 4.10 General-Purpose Dataset Interface
5. [Novel Contributions and Claims](#5-novel-contributions-and-claims)
6. [Mathematical Formulations](#6-mathematical-formulations)
7. [Data Flow and Algorithms](#7-data-flow-and-algorithms)
8. [Implementation Details](#8-implementation-details)
9. [Experimental Configuration](#9-experimental-configuration)
10. [Advantages Over Prior Art](#10-advantages-over-prior-art)

---

## 1. FIELD OF THE INVENTION

This invention relates to systems and methods for **synthetic image generation for class-imbalanced datasets**, specifically to a machine learning pipeline that combines:

- Deep convolutional neural network encoding of images into a structured embedding manifold
- Spherical linear interpolation (SLERP) in a hyperspherical embedding space as a replacement for the conventional linear interpolation used in Synthetic Minority Oversampling Technique (SMOTE)
- Class-conditional image decoding using a Deep Convolutional Generative Adversarial Network (DCGAN) architecture augmented with self-attention mechanisms and class embedding injection
- End-to-end joint adversarial training with Wasserstein distance and gradient penalty (WGAN-GP)
- Exponential Moving Average (EMA) of generator weights applied at inference time
- Multi-scale feature matching loss against discriminator intermediate representations

The invention is applicable to any supervised image classification problem where the training dataset suffers from class imbalance, including but not limited to medical imaging, satellite imagery, industrial defect detection, wildlife monitoring, and biometric identification.

---

## 2. BACKGROUND AND PROBLEM STATEMENT

### 2.1 The Class Imbalance Problem

Class imbalance — where some classes contain substantially fewer examples than others — is a pervasive challenge in supervised machine learning applied to image classification. In many real-world domains (e.g., rare disease detection, fraud detection in surveillance video, detection of rare manufacturing defects), the ratio of majority to minority class samples may exceed 100:1. Standard classifiers trained on such data exhibit strong bias toward majority classes, resulting in poor sensitivity on minority classes.

### 2.2 Limitations of Existing Approaches

**Tabular SMOTE (Chawla et al., 2002):** The original SMOTE algorithm addresses class imbalance by synthesising new minority-class samples through **linear interpolation** between existing samples and their k-nearest neighbours in the raw feature space. While effective for low-dimensional tabular data, this approach fails critically when applied directly to high-dimensional image pixel spaces because:
  - Linear interpolation of pixel values produces blurry, perceptually unrealistic images
  - The pixel manifold is not locally linear; real images occupy a low-dimensional nonlinear manifold within the high-dimensional pixel space
  - No semantic constraints are imposed, so interpolated images may be implausible

**GAN-based Augmentation:** Generative Adversarial Networks (Goodfellow et al., 2014) can generate visually realistic images, but standard GAN training:
  - Does not directly target class imbalance; training a separate GAN per minority class is resource-prohibitive
  - Does not guarantee that generated images will occupy the same semantic region as the target class
  - Suffers from mode collapse, training instability, and poor diversity

**Variational Autoencoder (VAE) Approaches:** VAEs provide a latent space for interpolation but:
  - The Gaussian prior assumption does not match the geometry of real image embeddings
  - Reconstruction quality is typically inferior to GAN-based methods due to the MSE/ELBO objective blurring fine details
  - Linear interpolation in the VAE latent space still fails to respect the curved geometry of the data manifold

### 2.3 Gap in the Prior Art

No prior art simultaneously addresses: (a) manifold-faithful interpolation in an **adversarially trained** embedding space, (b) **class-conditional** synthesis during the interpolation-to-image mapping, (c) use of **spherical geometry** (SLERP) reflecting the L2-normalized embedding manifold, and (d) **EMA-stabilised** inference weights applied post-training. The present invention fills this gap.

---

## 3. SUMMARY OF THE INVENTION

The invention provides a **five-stage pipeline** that:

1. **Encodes** real images into an L2-normalised hyperspherical embedding space using a pretrained ResNet backbone with a learned projection head
2. **Oversamples** minority-class embeddings using a novel **SLERP-SMOTE** algorithm that performs spherical linear interpolation along geodesics of the unit hypersphere, preserving the manifold structure of the embedding space
3. **Decodes** both real and synthetic embeddings back to image space using a class-conditional DCGAN generator incorporating self-attention at intermediate spatial resolutions
4. **Trains** the encoder and decoder jointly end-to-end using a phased adversarial objective: pure reconstruction in the first 30% of training epochs, followed by WGAN-GP adversarial training with feature matching loss for the remaining 70%
5. **Applies** Exponential Moving Average (EMA) of generator weights at the conclusion of training, replacing instantaneous weights with their temporally smoothed counterparts for higher-quality inference

The system is fully general: it accepts any image dataset organised into per-class folders, automatically determines class counts and imbalance ratios, and produces a balanced augmented dataset alongside quality assessment metrics.

---

## 4. DETAILED DESCRIPTION OF THE ARCHITECTURE

### 4.1 System Overview

The system comprises six principal modules arranged in a sequential pipeline:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   SMOTE IMAGE SYNTHESIS PIPELINE                         │
│                                                                          │
│  ┌──────────┐    ┌───────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Dataset  │    │  ResNet   │    │   SLERP-     │    │ Class-Cond.  │  │
│  │  Loader   │───▶│ Encoder   │───▶│   SMOTE      │───▶│   DCGAN      │  │
│  │(any class │    │(L2-norm.) │    │ (Geodesic    │    │  Decoder     │  │
│  │ folder)   │    │           │    │ interp.)     │    │ (EMA weights)│  │
│  └──────────┘    └─────┬─────┘    └──────────────┘    └──────┬───────┘  │
│                        │                                       │          │
│                        │   ┌──────────────────────────────┐   │          │
│                        └──▶│  End-to-End Joint Training   │◀──┘          │
│                            │  Phase 1: MSE + L1 + Percep. │             │
│                            │  Phase 2: + WGAN-GP + FeatM. │             │
│                            │  Post:    EMA weights applied │             │
│                            └──────────────────────────────┘             │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Quality Assessor: FID, LPIPS, SSIM, PSNR, MSE, Diversity Index   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

**Source files:**
- `smote_image_synthesis/pipeline.py` — `SynthesisPipeline` orchestrator
- `smote_image_synthesis/encoders/resnet_encoder.py` — `ResNetEncoder`
- `smote_image_synthesis/smote/constrained_smote.py` — `ConstrainedSMOTE`
- `smote_image_synthesis/decoders/dcgan_decoder.py` — `DCGANDecoder`
- `smote_image_synthesis/decoders/autoencoder_trainer.py` — `PerceptualLoss`
- `smote_image_synthesis/quality/assessor.py` — `QualityAssessor`
- `run_pipeline.py` — General-purpose dataset interface

---

### 4.2 Image Encoder Module

**File:** `smote_image_synthesis/encoders/resnet_encoder.py`
**Class:** `ResNetEncoder`

#### 4.2.1 Architecture

The encoder transforms a raw image tensor of shape `[B, C, H, W]` into a compact embedding vector of shape `[B, D]` where `D` is the configurable embedding dimension (default 512).

The architecture consists of three sequential components:

**Component 1 — ResNet Backbone:**
One of three supported ResNet variants is loaded with optional ImageNet pretraining:
- `ResNet18`: 11.7M parameters, 512 features at global average pooling output
- `ResNet50`: 25.6M parameters, 2048 features at global average pooling output
- `ResNet101`: 44.5M parameters, 2048 features at global average pooling output

The final classification layer (`fc`) is removed, leaving the backbone to output a spatial feature tensor that is subsequently flattened to `[B, num_features]` via global average pooling.

**Component 2 — Embedding Projection Head:**
A learned head that maps backbone features to the target embedding dimension:
```
Dropout(p=0.1)
→ Linear(num_features, D)
→ BatchNorm1d(D)
→ ReLU(inplace=True)
→ L2Normalize()           [NOVEL: projects onto unit hypersphere]
```

**Component 3 — L2 Normalisation Layer (`_L2Normalize`):**
A novel addition unique to this architecture. Every embedding vector is projected onto the unit hypersphere by dividing by its L2 norm:

```python
class _L2Normalize(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)
```

This normalisation is the **critical prerequisite** for the SLERP interpolation module (Section 4.3). When all embeddings have unit norm, SLERP becomes a pure geodesic interpolation along the great-circle arc of the hypersphere, which is the mathematically optimal path through the embedding manifold.

#### 4.2.2 Pretrained Weight Loading

Pretrained weights are loaded using the modern non-deprecated torchvision API:

```python
_WEIGHTS_MAP = {
    'resnet18':  'ResNet18_Weights',
    'resnet50':  'ResNet50_Weights',
    'resnet101': 'ResNet101_Weights',
}
weights_cls = getattr(torchvision.models, _WEIGHTS_MAP[architecture])
backbone = resnet_class(weights=weights_cls.IMAGENET1K_V1)
```

#### 4.2.3 Configurable Parameters

| Parameter | Default | Description |
|---|---|---|
| `architecture` | `'resnet18'` | Backbone variant |
| `embedding_dim` | `512` | Output embedding dimension D |
| `pretrained` | `True` | Load ImageNet-1K weights |
| `freeze_backbone` | `False` | Whether backbone gradients flow |
| `dropout_rate` | `0.1` | Dropout probability in projection head |
| `normalize_output` | `True` | L2-normalise output (recommended) |

During end-to-end training, `freeze_backbone=False` allows all backbone parameters to be jointly optimised with the decoder, enabling task-specific fine-tuning of the ResNet features.

---

### 4.3 Spherical SMOTE Oversampling Module

**File:** `smote_image_synthesis/smote/constrained_smote.py`
**Class:** `ConstrainedSMOTE`

#### 4.3.1 Motivation for SLERP

Standard SMOTE performs linear interpolation:

```
z_synthetic = (1 - t) * z_i + t * z_j,   t ~ Uniform(0, 1)
```

In a high-dimensional Euclidean space, this creates points along the straight-line chord between z_i and z_j. However, when embeddings are constrained to the unit hypersphere (as in our encoder), the straight chord passes through the **interior** of the sphere — these interior points are not valid embedding vectors, as they have norms less than 1. The set of valid embeddings lies on the surface of the sphere, and the natural path between two points on a sphere is the **great-circle arc**, parameterised by spherical linear interpolation.

#### 4.3.2 SLERP Algorithm

**Function:** `ConstrainedSMOTE._slerp(v0, v1, t)`

Given two embedding vectors v₀ and v₁ and interpolation weight t ∈ [0, 1]:

1. Compute unit vectors: `u₀ = v₀/‖v₀‖`, `u₁ = v₁/‖v₁‖`
2. Compute angular distance: `ω = arccos(clip(u₀ · u₁, -1, 1))`
3. If `|ω| < ε` (nearly parallel), fall back to linear interpolation of unit vectors
4. Otherwise, compute spherical interpolation:
   ```
   interp_unit = sin((1-t)·ω)/sin(ω) · u₀  +  sin(t·ω)/sin(ω) · u₁
   ```
5. Scale by linearly interpolated magnitude:
   ```
   interp_norm = (1-t)·‖v₀‖ + t·‖v₁‖
   z_synthetic = interp_unit · interp_norm
   ```

When `normalize_output=True` in the encoder, `‖v₀‖ = ‖v₁‖ = 1`, so step 5 reduces to `interp_norm = 1` — the synthetic embedding stays on the unit hypersphere.

```python
@staticmethod
def _slerp(v0, v1, t):
    n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
    if n0 < 1e-8 or n1 < 1e-8:
        return (1.0 - t) * v0 + t * v1
    u0, u1 = v0 / n0, v1 / n1
    dot = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
    omega = np.arccos(dot)
    if abs(omega) < 1e-6:
        interp_unit = (1.0 - t) * u0 + t * u1
        norm_iu = np.linalg.norm(interp_unit)
        if norm_iu > 1e-8:
            interp_unit = interp_unit / norm_iu
    else:
        s = np.sin(omega)
        interp_unit = (np.sin((1.0-t)*omega)/s * u0
                      + np.sin(t*omega)/s * u1)
    return interp_unit * ((1.0 - t) * n0 + t * n1)
```

#### 4.3.3 SLERP-SMOTE Generation Algorithm

**Function:** `ConstrainedSMOTE._generate_slerp(work_embeddings, n_samples)`

For each class c in the dataset:

1. Extract class embeddings: `E_c = {z_i : label_i = c}`
2. Fit a k-nearest-neighbour graph on `E_c` with k = `min(k_neighbors, |E_c| - 1)`
3. For each of `per_class = ceil(n_samples / num_classes)` synthetic samples:
   a. Sample anchor index i uniformly from `{0, ..., |E_c| - 1}`
   b. Sample neighbour column j uniformly from `{1, ..., k}` (excluding self at column 0)
   c. Sample interpolation weight t ~ Uniform(0, 1)
   d. Compute `z_synthetic = SLERP(E_c[i], E_c[nn_idx[i, j]], t)`
4. Collect all synthetics and trim to exactly `n_samples` via uniform random subsampling

This procedure guarantees:
- **Exact count**: ceiling arithmetic followed by trimming always produces the exact number of requested samples
- **Class balance**: samples are distributed evenly across all classes
- **Manifold fidelity**: all synthetic embeddings lie on the embedding manifold by construction (on the unit sphere for L2-normalised encoders)

#### 4.3.4 Constraint Mechanisms

The `ConstrainedSMOTE` module applies multiple validation constraints:

**Semantic Clustering:** For each class, k-means (or DBSCAN / GMM / Agglomerative) clustering is applied. Cluster membership is used to validate that interpolation occurs within semantically coherent regions. The optimal number of clusters is determined via the elbow method on the inertia curve.

**Outlier Detection:** Per-class outlier detectors are fitted using:
- `IsolationForest` (for high-dimensional embedding spaces)
- `OneClassSVM` (for compact, well-separated classes)
- `LocalOutlierFactor` (density-based, for irregular shapes)

Synthetic samples falling outside the learned decision boundary are filtered.

**Distance Thresholding:** An optional maximum-distance filter removes synthetic samples whose nearest-neighbour distance in the original space exceeds a configurable threshold, preventing extrapolation beyond the training distribution.

**StandardScaler Normalisation:** When `normalize_embeddings=True` (used when encoder L2 normalisation is disabled), a `StandardScaler` is fitted on the training embeddings. SMOTE operates in the standardised space and outputs are inverse-transformed to the original scale. This scaler round-trip is fully preserved in both the SLERP and linear paths.

#### 4.3.5 Configurable Parameters

| Parameter | Default | Description |
|---|---|---|
| `k_neighbors` | `5` | Neighbours in k-NN graph |
| `use_slerp` | `True` | Use geodesic (SLERP) interpolation |
| `normalize_embeddings` | `False`* | Apply StandardScaler |
| `use_clustering` | `True` | Apply clustering constraints |
| `clustering_method` | `'kmeans'` | Clustering algorithm |
| `outlier_detection_threshold` | `0.1` | Contamination rate for outlier detectors |
| `max_distance_threshold` | `None` | Maximum valid interpolation distance |
| `random_state` | `42` | Reproducibility seed |

*Set to `False` when `normalize_output=True` on the encoder, since L2 normalisation already constrains embeddings to the unit sphere.

---

### 4.4 Class-Conditional DCGAN Decoder

**File:** `smote_image_synthesis/decoders/dcgan_decoder.py`
**Class:** `DCGANDecoder`

#### 4.4.1 Overview

The decoder maps an embedding vector (plus optional class label) to a full-resolution RGB image. It is a DCGAN-style convolutional generator augmented with three novel components: (a) class embedding injection, (b) self-attention at 16×16 spatial resolution, and (c) zero-class fallback for unconditional generation.

#### 4.4.2 Class Conditioning via Embedding Injection

The decoder accepts a class label as an auxiliary input. A learnable embedding table maps discrete class indices to dense class embedding vectors:

```
class_embed: nn.Embedding(num_classes, class_embed_dim=64)
```

The class embedding is concatenated to the image embedding **before** the first linear projection layer:

```
z_conditioned = concat([z_image, class_embed(label)],  dim=1)
                       \_______/  \_________________/
                      embedding_dim  class_embed_dim
```

This concatenation expands the effective input dimension from `D` to `D + 64`. The first Linear layer of the generator is sized accordingly.

When no class label is provided (unconditional generation), a zero vector of dimension `class_embed_dim` is substituted:
```python
c = torch.zeros(batch_size, class_embed.embedding_dim, device=z.device)
```
This implements a learnable "null class" prior, enabling backward-compatible unconditional decoding.

The class conditioning module is implemented as a custom `_Generator` module:

```python
class _Generator(nn.Module):
    def __init__(self, main: nn.Sequential, class_embed: Optional[nn.Embedding]):
        super().__init__()
        self.main = main
        self.class_embed = class_embed

    def forward(self, z, labels=None):
        if self.class_embed is not None:
            if labels is not None:
                c = self.class_embed(labels.to(z.device).long())
            else:
                c = torch.zeros(z.size(0), self.class_embed.embedding_dim, device=z.device)
            z = torch.cat([z, c], dim=1)
        return self.main(z)
```

#### 4.4.3 Generator Architecture (for 64×64 output, base_channels=512)

```
Input: z [B, 512]  + class_embed [B, 64]  →  z_cond [B, 576]

Linear(576, 512*4*4, bias=False)          →  [B, 8192]
BatchNorm1d(8192)
ReLU(inplace=True)
Reshape                                    →  [B, 512, 4, 4]

ConvTranspose2d(512→256, 4×4, stride=2, pad=1, bias=False)   →  [B, 256, 8, 8]
BatchNorm2d(256)
ReLU(inplace=True)

ConvTranspose2d(256→128, 4×4, stride=2, pad=1, bias=False)   →  [B, 128, 16, 16]
BatchNorm2d(128)
ReLU(inplace=True)
SelfAttention2d(128)                      ← NOVEL: inserted at 16×16

ConvTranspose2d(128→64, 4×4, stride=2, pad=1, bias=False)    →  [B, 64, 32, 32]
BatchNorm2d(64)
ReLU(inplace=True)

ConvTranspose2d(64→3, 4×4, stride=2, pad=1, bias=True)       →  [B, 3, 64, 64]
Tanh()
```

**For 32×32 output** (3 upsampling steps):
```
Linear(576, 512*4*4) → Reshape(512,4,4)
→ DeConv(512→256, 4×4, 2×) → 8×8
→ DeConv(256→128, 4×4, 2×) → 16×16  + SelfAttention2d(128)
→ DeConv(128→3,   4×4, 2×) → 32×32
```

The number of upsampling steps is determined dynamically: `n_up = log₂(H / 4)` where H is the target image height.

#### 4.4.4 Self-Attention Mechanism (SAGAN-style)

**Class:** `SelfAttention2d`

A non-local self-attention block (Zhang et al., 2019) is inserted at the 16×16 spatial resolution. This is the first spatial resolution large enough to capture meaningful long-range dependencies, making it the optimal location for attention without prohibitive memory cost.

**Architecture:**
```
Input: x  [B, C, H, W]   (here C=128, H=W=16)

q = Conv2d(C, C//8, 1×1, bias=False)  →  [B, C/8, H, W]
k = Conv2d(C, C//8, 1×1, bias=False)  →  [B, C/8, H, W]
v = Conv2d(C, C,   1×1, bias=False)   →  [B, C, H, W]

q = reshape(B, C/8, H*W).permute(0,2,1)  →  [B, H*W, C/8]  (query)
k = reshape(B, C/8, H*W)                  →  [B, C/8, H*W]  (key)
attn = softmax(bmm(q, k) / sqrt(C/8))    →  [B, H*W, H*W]  (attention map)

v = reshape(B, C, H*W)                   →  [B, C, H*W]
out = bmm(v, attn.T).reshape(B, C, H, W) →  [B, C, H, W]

return x + γ · out                       (residual with learned scale γ)
```

The learnable scalar `γ` is initialised to 0, so the attention block begins as an identity transformation and gradually learns to apply attention corrections during training. This stabilises early training when the attention map is uninformative.

**Mathematical effect:** The attention map `attn[b, i, j]` captures the relevance of spatial location j to the feature synthesis at location i. This allows the generator to enforce global spatial consistency — for example, ensuring that a cat's eye and ear are coherently positioned relative to each other — which purely local convolutional operations cannot.

#### 4.4.5 Weight Initialisation

All weights are initialised following the DCGAN specification:
- `ConvTranspose2d`, `Conv2d`: `N(0, 0.02²)`
- `BatchNorm1d`, `BatchNorm2d`: weight `N(1.0, 0.02²)`, bias `0`
- `Linear`: `N(0, 0.02²)`, bias `0`
- `nn.Embedding` (class): `N(0, 0.02²)`

#### 4.4.6 Configurable Parameters

| Parameter | Default | Description |
|---|---|---|
| `embedding_dim` | `512` | Input embedding dimension |
| `image_shape` | `(3,64,64)` | Output image (C, H, W) |
| `base_channels` | `256` | Widest feature map channel count |
| `num_classes` | `0` | Number of classes (0 = unconditional) |
| `class_embed_dim` | `64` | Dimension of class embedding |
| `use_self_attention` | `True` | Insert SelfAttention2d at 16×16 |

---

### 4.5 End-to-End Adversarial Training Pipeline

**File:** `smote_image_synthesis/pipeline.py`
**Method:** `SynthesisPipeline._train_end_to_end()`

#### 4.5.1 Joint Training Rationale

Training the encoder and decoder independently creates a semantic mismatch: an encoder trained for classification may produce embeddings that are not well-suited for image reconstruction. Joint end-to-end training forces the encoder to produce embeddings that are simultaneously:
1. **Discriminative** — different classes occupy distinct regions
2. **Reconstructible** — the decoder can recover the original image from the embedding
3. **Interpolable** — neighbouring embeddings correspond to perceptually similar images

This joint optimisation is the foundation that makes SLERP-SMOTE semantically meaningful.

#### 4.5.2 Phased Training Schedule

Training is divided into two global phases based on the total epoch count:

**Phase 1 — Pure Reconstruction (epochs 0 to 0.3 × T_total):**
Only reconstruction losses are active. This phase establishes stable encoder representations before introducing adversarial pressure.

**Phase 2 — Adversarial Refinement (epochs 0.3 × T_total to T_total):**
The WGAN-GP discriminator is activated. The adversarial weight λ_adv ramps linearly from 0.05 to 0.20 over the adversarial phase, preventing mode collapse from sudden adversarial pressure.

The phase boundary is computed globally so that segmented training (checkpoint-and-resume) seamlessly continues from the correct phase without restarting the reconstruction warmup.

#### 4.5.3 Generator Objective Function

The total generator loss at step t is:

**Phase 1 (reconstruction only):**
```
L_G = L_MSE + 0.5 · L_L1 + 0.05 · L_percep
```

**Phase 2 (with adversarial and feature matching):**
```
L_G = L_MSE + 0.5 · L_L1 + 0.05 · L_percep + λ_adv · L_adv + 0.1 · L_FM
```

Where:
- `L_MSE = MSE(G(E(x)), x)` — pixel-level mean squared error
- `L_L1 = L1(G(E(x)), x)` — pixel-level mean absolute error (sharper edges than MSE alone)
- `L_percep` — VGG-based perceptual loss (Section 4.8)
- `L_adv = -E[D(G(E(x)))]` — WGAN generator loss (maximise critic score on fakes)
- `L_FM` — feature matching loss (Section 4.7)
- `λ_adv = 0.05 + 0.15 · frac` where `frac = (epoch - warmup_end) / (T_total - warmup_end)`

**Gradient clipping** is applied to all generator parameters with `max_norm=1.0` after the backward pass to prevent gradient explosions during early adversarial training.

#### 4.5.4 Optimiser and Learning Rate Schedule

A single Adam optimiser with parameters `(β₁=0.5, β₂=0.999, lr=2×10⁻⁴)` is used for all generator parameters (encoder + decoder jointly). The learning rate follows a **Cosine Annealing** schedule:

```
lr(t) = lr_min + 0.5 · (lr_max - lr_min) · (1 + cos(π · t / T_max))
```

where `T_max` is the total number of global training epochs and `lr_min = 10⁻⁵`. For segmented (checkpoint-and-resume) training, the scheduler is restored to the correct phase using `last_epoch=global_start_epoch - 1`, ensuring the learning rate trajectory is continuous across sessions. This is made possible by pre-initialising `initial_lr` in all parameter groups before constructing the scheduler.

#### 4.5.5 Segmented Training and Checkpoint Resumption

The pipeline supports interruption and resumption:

```python
pipeline.fit(images, labels, decoder_epochs=50,
             start_epoch=0,   total_epochs=200)  # segment 1
pipeline.fit(images, labels, decoder_epochs=50,
             start_epoch=50,  total_epochs=200)  # segment 2
pipeline.fit(images, labels, decoder_epochs=100,
             start_epoch=100, total_epochs=200)  # segment 3
```

The `global_epoch = global_start_epoch + local_epoch` is used throughout for:
- Phase boundary determination (GAN warmup)
- Learning rate schedule position
- Adversarial lambda ramping
- Epoch logging

---

### 4.6 Exponential Moving Average Inference Mechanism

**File:** `smote_image_synthesis/pipeline.py`
**Class:** `_EMA`

#### 4.6.1 Mechanism

The EMA module maintains a shadow copy of every learnable decoder parameter:

```
shadow_θ = {θ_name: clone(θ_value)  for all θ in decoder.parameters()}
```

After each generator optimiser step, the shadow parameters are updated:

```
shadow_θ ← decay · shadow_θ + (1 - decay) · θ_current
```

with `decay = 0.9999`. This corresponds to an effective lookback window of approximately `1/(1-0.9999) = 10,000` gradient steps.

#### 4.6.2 Application at Inference

After all training epochs complete, the EMA weights **replace** the instantaneous model weights:

```python
def apply(self, model):
    backup = {}
    for name, param in model.named_parameters():
        if name in self.shadow:
            backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])
    return backup
```

All subsequent calls to `decoder.decode()` and `pipeline.generate_synthetic_images()` use the EMA-smoothed weights. The backup enables restoration of the instantaneous weights if needed.

#### 4.6.3 Why EMA Improves Quality

During GAN training, generator weights oscillate around a local optimum due to the adversarial minimax dynamics. The instantaneous weights at any given step may be in a momentarily unfavourable configuration. The EMA averages out these oscillations, converging to a stable point that consistently produces higher-quality outputs. This technique is standard in state-of-the-art GAN systems (e.g., StyleGAN2, BigGAN) and has been shown to reduce FID scores by 10-30% compared to using instantaneous weights.

#### 4.6.4 EMA Update Frequency

The EMA shadow is updated after **every generator gradient step** (not just every epoch), providing fine-grained temporal smoothing:

```python
# In the training loop, after every opt_gen.step():
ema.update(self.decoder.model)
```

---

### 4.7 Discriminator with Feature Matching

**File:** `smote_image_synthesis/pipeline.py`
**Method:** `SynthesisPipeline._build_discriminator()`
**Inner class:** `Discriminator`

#### 4.7.1 WGAN-GP Discriminator Architecture

The discriminator is a fully convolutional critic network (no BatchNorm — gradient penalty provides the Lipschitz constraint):

```
Input: x [B, C, H, W]

Repeat while spatial_size > 4:
    Conv2d(in_ch, out_ch, 4×4, stride=2, padding=1, bias=False)
    LeakyReLU(0.2, inplace=True)
    in_ch = out_ch; out_ch = min(out_ch * 2, 512)

Final:
    Conv2d(in_ch, 1, 4×4, stride=1, padding=0, bias=True)
    Reshape → [B]   (scalar critic score per image)
```

For a `64×64` input with `base_channels=64`:
```
64×64×3 → 32×32×64 → 16×16×128 → 8×8×256 → 4×4×512 → 1×1×1 → scalar
```

#### 4.7.2 WGAN-GP Discriminator Objective

The discriminator maximises the Wasserstein-1 distance estimate with gradient penalty:

```
L_D = -E[D(x_real)] + E[D(G(E(x)))] + 10 · E[(‖∇D(x̂)‖₂ - 1)²]
```

where `x̂ = α·x_real + (1-α)·G(E(x))` with `α ~ Uniform(0,1)` is the interpolated sample used for the gradient penalty computation. The gradient penalty coefficient is fixed at 10 (following Gulrajani et al., 2017).

The discriminator is updated `n_critic=2` times per generator update step.

#### 4.7.3 Feature Matching Loss (Novel)

The `Discriminator` class exposes intermediate feature maps via a dedicated method:

```python
def get_features(self, x):
    features = []
    for layer in self.feat_layers:
        x = layer(x)
        if isinstance(layer, nn.LeakyReLU):
            features.append(x)
    return features
```

This returns one feature map per convolutional block (after each LeakyReLU activation), giving a hierarchical multi-scale representation of the input.

The feature matching loss penalises the L1 distance between discriminator features of real and generated images:

```
L_FM = Σ_{l=1}^{L} L1(D_l(G(E(x))), D_l(x_real))
```

where `D_l(·)` denotes the feature map extracted at the l-th discriminator convolutional block. This loss encourages the generator to match the statistics of real images at multiple scales simultaneously, leading to better textures and structural coherence.

The feature matching loss is incorporated into the generator objective with weight 0.1 during the adversarial phase.

---

### 4.8 Perceptual Loss Module

**File:** `smote_image_synthesis/decoders/autoencoder_trainer.py`
**Class:** `PerceptualLoss`

#### 4.8.1 Architecture

Perceptual loss (Johnson et al., 2016) measures image similarity in the feature space of a pretrained VGG16 network rather than pixel space. This captures perceptual similarity that pixel-level losses miss (e.g., textures, object shapes, high-frequency detail).

**VGG16 Feature Extraction Layers:**
```
relu1_2  (index 3)   — low-level edges and colours
relu2_2  (index 8)   — corners and simple textures
relu3_3  (index 15)  — complex textures and patterns
relu4_3  (index 22)  — high-level object parts
```

The perceptual loss is:
```
L_percep = Σ_{l ∈ {3,8,15,22}} MSE(VGG_l(G(E(x))), VGG_l(x_real))
```

Images are mapped from the model's `[-1, 1]` range to `[0, 1]` before VGG processing, then normalised with ImageNet statistics `(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.

#### 4.8.2 Efficiency Optimisation

Target layer indices are precomputed once at initialisation as a `frozenset`, replacing an O(N) linear search per forward pass with O(1) set membership testing:

```python
self._target_layer_indices = frozenset(
    self.layer_name_mapping[n]
    for n in self.layers
    if n in self.layer_name_mapping
)
# In _extract_features:
if i in self._target_layer_indices:
    features.append(x)
```

This avoids rebuilding the target layer list on every call (previously O(N²) per batch).

---

### 4.9 Quality Assessment Module

**File:** `smote_image_synthesis/quality/assessor.py`
**Class:** `QualityAssessor`

The quality assessor computes seven distinct metrics to characterise synthetic image quality from different perspectives:

| Metric | Mathematical Definition | Interpretation |
|---|---|---|
| **MSE** | `E[(x_syn - x_real)²]` | Pixel-level fidelity |
| **MAE** | `E[|x_syn - x_real|]` | Pixel-level fidelity (robust) |
| **PSNR** | `20·log₁₀(1/√MSE)` dB | Signal-to-noise ratio |
| **SSIM** | Structural similarity index | Perceptual similarity |
| **MS-SSIM** | Multi-scale SSIM (scales: 1.0, 0.5, 0.25) | Multi-resolution perceptual similarity |
| **LPIPS** | VGG16 feature distance | Perceptual distance (higher = more different) |
| **FID** | Fréchet distance between Inception feature distributions | Distribution-level realism (lower = better) |

**Diversity Metrics:**
- `mean_pairwise_distance`: Average Euclidean distance between all synthetic image pairs
- `std_pairwise_distance`: Standard deviation of pairwise distances (higher = more varied)
- `min_pairwise_distance`: Minimum inter-sample distance (detects mode collapse)
- `max_pairwise_distance`: Maximum inter-sample distance
- `diversity_index`: Normalised diversity score ∈ [0, 1]

**FID Implementation:** Inception V3 features (pool3 output, 2048-dim) are extracted for both real and synthetic images. The Fréchet distance between the multivariate Gaussians fitted to each feature set is computed using the matrix square root formula:

```
FID = ‖μ_real - μ_syn‖² + Tr(Σ_real + Σ_syn - 2·√(Σ_real·Σ_syn))
```

---

### 4.10 General-Purpose Dataset Interface

**File:** `run_pipeline.py`

The general-purpose interface accepts any image dataset organised in the standard folder hierarchy:

```
dataset_root/
    class_0/  image_001.jpg  image_002.png  ...
    class_1/  image_001.jpg  ...
    class_N/  image_001.jpg  ...
```

#### 4.10.1 Automatic Dataset Loading (`FolderDataset`)

The `FolderDataset` class:
1. Scans all sub-directories of the root folder, treating each sub-directory name as a class label
2. Assigns integer class indices in alphabetical order
3. Loads all image files with extensions `{.jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp}`
4. Applies the standard preprocessing transform pipeline:
   - Resize to square at target resolution
   - RandomHorizontalFlip (data augmentation)
   - ColorJitter (brightness ±15%, contrast ±15%, saturation ±10%)
   - ToTensor (scale to [0,1])
   - Normalize to [-1, 1] using mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]

#### 4.10.2 Automatic Class Balancing

Three balancing strategies are supported:

| Strategy | Target Count per Class | Use Case |
|---|---|---|
| `majority` | max(all class counts) | Match the largest class |
| `mean` | mean(all class counts) | Moderate balancing |
| `N` (integer) | N | Exact user-specified count |

The number of synthetic samples generated per class is `max(0, target - current_count)`, so no over-generation occurs for classes that already meet or exceed the target.

#### 4.10.3 Output Artefacts

| File | Contents |
|---|---|
| `real_samples.png` | Grid of real training images with class labels |
| `synthetic_samples.png` | Grid of SMOTE-generated synthetic images |
| `comparison.png` | Side-by-side real vs. synthetic comparison |
| `class_balance.png` | Bar chart: class counts before and after augmentation |
| `quality_metrics.json` | All quality metrics in machine-readable format |
| `pipeline_encoder.pth` | Serialised encoder state dict |
| `pipeline_decoder.pth` | Serialised decoder state dict (EMA weights) |
| `ckpt_epoch_N/` | Intermediate checkpoint directories |

---

## 5. NOVEL CONTRIBUTIONS AND CLAIMS

The following aspects of this invention are novel with respect to the prior art and constitute the basis for patent claims:

### Claim 1 — SLERP-SMOTE on Adversarially Trained Hyperspherical Embeddings
A method for synthetic minority-class image generation comprising: training an image encoder that produces L2-normalised embeddings on the unit hypersphere via joint end-to-end adversarial training; and generating synthetic embedding vectors by performing spherical linear interpolation (SLERP) between k-nearest-neighbour pairs in the hyperspherical embedding space.

### Claim 2 — Class-Conditional DCGAN Decoder with Embedded Class Prior
A generative decoder architecture comprising: a learnable class embedding table that maps discrete class indices to dense vectors; concatenation of class embedding with image embedding prior to the first linear projection layer; and a fallback zero-class embedding enabling backward-compatible unconditional generation.

### Claim 3 — Self-Attention at 16×16 Spatial Resolution in DCGAN Generator
A method of inserting a non-local self-attention block (SAGAN architecture) specifically at the 16×16 spatial feature map within a DCGAN-style generator, where the attention scale parameter γ is initialised to zero to preserve training stability.

### Claim 4 — EMA Post-Training Weight Substitution for SMOTE-Based Image Synthesis
A method of applying Exponential Moving Average weight smoothing to a jointly trained adversarial generator, wherein the EMA shadow weights are computed during training and permanently substituted for the instantaneous weights at the conclusion of training, and wherein all subsequent SMOTE-based synthetic image generation uses exclusively the EMA-smoothed weights.

### Claim 5 — Multi-Scale Feature Matching Against WGAN-GP Discriminator Features
A training objective that combines Wasserstein adversarial loss with a feature matching loss computed against intermediate activations of the WGAN-GP discriminator, where the discriminator architecture explicitly exposes per-block feature maps via a `get_features()` method for use in the generator's training objective.

### Claim 6 — Globally Consistent Phased Adversarial Training Across Segmented Sessions
A training procedure comprising: a reconstruction-only phase covering the first 30% of total epochs; an adversarial phase covering the remaining 70%; a global epoch counter maintained across independently executed training segments; and a CosineAnnealingLR schedule with `last_epoch` initialisation that correctly resumes the learning rate trajectory from any checkpoint.

### Claim 7 — Pipeline for Universal Class-Imbalanced Image Dataset Augmentation
An end-to-end configurable pipeline that accepts any image dataset in standard folder-class layout and produces a class-balanced augmented dataset via SLERP-SMOTE embedding interpolation and class-conditional adversarial decoding, with automatic imbalance detection and configurable balancing strategy.

### Claim 8 — von Mises-Fisher Distribution Sampling on Adversarially Trained Hyperspherical Embeddings
A method for synthetic minority-class embedding generation comprising: fitting a von Mises-Fisher (vMF) distribution per class to the L2-normalised embeddings residing on the unit hypersphere S^(D-1); estimating the mean direction μ and concentration parameter κ via maximum likelihood (Banerjee et al. approximation: κ̂ = r̄(D - r̄²)/(1 - r̄²)); sampling new unit-sphere embeddings from vMF(μ, κ·σ) using Wood's (1994) rejection algorithm with Householder rotation; and scaling samples back to the average class embedding norm. The concentration scale σ provides a continuous fidelity/diversity trade-off: σ > 1 increases fidelity, σ < 1 increases diversity. This is novel because k-NN SLERP is limited to interpolation between observed pairs and cannot sample the full class distribution on the hypersphere.

### Claim 9 — Density-Adaptive SLERP Interpolation Parameter Scheduling
A method for SLERP-SMOTE sample generation wherein the interpolation parameter t is selected based on the estimated local embedding density along each geodesic arc: when the midpoint density (approximated as the mean of per-point k-NN density estimates for the pair endpoints) falls below a class density threshold, t is sampled from a Beta(3, 3) distribution concentrated near 0.5 (gap-filling mode); otherwise t is sampled uniformly from [0, 1]. This density-weighted scheduling fills sparse regions of the embedding manifold more aggressively than existing pairs, producing synthetic samples that improve manifold coverage without drifting toward already-dense regions.

### Claim 10 — Projection Discriminator with Spectral Normalisation for Class-Conditional WGAN-GP
A discriminator architecture for class-conditional Wasserstein GAN training comprising: spectral normalisation (Miyato et al. 2018) applied to every convolutional layer providing per-layer Lipschitz-1 constraint via power iteration, complementing the global gradient penalty; a class embedding table V ∈ ℝ^{K×C} where K is the number of classes and C is the penultimate feature dimension; and a projection score augmentation D_proj(x, y) = D_base(x) + ⟨V·y, GAP(φ(x))⟩ where φ(x) is the penultimate feature map and GAP denotes global average pooling. The combination of per-layer SN and global GP provides dual Lipschitz regularisation superior to either constraint alone.

### Claim 11 — Intra-Class Embedding Repulsion Loss for Diversity Preservation in SMOTE-GAN Pipelines
A training regulariser applied during the adversarial phase that penalises within-class embedding pairs that are closer than a margin threshold: L_repulse = mean(ReLU(margin - ‖z_i - z_j‖₂)²) for all same-class pairs (i, j) in the batch. Applied to the encoder's output embeddings with weight λ_repulse = 0.01 during Phase 2 only, this loss explicitly prevents per-class mode collapse in the embedding space without affecting between-class separation.

### Claim 12 — Adaptive Wasserstein Distance Monitoring for Dynamic Adversarial Loss Scheduling
A method for dynamically adjusting the adversarial loss weight λ_adv during WGAN-GP training using a closed-loop controller: an exponential moving average of the epoch-level Wasserstein distance estimate is maintained (W_ema ← 0.99 · W_ema + 0.01 · W_current); the change in W_ema over a sliding window of 10 epochs determines whether to increase (when W-distance is improving) or decrease (when stagnant) λ_adv with step size 0.005, clamped to [0.01, 0.50]. This replaces heuristic linear ramps with a feedback-controlled schedule that responds to actual GAN convergence dynamics.

### Claim 13 — Synthesis Ancestry Tracking for Synthetic Dataset Data Provenance
A method for recording the generative lineage of every synthetic embedding produced by SLERP-SMOTE, comprising: storing for each synthetic sample the indices of its two parent real embeddings, the interpolation weight t, the method used (SLERP or vMF), and the within-class cluster assignment. This ancestry metadata enables auditing, reproducibility, and traceability of synthetic augmentation datasets — properties required by data governance regulations in high-stakes domains such as medical imaging and biometric identification.

---

## 6. MATHEMATICAL FORMULATIONS

### 6.1 SLERP Interpolation

For unit-normalised vectors **u₀**, **u₁** ∈ ℝᴰ with ‖**u₀**‖ = ‖**u₁**‖ = 1:

```
ω = arccos(clip(**u₀** · **u₁**, -1, 1))

SLERP(**u₀**, **u₁**, t) = sin((1-t)ω)/sin(ω) · **u₀**  +  sin(tω)/sin(ω) · **u₁**
```

For general (non-normalised) vectors **v₀**, **v₁** ∈ ℝᴰ:
```
**u₀** = **v₀**/‖**v₀**‖,  **u₁** = **v₁**/‖**v₁**‖
z_synth = SLERP(**u₀**, **u₁**, t) · ((1-t)·‖**v₀**‖ + t·‖**v₁**‖)
```

### 6.2 WGAN-GP Total Loss

**Discriminator:**
```
L_D = -E_{x~P_r}[D(x)] + E_{z~P_z}[D(G(z))]
    + 10 · E_{x̂~P_{x̂}}[(‖∇_{x̂} D(x̂)‖₂ - 1)²]
```

**Generator (Phase 2):**
```
L_G = MSE(G(E(x)), x)
    + 0.5 · L1(G(E(x)), x)
    + 0.05 · Σ_l MSE(VGG_l(G(E(x))), VGG_l(x))
    + λ_adv(t) · (-E[D(G(E(x)))])
    + 0.1 · Σ_l L1(D_l(G(E(x))), D_l(x))
```

where `λ_adv(t) = 0.05 + 0.15 · (t - t_warmup) / (T_total - t_warmup)`.

### 6.3 Exponential Moving Average

Let `θ_t` be the model parameters at step `t`:
```
shadow_θ_t = 0.9999 · shadow_θ_{t-1} + 0.0001 · θ_t
```

At inference: `θ_inference = shadow_θ_T` (final shadow state).

### 6.4 Self-Attention

For feature map `X ∈ ℝ^{B×C×H×W}` at 16×16:
```
Q = W_q · X_flat ∈ ℝ^{B×(HW)×(C/8)}
K = W_k · X_flat ∈ ℝ^{B×(C/8)×(HW)}
A = softmax(Q · K / √(C/8)) ∈ ℝ^{B×(HW)×(HW)}
V = W_v · X_flat ∈ ℝ^{B×C×(HW)}
Out = γ · reshape(V · Aᵀ) + X
```
where γ ∈ ℝ is a learnable scalar initialised to 0.

### 6.5 L2 Normalisation

```
z_normalised = z / ‖z‖₂  =  z / sqrt(Σᵢ zᵢ²)
```
This projects z onto the unit hypersphere S^{D-1}.

### 6.6 FID Score

Let μ_r, Σ_r and μ_s, Σ_s be the mean and covariance of Inception-V3 features for real and synthetic images respectively:
```
FID = ‖μ_r - μ_s‖₂²  +  Tr(Σ_r + Σ_s - 2·(Σ_r·Σ_s)^{1/2})
```
Lower FID indicates the synthetic distribution is closer to the real distribution.

### 6.7 von Mises-Fisher Distribution

The vMF distribution on the unit hypersphere S^{D-1} has density:
```
f(x; μ, κ) = C_D(κ) · exp(κ · μᵀx)
```
where C_D(κ) = κ^{D/2-1} / ((2π)^{D/2} · I_{D/2-1}(κ)) is the normalisation constant
and I_ν denotes the modified Bessel function of the first kind.

**MLE (Banerjee et al. 2005):**
```
μ̂ = (Σ xᵢ) / ‖Σ xᵢ‖₂            (normalised mean direction)
r̄ = ‖(1/n) Σ xᵢ‖₂               (mean resultant length, r̄ ∈ [0,1])
κ̂ ≈ r̄(D - r̄²) / (1 - r̄²)       (concentration approximation)
```

### 6.8 Projection Discriminator Score

For image x with class label y, the projection discriminator computes:
```
D_proj(x, y) = φ(x)ᵀ · w + ⟨V · y, GAP(h(x))⟩
```
where φ(x) is the penultimate feature map, h(x) = GAP(φ(x)) ∈ ℝ^C,
V ∈ ℝ^{K×C} is the class embedding table, and w is the final linear weight.

### 6.9 Intra-Class Repulsion Loss

For a mini-batch B with same-class pairs P_c = {(i,j) : y_i = y_j, i < j}:
```
L_repulse = (1/|P_c|) Σ_{(i,j)∈P_c} max(0, margin - ‖z_i - z_j‖₂)²
```
Total generator loss in Phase 2:
```
L_G = L_recon + λ_adv · L_adv + 0.1 · L_FM + λ_repulse · L_repulse
```

### 6.10 Adaptive λ_adv Control

EMA of Wasserstein estimate per epoch e:
```
W_ema[e] = 0.99 · W_ema[e-1] + 0.01 · (E[D(x_real)] - E[D(x_fake)])_e
dW = W_ema[e] - W_ema[e - W]       (W = 10-epoch window)
λ_adv[e+1] = clip(λ_adv[e] + 0.005 · sign(-dW), 0.01, 0.50)
```

---

## 7. DATA FLOW AND ALGORITHMS

### 7.1 Training Phase Data Flow

```
Images [N, C, H, W]  +  Labels [N]
          │
          ▼
  ResNetEncoder.encode()
          │
          ├──► embeddings [N, D]  ──►  ConstrainedSMOTE.fit(embeddings, labels)
          │
  SynthesisPipeline._train_end_to_end():
          │
    ┌─────▼──────────────────────────────────────────────────────────────┐
    │  For each epoch e in [0, T_total):                                 │
    │    global_epoch = start_epoch + e                                  │
    │    adv_active = (global_epoch >= 0.3 * T_total)                   │
    │                                                                    │
    │    For each mini-batch (x_batch, y_batch):                        │
    │                                                                    │
    │      // Discriminator step (if adv_active):                       │
    │      for _ in range(2):                                            │
    │        z = encoder(x_batch)                                        │
    │        x_fake = decoder(z, y_batch)  // no grad                   │
    │        d_real = disc(x_batch);  d_fake = disc(x_fake)             │
    │        gp = gradient_penalty(disc, x_batch, x_fake)               │
    │        L_D = -d_real + d_fake + 10*gp;  L_D.backward()            │
    │                                                                    │
    │      // Generator step:                                            │
    │      z = encoder(x_batch)                                          │
    │      x_recon = decoder(z, y_batch)                                │
    │      L_G = MSE + 0.5*L1 + 0.05*perceptual                        │
    │      if adv_active:                                                │
    │        L_G += λ_adv * (-disc(x_recon))                           │
    │        L_G += 0.1 * feature_matching(disc, x_batch, x_recon)     │
    │      clip_grad_norm_(gen_params, 1.0)                             │
    │      L_G.backward();  opt_gen.step()                              │
    │                                                                    │
    │      ema.update(decoder.model)   // shadow ← 0.9999*shadow + ...  │
    │                                                                    │
    │    sched_gen.step()              // Cosine annealing               │
    │                                                                    │
    │  ema.apply(decoder.model)        // Replace weights post-training  │
    └────────────────────────────────────────────────────────────────────┘
          │
  SynthesisPipeline.fit() (continued):
          │
    embeddings = encoder.encode(images)            // EMA decoder now active
    smote.fit(embeddings.cpu().numpy(), labels)    // Fit SLERP-SMOTE
```

### 7.2 Inference Phase Data Flow

```
SynthesisPipeline.generate_synthetic_images(n_samples):
          │
          ▼
  ConstrainedSMOTE.generate_synthetic(n_samples):
    work_embs = scaler.transform(embeddings)  // if scaler enabled
    syn_embs, syn_labels = _generate_slerp(work_embs, n_samples):
      For each class c:
        E_c = work_embs[labels==c]
        kNN.fit(E_c)
        For per_class = ceil(n/num_classes) iterations:
          Sample anchor i, neighbour j, weight t~U(0,1)
          z_syn = SLERP(E_c[i], E_c[nn[i,j]], t)
      Trim to exact n_samples
    syn_embs = scaler.inverse_transform(syn_embs)  // if scaler enabled
    return syn_embs [n, D], syn_labels [n]
          │
          ▼
  DCGANDecoder.decode(syn_embs_tensor, syn_labels_tensor):
    model.eval()
    with torch.no_grad():
      z_cond = concat([z, class_embed(labels)], dim=1)
      images = generator_backbone(z_cond)    // EMA weights used
    return images [n, C, H, W]   // Tanh output ∈ [-1, 1]
```

---

## 8. IMPLEMENTATION DETAILS

### 8.1 Software Stack

| Component | Library | Version |
|---|---|---|
| Deep learning | PyTorch | ≥ 2.0 |
| Computer vision | torchvision | ≥ 0.15 |
| SMOTE | imbalanced-learn | ≥ 0.12 |
| Clustering | scikit-learn | ≥ 1.3 |
| Image metrics | scikit-image, scipy | ≥ 0.21, ≥ 1.11 |
| Data processing | NumPy | ≥ 1.24 |
| Visualisation | matplotlib | ≥ 3.7 |

### 8.2 Module Hierarchy

```
smote_image_synthesis/
├── __init__.py
├── pipeline.py              ← SynthesisPipeline, _EMA
├── encoders/
│   ├── base.py              ← ImageEncoder (abstract)
│   └── resnet_encoder.py    ← ResNetEncoder, _L2Normalize
├── decoders/
│   ├── base.py              ← BaseDecoder (abstract)
│   ├── dcgan_decoder.py     ← DCGANDecoder, SelfAttention2d, _Generator
│   ├── autoencoder_decoder.py ← AutoencoderDecoder, ReshapeLayer
│   └── autoencoder_trainer.py ← AutoencoderTrainer, PerceptualLoss
├── smote/
│   └── constrained_smote.py ← ConstrainedSMOTE (with SLERP)
└── quality/
    ├── assessor.py          ← QualityAssessor
    └── reporter.py          ← QualityReporter, _NumpyEncoder
```

### 8.3 Model Persistence

Encoder and decoder weights are serialised independently using `torch.save(state_dict)`. The serialisation format is:

```json
{
  "embedding_dim": 512,
  "image_shape": [3, 64, 64],
  "model_path": "pipeline_decoder.pth",
  "config": {
    "base_channels": 512,
    "num_classes": 2,
    "class_embed_dim": 64,
    "use_self_attention": true
  }
}
```

The `load_from_config()` class method on each decoder reconstructs the full model from this JSON specification before loading the weight file, enabling full architecture recovery from disk.

### 8.4 Supported Decoder Architectures

The pipeline supports four decoder architectures through a unified `BaseDecoder` interface:

| Decoder | Training Method | Best For |
|---|---|---|
| `DCGANDecoder` | End-to-end + WGAN-GP (primary) | High visual quality, class fidelity |
| `AutoencoderDecoder` | End-to-end + MSE/perceptual | Faithful reconstruction |
| `VAEDecoder` | VAE ELBO via `VAETrainer` | Smooth latent interpolation |
| `DiffusionDecoder` | DDPM via `DiffusionTrainer` | Highest quality (slow) |

---

## 9. EXPERIMENTAL CONFIGURATION

### 9.1 Reference Experiment: CIFAR-10 Cats and Dogs

**Dataset:** CIFAR-10 classes 3 (cat) and 5 (dog)
**Training samples:** 300 per class (600 total)
**Image resolution:** 32×32 (native) or 64×64 (upsampled)
**Embedding dimension:** 512
**Encoder backbone:** ResNet18 (pretrained ImageNet-1K)
**Decoder:** DCGANDecoder (base_channels=512, num_classes=2, use_self_attention=True)
**Training epochs:** 150
**Batch size:** 32
**GAN warmup:** epochs 0–44 (reconstruction only); epochs 45–149 (adversarial)
**Optimiser:** Adam (β₁=0.5, β₂=0.999, lr=2×10⁻⁴)
**LR schedule:** CosineAnnealing (T_max=150, η_min=10⁻⁵)
**EMA decay:** 0.9999

### 9.2 Recommended Settings for Research Quality

| Setting | Value | Rationale |
|---|---|---|
| `image_size` | 64 | 4× more pixels than CIFAR native |
| `base_channels` | 512 | Wider generator = finer details |
| `epochs` | 200 | Longer adversarial phase |
| `n_per_class` | 500+ | Stronger SMOTE neighbourhood coverage |
| `normalize_output` | True | Essential for SLERP geometry |
| `use_slerp` | True | Manifold-faithful interpolation |
| `num_classes` | auto | Always match dataset class count |
| `use_self_attention` | True | Long-range coherence |

---

## 10. ADVANTAGES OVER PRIOR ART

| Feature | Standard SMOTE | GAN Augmentation | This Invention |
|---|---|---|---|
| Operates in pixel space | ✗ (direct) | ✗ | ✓ (embedding space) |
| Manifold-faithful interpolation | ✗ (linear) | N/A | ✓ (SLERP geodesic) |
| Class-conditional generation | ✗ | Partial | ✓ (class embedding) |
| Joint encoder-decoder training | ✗ | ✗ | ✓ |
| Adversarial sharpness | ✗ | ✓ | ✓ (WGAN-GP) |
| Multi-scale feature matching | ✗ | Partial | ✓ |
| EMA inference weights | ✗ | Partial | ✓ |
| Long-range spatial coherence | ✗ | Partial | ✓ (self-attention) |
| Class imbalance targeting | ✓ | ✗ | ✓ |
| Universal dataset interface | ✗ | ✗ | ✓ |
| Exact sample count control | ✗ | ✗ | ✓ (ceiling + trim) |
| Training resumption support | N/A | Partial | ✓ (global epochs) |

---

## APPENDIX A — KEY SOURCE CODE LOCATIONS

| Functionality | File | Class / Method |
|---|---|---|
| SLERP algorithm | `smote/constrained_smote.py` | `ConstrainedSMOTE._slerp()` |
| SLERP-SMOTE generation | `smote/constrained_smote.py` | `ConstrainedSMOTE._generate_slerp()` |
| Class embedding injection | `decoders/dcgan_decoder.py` | `_Generator.forward()` |
| Self-attention block | `decoders/dcgan_decoder.py` | `SelfAttention2d.forward()` |
| EMA tracker | `pipeline.py` | `_EMA` class |
| EMA application | `pipeline.py` | `_EMA.apply()` |
| Feature matching loss | `pipeline.py` | `_train_end_to_end()` |
| Discriminator feature extractor | `pipeline.py` | `Discriminator.get_features()` |
| L2 normalisation layer | `encoders/resnet_encoder.py` | `_L2Normalize.forward()` |
| Phased training schedule | `pipeline.py` | `_train_end_to_end()` |
| Cosine LR resumption | `pipeline.py` | `sched_gen` construction |
| Class-conditional decode | `decoders/dcgan_decoder.py` | `DCGANDecoder.decode()` |
| Perceptual loss | `decoders/autoencoder_trainer.py` | `PerceptualLoss.forward()` |
| Dataset auto-loading | `run_pipeline.py` | `FolderDataset`, `load_dataset()` |
| Auto-balancing | `run_pipeline.py` | `compute_target_counts()` |

---

## APPENDIX B — PARAMETER COUNT SUMMARY

**Encoder (ResNet18, D=512, with L2Norm):**
- Backbone: ~11.7M parameters
- Projection head: 512×512 = ~263K parameters
- Total: ~11.96M parameters

**Decoder (DCGANDecoder, D=512, 64×64, base_ch=512, 2 classes, attention):**
- Class embedding: 2 × 64 = 128 parameters
- Linear projection: 576 × 8192 = ~4.72M parameters
- 4 ConvTranspose2d blocks: ~8.4M parameters
- SelfAttention2d (128 channels): ~49K parameters
- Total: ~13.17M parameters

**Discriminator (64×64 input, base_ch=64):**
- 4 Conv2d blocks: ~1.8M parameters
- Final Conv2d: ~131K parameters
- Total: ~1.93M parameters

**Grand Total (all trained modules):** ~27.06M parameters

---

*This document constitutes a complete technical disclosure of the invention for purposes of patent prosecution. All source code referenced herein is maintained under version control and constitutes the definitive implementation of the described system.*

*Document prepared: 2026-03-15*
