# SMOTE for Images

Synthetic image generation via SMOTE applied in **embedding space** — not pixel space.
Images are encoded to a latent hypersphere, new embeddings synthesised using geodesic SLERP or von Mises-Fisher sampling, and decoded back to pixels using a class-conditional DCGAN trained jointly with the encoder via WGAN-GP.

---

## Table of Contents

1. [Research Problem](#research-problem)
2. [Architecture Overview](#architecture-overview)
3. [Quick Start](#quick-start)
4. [Running the Pipeline](#running-the-pipeline)
5. [Frontier Pipeline](#frontier-pipeline)
6. [Component Reference](#component-reference)
   - [SynthesisPipeline](#synthesispipeline)
   - [ResNetEncoder](#resnetencoder)
   - [DCGANDecoder](#dcgandecoder)
   - [Discriminator](#discriminator)
   - [ConstrainedSMOTE](#constrainedsmote)
   - [SLERP](#slerp)
   - [vMF Sampling](#vmf-sampling)
   - [Density-Weighted t](#density-weighted-t)
   - [Cluster Constraints](#cluster-constraints)
   - [Outlier Detection](#outlier-detection)
   - [Ancestry Tracking](#ancestry-tracking)
   - [Decoder Zoo](#decoder-zoo)
   - [QualityAssessor](#qualityassessor)
   - [QualityReporter](#qualityreporter)
   - [Data Models](#data-models)
   - [EMA](#ema)
   - [Repulsion Loss](#repulsion-loss)
   - [Adaptive λ_adv](#adaptive-λ_adv)
7. [Training Details](#training-details)
8. [Configuration System](#configuration-system)
9. [Testing](#testing)
10. [Installation](#installation)
11. [Project Structure](#project-structure)
12. [Documentation Index](#documentation-index)

---

## Research Problem

Standard SMOTE interpolates linearly between feature vectors: `e_syn = e_i + t(e_j - e_i)`.
For tabular data this is fine. For high-dimensional image embeddings this is wrong — the learned representation manifold is curved, not flat. Linear interpolation crosses *off-manifold* regions, producing embeddings that decode to blurry or incoherent images.

**This project solves it by:**

- Constraining all interpolation to the unit hypersphere via SLERP (great-circle arcs)
- Alternatively, sampling directly from the von Mises-Fisher distribution fitted to each class
- Training encoder and decoder *jointly* so the latent space is shaped by both reconstruction and adversarial objectives
- Adding cluster constraints, outlier rejection, and density-weighted sampling to further improve manifold adherence

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  INPUT                                                                  │
│  Dataset: class_a/ class_b/ class_c/  (images in subfolders)           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ImagePreprocessor                                                      │
│  • Resize to N×N (default 64×64)                                       │
│  • RGB, ImageNet normalise (mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])    │
│  • Optional augmentation: hflip, ColorJitter                           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  torch.Tensor [N, 3, H, W]
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ResNetEncoder  (resnet18 / resnet50 / resnet101)                       │
│  • ResNet backbone (pretrained ImageNet, all layers unfrozen in E2E)   │
│  • Remove FC head → Flatten → Dropout → Linear(backbone_dim, 512)     │
│  • BatchNorm1d → ReLU → L2Normalize  →  unit hypersphere              │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  embeddings [N, 512]  ‖e‖₂ = 1
                                │
               ┌────────────────┴────────────────┐
               │                                 │
               ▼                                 ▼
┌──────────────────────────┐     ┌───────────────────────────────────────┐
│  ConstrainedSMOTE        │     │  _train_end_to_end (joint E2E)        │
│                          │     │                                       │
│  fit(embeddings, labels) │     │  Phase 1  (epochs 0 → 30%):          │
│                          │     │  G = MSE + 0.5×L1 + 0.05×VGG         │
│  generate_synthetic():   │     │                                       │
│  ── SLERP interpolation  │     │  Phase 2  (epochs 30% → 100%):       │
│  ── vMF distribution     │     │  D = -E[D(real)] + E[D(fake)]        │
│  ── density-weighted t   │     │      + 10×GP   (5 D steps per G)     │
│  ── cluster constraints  │     │  G = recon + λ_adv×(-E[D(fake)])     │
│  ── outlier filtering    │     │      + 0.1×FM + 0.01×repulsion       │
│  ── ancestry tracking    │     │                                       │
│                          │     │  λ_adv adaptive via W-distance EMA   │
│  → synthetic embs [M,512]│     │  EMA(encoder+decoder, decay=0.9999)  │
└──────────────┬───────────┘     └───────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DCGANDecoder  (class-conditional)                                      │
│                                                                         │
│  emb[512] + class_emb[64]  →  [576]                                    │
│  → Linear(576, ch₀×16) → BN → ReLU → Reshape(ch₀, 4, 4)              │
│  → DeConvBlock  4→8    (ch₀   → ch₀/2)   BN+ReLU                      │
│  → DeConvBlock  8→16   (ch₀/2 → ch₀/4)   BN+ReLU                      │
│  → SelfAttention2d  ←── SAGAN non-local attention (γ=0 at init)       │
│  → DeConvBlock  16→32  (ch₀/4 → ch₀/8)   BN+ReLU                      │
│  → DeConvBlock  32→64  (ch₀/8 → ch₀/16)  BN+ReLU   [if size=64]       │
│  → ConvTranspose2d → Tanh  →  [3, H, W]                                │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  synthetic images [M, 3, H, W]
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  QualityAssessor                                                        │
│  • FID · LPIPS · SSIM · MS-SSIM · PSNR · MSE · MAE · diversity         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  QualityReporter  →  HTML report + CSV + comparison grids + plots       │
└─────────────────────────────────────────────────────────────────────────┘

OUTPUTS
  real_samples.png        synthetic_samples.png       comparison.png
  class_balance.png       quality_metrics.json
  pipeline_encoder.pth    pipeline_decoder.pth
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Organise dataset — one subfolder per class
my_dataset/
    cats/    cat001.jpg  cat002.jpg  ...
    dogs/    dog001.jpg  dog002.jpg  ...
    birds/   bird001.jpg ...

# 3. Run
python run_pipeline.py --data-dir my_dataset

# Outputs → pipeline_output/
```

---

## Running the Pipeline

### `run_pipeline.py` — general-purpose entry point

```bash
python run_pipeline.py \
    --data-dir      /path/to/dataset    # required: root with class subdirs
    --output-dir    pipeline_output     # where to write everything
    --image-size    64                  # resize to N×N
    --epochs        150                 # training epochs
    --architecture  resnet18            # resnet18 | resnet50
    --embedding-dim 512
    --base-channels 256                 # 256=fast  512=quality
    --batch-size    32
    --n-per-class   None                # cap images per class (None=all)
    --n-synthetic   None                # exact count to generate (None=auto-balance)
    --balance-to    majority            # majority | mean | <integer>
    --save-every    0                   # checkpoint every N epochs (0=end only)
    --resume        pipeline_output/ckpt_epoch_50
    --no-pretrained                     # skip ImageNet init
```

**What happens step by step:**

| Step | What |
|------|------|
| 1 | Load images from class subdirs, apply transforms |
| 2 | Build `ResNetEncoder` + `DCGANDecoder` + `ConstrainedSMOTE` + `QualityAssessor` |
| 3 | Resume from checkpoint if `--resume` given |
| 4 | Run `pipeline.fit()` — joint E2E training, then SMOTE fit |
| 5 | Compute how many synthetics needed to balance classes |
| 6 | Run `pipeline.generate_synthetic_images()` |
| 7 | Save comparison grids and class-balance chart |
| 8 | Run `pipeline.evaluate_quality()`, save `quality_metrics.json` |
| 9 | Save final encoder/decoder weights |

**Dataset format:**

```
data_root/
    class_a/   img001.jpg  img002.png  img003.webp  ...
    class_b/   img001.jpg  ...
    class_c/   ...
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`

Images that fail to open are replaced with a black frame and a warning is logged — the run continues.

**Typical commands:**

```bash
# Quick test
python run_pipeline.py --data-dir my_dataset --epochs 50 --image-size 32

# Quality run
python run_pipeline.py --data-dir my_dataset --epochs 300 --architecture resnet50 \
    --base-channels 512 --image-size 128

# Balance to a fixed count per class
python run_pipeline.py --data-dir my_dataset --balance-to 1000

# Cap large classes, generate exactly 200 synthetics
python run_pipeline.py --data-dir my_dataset --n-per-class 300 --n-synthetic 200

# Resume interrupted training
python run_pipeline.py --data-dir my_dataset \
    --resume pipeline_output/ckpt_epoch_100 --epochs 200

# Checkpoint every 50 epochs
python run_pipeline.py --data-dir my_dataset --epochs 200 --save-every 50
```

### `test_frontier.py` — frontier CIFAR-10 reference run

```bash
python test_frontier.py
# Downloads CIFAR-10 to data/ automatically
# Outputs → frontier_output/
```

Trains 200 epochs on CIFAR-10 cats & dogs with every improvement enabled. Saves comparison grids at epoch 100 and 200 (real | SLERP-synthetic | vMF-synthetic side by side).

### `demo_pipeline.py` — no real data needed

```bash
python demo_pipeline.py
python demo_pipeline.py --n-samples 200 --decoder-type vae --generate-report
python demo_pipeline.py --help
```

Generates synthetic checkerboard patterns internally for fast smoke-testing.

### `run_app.py` — interactive menu

```bash
python run_app.py
```

### `train_cats_dogs.py` — cats/dogs training example

```bash
python train_cats_dogs.py
```

---

## Frontier Pipeline

`test_frontier.py` is the reference implementation with all frontier improvements active simultaneously. Key settings:

```python
encoder = ResNetEncoder(
    architecture='resnet18',
    normalize_output=True,     # L2-norm → unit hypersphere
)

decoder = DCGANDecoder(
    base_channels=256,         # 256 converges 2× faster on small datasets
    num_classes=2,             # class-conditional
    use_self_attention=True,   # SAGAN at 16×16
)

smote = ConstrainedSMOTE(
    use_slerp=True,
    density_weighted_t=True,
    use_cluster_constraints=True,
    use_outlier_detection=True,
    boundary_detection_method='isolation',
    track_ancestry=True,
    normalize_embeddings=False,  # encoder already normalises
)
```

**Achieved results** (`frontier_output/metrics.json`):

| Epochs | Avg G-loss (last 10 ep) | Status |
|--------|------------------------|--------|
| 100 | 0.4250 | GAN warming up |
| 200 | **0.0839** | Converged |

Target: `G-loss ≤ 0.10`

**Loop logic:** after each 100-epoch segment the script checks if `G-loss < 0.10` (target reached) or improvement `< 0.03` between segments (plateau). Exits when either condition met or epoch budget exhausted.

**Comparison grids saved:** `frontier_output/comparison_ep100.png`, `comparison_ep200.png` — three rows: real images | SLERP synthetics | vMF synthetics.

---

## Component Reference

### SynthesisPipeline

`smote_image_synthesis/pipeline.py`

Main orchestrator. Owns encoder, decoder, SMOTE, and quality assessor. Provides the full `fit → generate → evaluate` lifecycle.

```python
from smote_image_synthesis import SynthesisPipeline

pipeline = SynthesisPipeline(
    encoder,           # ResNetEncoder (or any ImageEncoder subclass)
    decoder,           # DCGANDecoder (or any BaseDecoder subclass)
    smote,             # ConstrainedSMOTE
    quality_assessor,  # QualityAssessor | None  (auto-created if None)
)
```

**Validation on construction:** raises `ValueError` if `encoder.embedding_dim ≠ decoder.embedding_dim`.

#### `fit()`

```python
pipeline.fit(
    images,                # torch.Tensor [N, C, H, W] — already on device
    labels,                # np.ndarray [N] — integer class indices
    train_decoder=True,
    decoder_epochs=100,    # epochs for THIS call (segment)
    start_epoch=0,         # global epoch offset — aligns cosine LR schedule
    total_epochs=0,        # 0 = same as decoder_epochs
)
```

Routing logic:

| Decoder type | Training path |
|-------------|--------------|
| `DCGANDecoder` | `_train_end_to_end()` — joint encoder+decoder |
| `AutoencoderDecoder` | `_train_end_to_end()` — joint encoder+decoder |
| `VAEDecoder` | `VAETrainer` — separate |
| `GANDecoder` | `GANTrainer` — separate |
| `DiffusionDecoder` | `DiffusionTrainer` — separate |

After decoder training in all cases: `encoder.encode(images)` → `smote.fit(embeddings, labels)`.

#### `generate_synthetic_images()`

```python
syn_images, syn_labels = pipeline.generate_synthetic_images(n_samples=500)
```

Calls `smote.generate_synthetic(n_samples)` then batches embeddings through `decoder.decode()`. Passes class labels to decoder if `decoder.num_classes > 0`.

Returns `torch.Tensor [N, C, H, W]` and `np.ndarray [N]`.

#### `evaluate_quality()`

```python
metrics = pipeline.evaluate_quality(synthetic_images, real_images)
# Returns flat dict: {'fid': ..., 'ssim': ..., 'psnr': ..., ...}
```

Flattens any nested dicts from `QualityAssessor`.

#### `save_pipeline()` / `load_pipeline()`

```python
pipeline.save_pipeline("output/pipeline")
# Writes: output/pipeline_encoder.pth, output/pipeline_decoder.pth

pipeline.load_pipeline("output/pipeline")
```

---

### ResNetEncoder

`smote_image_synthesis/encoders/resnet_encoder.py`

```python
from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder

encoder = ResNetEncoder(
    architecture='resnet18',   # 'resnet18' | 'resnet50' | 'resnet101'
    embedding_dim=512,
    pretrained=True,           # ImageNet1K_V1 weights
    freeze_backbone=False,     # True = only train projection head
    dropout_rate=0.1,
    normalize_output=True,     # L2-normalise → unit sphere (needed for SLERP)
    device=None,               # auto-detected (cuda if available)
)
```

**Model structure:**

```
ResNet backbone (all conv layers, no original FC)
  → Flatten
  → Dropout(dropout_rate)
  → Linear(backbone_features, embedding_dim)
  → BatchNorm1d(embedding_dim)
  → ReLU
  → L2Normalize  [if normalize_output=True]
```

The `_L2Normalize` module applies `F.normalize(x, p=2, dim=1)` — projects each embedding onto the unit hypersphere. This is a hard precondition for SLERP to be geometrically valid.

**Key methods:**

```python
embs = encoder.encode(images)                               # [N, D]
embs = encoder.encode_with_memory_management(images, 32)    # batched, safe on GPU

encoder.set_backbone_frozen(True)   # freeze conv layers dynamically
encoder.set_backbone_frozen(False)  # unfreeze for E2E training

# Supervised fine-tuning (adds a temporary linear classifier)
history = encoder.fine_tune(dataloader, num_epochs=5, unfreeze_after_epochs=2)

feats = encoder.get_feature_maps(images, layer_name='layer4')  # intermediate features
info  = encoder.get_model_info()   # param counts, architecture details
```

**Loading from saved config:**

```python
encoder = ResNetEncoder.load_from_config('output/pipeline_encoder.json')
```

Config JSON is written by `save_model()` and contains architecture, embedding_dim, and model path.

---

### DCGANDecoder

`smote_image_synthesis/decoders/dcgan_decoder.py`

The frontier decoder. Class-conditional, trained only via `SynthesisPipeline._train_end_to_end()` — it has no standalone trainer.

```python
from smote_image_synthesis.decoders.dcgan_decoder import DCGANDecoder

decoder = DCGANDecoder(
    embedding_dim=512,
    image_shape=(3, 64, 64),    # (C, H, W) — H/W must be powers of 2, ≥ 8
    base_channels=256,          # ch₀; channel schedule: [256, 128, 64, 32, ...]
    num_classes=0,              # 0 = unconditional; >0 = class-conditional
    class_embed_dim=64,         # dim of nn.Embedding for class labels
    use_self_attention=True,    # SelfAttention2d at 16×16 spatial
    device=None,
)
```

**Generator architecture** for `image_shape=(3,64,64)`, `base_channels=256`, `num_classes=2`:

```
Input:  emb[512] ‖ class_emb[64]  =  [576]
        ↓
Linear(576, 256×4×4)  BN1d  ReLU  Reshape(256, 4, 4)
        ↓
ConvTranspose2d(256→128, k=4, s=2, p=1)  BN2d  ReLU   →  8×8
        ↓
ConvTranspose2d(128→64,  k=4, s=2, p=1)  BN2d  ReLU   →  16×16
        ↓
SelfAttention2d(64)   ← inserted here when use_self_attention=True
        ↓
ConvTranspose2d(64→32,   k=4, s=2, p=1)  BN2d  ReLU   →  32×32
        ↓
ConvTranspose2d(32→3,    k=4, s=2, p=1)  Tanh          →  64×64

Output: [3, 64, 64] in range (-1, 1)
```

Channel schedule is `[max(32, base_ch // 2^i) for i in 0..n_up]`.

Self-attention is inserted **once** when spatial resolution reaches 16×16 — the sweet spot between local texture (too early) and semantic layout (too late) according to SAGAN.

**Weight initialisation:** all Conv/ConvTranspose: `N(0, 0.02)`. All BN: weight `N(1, 0.02)`, bias 0. Embedding: `N(0, 0.02)`.

**`decode()` method:**

```python
images = decoder.decode(embeddings, labels=None)
# embeddings: [N, 512] — on CPU or device, auto-moved
# labels: [N] long or None — required if num_classes > 0
# returns [N, 3, H, W] on CPU
```

Runs in `torch.no_grad()` mode. Temporarily switches to eval if model is training.

**Loading from saved config:**

```python
decoder = DCGANDecoder.load_from_config('frontier_output/frontier_pipeline_decoder.json')
```

---

### Discriminator

Built dynamically inside `SynthesisPipeline._build_discriminator()`. Not exported directly.

**Architecture:**

```
Input [3, H, W]
  → SpectralNorm(Conv2d(3,   64,  4, 2, 1))  LeakyReLU(0.2)   →  H/2
  → SpectralNorm(Conv2d(64,  128, 4, 2, 1))  LeakyReLU(0.2)   →  H/4
  → SpectralNorm(Conv2d(128, 256, 4, 2, 1))  LeakyReLU(0.2)   →  H/8
  → ...  (until spatial = 4)
  → SpectralNorm(Conv2d(last_ch, 1, 4, 1, 0))                 → scalar

  [if num_classes > 0]
  Projection head: score += <Embedding(label), GAP(penultimate_feat)>
```

- **Spectral normalisation** on every `Conv2d` via `torch.nn.utils.spectral_norm` — enforces Lipschitz-1 per layer by dividing weight by its largest singular value (power iteration). Complements the global WGAN-GP gradient penalty.
- **Projection discriminator** (Miyato & Koyama, 2018): class-conditional score without concatenating class info in the input. Formally: `D(x,y) = φ(x)ᵀ·V_y + f(φ(x))` where `V_y` is the class embedding and `f` is a linear head on the global average pooled features.
- **No BatchNorm** — WGAN-GP gradient penalty requires computing gradients through the discriminator; BN would break the gradient w.r.t. interpolated inputs.
- **Feature extraction** at 3 evenly-spaced depths (early/mid/late LeakyReLU outputs) for multi-scale feature matching.

---

### ConstrainedSMOTE

`smote_image_synthesis/smote/constrained_smote.py`

```python
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE

smote = ConstrainedSMOTE(
    # Core
    k_neighbors=5,
    sampling_strategy='auto',      # 'auto' | 'minority' | dict {class_idx: n_samples}
    random_state=42,
    normalize_embeddings=False,    # False when encoder already L2-normalises

    # Interpolation method (mutually exclusive; vMF takes priority)
    use_slerp=True,
    use_vmf=False,
    vmf_concentration_scale=1.0,  # scale fitted κ: >1 tighter, <1 more diverse

    # Frontier improvements
    density_weighted_t=True,
    use_cluster_constraints=True,
    use_outlier_detection=True,
    boundary_detection_method='isolation',  # 'density' | 'svm' | 'isolation'
    track_ancestry=True,

    # Clustering (required for cluster constraints)
    use_clustering=True,
    clustering_method='kmeans',   # 'kmeans' | 'dbscan' | 'hierarchical' | 'gmm'
    n_clusters=None,              # auto if None

    # Validation thresholds
    max_distance_threshold=None,
    outlier_detection_threshold=0.1,
    min_samples_per_class=2,
)
```

**Fit:**

```python
smote.fit(embeddings_np, labels)      # np.ndarray [N,D], np.ndarray [N]
```

Internally:
1. Optionally `StandardScaler` normalises embeddings (if `normalize_embeddings=True`)
2. `_apply_clustering_constraints()` — fit per-class cluster models, store `cluster_assignments`
3. `_initialize_outlier_detectors()` — fit per-class `IsolationForest` / `OneClassSVM` / `LOF`
4. Fit fallback sklearn `SMOTE` (used when both `use_slerp=False` and `use_vmf=False`)

**Generate:**

```python
syn_embs, syn_labs = smote.generate_synthetic(n_samples=500)

# With ancestry provenance
syn_embs, syn_labs, metadata = smote.generate_synthetic(n_samples=500, return_metadata=True)
```

```python
# Standard imbalanced-learn interface (fits + resamples)
X_res, y_res = smote.fit_resample(embeddings_np, labels)
```

---

### SLERP

Spherical Linear Interpolation between two embeddings on the unit hypersphere.

```
θ = arccos(clamp(u₀ · u₁, -1, 1))

if θ < 1e-6:   (nearly parallel — linear fallback)
    e_syn = ((1-t)·u₀ + t·u₁) / ‖...‖

else:
    e_syn = sin((1-t)·θ)/sin(θ) · u₀  +  sin(t·θ)/sin(θ) · u₁

e_syn = e_syn × ((1-t)·‖e₀‖ + t·‖e₁‖)   ← scale by lerp'd magnitude
```

The magnitude interpolation preserves the embedding "radius" when embeddings are not unit-normalised (e.g. if `normalize_output=False` on the encoder). When `normalize_output=True`, all norms are 1 so the magnitude step is a no-op.

Implementation: `ConstrainedSMOTE._slerp(v0, v1, t)` — static method, called per synthetic sample.

---

### vMF Sampling

Alternative to SLERP. Instead of picking two real embeddings and interpolating, fits a parametric distribution per class and samples from it.

**vMF distribution:** `p(x; μ, κ) ∝ exp(κ · μᵀ x)` on `S^(d-1)`.

- `μ` — mean direction (unit vector pointing toward class centroid on sphere)
- `κ` — concentration parameter (`κ=0` → uniform; `κ→∞` → delta at `μ`)

**Parameter estimation** (`_estimate_vmf_params`) — Banerjee et al. (2005) MLE approximation:

```
r̄ = ‖mean(unit_embeddings)‖
κ̂ = r̄(d - r̄²) / (1 - r̄²)
```

`vmf_concentration_scale` scales the fitted `κ̂` before sampling:
- `> 1` → tighter cluster around `μ` (higher fidelity, lower diversity)
- `< 1` → broader distribution (more diversity, may drift off-manifold)

**Sampling algorithm** (`_sample_vmf`) — Wood (1994) rejection sampler:

```
Parameters:
  b = (-2κ + √(4κ² + (d-1)²)) / (d-1)
  x₀ = (1-b)/(1+b)
  c = κ·x₀ + (d-1)·log(1 - x₀²)
  α = (d-1)/2

Per sample:
  Loop until accepted:
    Z ~ Beta(α, α)
    W = (1 - (1+b)Z) / (1 - (1-b)Z)
    U ~ Uniform(0,1)
    Accept if: κW + (d-1)·log(1 - x₀W) - c ≥ log(U)

  v ~ uniform on S^(d-2)  (d-1 dimensional sphere)
  x = [W, √(1-W²) · v]    (in frame where μ = e₁)
  Apply Householder reflection to rotate x so e₁ → μ
```

Samples are unit-normalised then scaled by the class average embedding norm to match the encoder's output scale.

---

### Density-Weighted t

Standard SMOTE samples `t ~ Uniform(0,1)`. Dense regions generate synthetics that cluster near real data (safe but redundant). Sparse regions — where the class boundary or intra-class variation gap is — get undersampled.

**Density estimate** per point `i`: `d(i) = 1 / (mean_kNN_distance(i) + ε)`

**Adaptive sampling rule** for each synthetic:

```
mid_density = 0.5 × (d(point_i) + d(point_j))

if mid_density < 0.8 × median_class_density:
    t ~ Beta(3, 3)     ← concentrated around 0.5 (fills the gap)
else:
    t ~ Uniform(0, 1)  ← standard SMOTE
```

`Beta(3,3)` has mean 0.5 and is unimodal — it pushes the synthetic to the midpoint of the geodesic arc, directly into the low-density gap.

---

### Cluster Constraints

Prevents interpolation between semantically different sub-modes within the same class (e.g., different dog breeds would be separate clusters even though both are labelled "dog").

**How it works:**

1. During `fit()`, per-class embeddings are clustered (K-means by default, auto-determining `n_clusters` from class size)
2. Each embedding gets a cluster assignment stored in `cluster_assignments[label]`
3. During generation, k-NN candidate neighbours for point `i` are filtered to **same-cluster only**
4. If a cluster has fewer than 2 members → falls back to global k-NN

```
without cluster constraints:    pick any k neighbour in the class
with cluster constraints:       pick only neighbours in cluster_assignments[i] == cluster_assignments[i]
```

Supported clustering methods: `'kmeans'`, `'dbscan'`, `'hierarchical'`, `'gmm'`. DBSCAN is skipped for cluster-constrained SLERP because its cluster IDs are not stable across the k-NN index.

---

### Outlier Detection

After each synthetic embedding is generated, it is evaluated by a per-class outlier detector:

| `boundary_detection_method` | Detector |
|-----------------------------|---------|
| `'isolation'` | `IsolationForest(contamination=outlier_detection_threshold)` |
| `'svm'` | `OneClassSVM(nu=outlier_detection_threshold)` |
| `'density'` | `LocalOutlierFactor(novelty=True)` |

**Retry logic:**

```
score = detector.predict([synthetic_emb])
if score == -1 (outlier):
    for attempt in 1..3:
        resample via SLERP with new random t
        if new sample is inlier: replace original, break
    if still outlier after 3 attempts: discard
```

LOF detectors require `novelty=True` to score new points — the code handles this correctly.

---

### Ancestry Tracking

When `track_ancestry=True`, every synthetic sample's provenance is recorded:

```python
smote.last_ancestry  # dict after generate_synthetic()

# Or request it directly:
syn_embs, syn_labs, ancestry = smote.generate_synthetic(return_metadata=True)

ancestry[42] = {
    'parent_a':    23,         # index in original embeddings array
    'parent_b':    47,
    'class_label': 1,
    't':           0.43,       # SLERP t  (or κ for vMF)
    'method':      'slerp',    # 'slerp' | 'vmf'
    'cluster':     2,          # cluster id (-1 if no cluster constraints)
}
```

For vMF: `parent_a = parent_b = -1` (no specific parent pair), `t` stores the fitted `κ` value.

---

### Decoder Zoo

All decoders share `BaseDecoder` (`decoders/base.py`) with `decode()`, `validate_embeddings()`, `save_model()`, `load_model()`.

#### AutoencoderDecoder + AutoencoderTrainer

```python
from smote_image_synthesis.decoders.autoencoder_decoder import AutoencoderDecoder
from smote_image_synthesis.decoders.autoencoder_trainer import AutoencoderTrainer

decoder = AutoencoderDecoder(embedding_dim=512, image_shape=(3,64,64))
trainer = AutoencoderTrainer(
    decoder,
    learning_rate=1e-3,
    use_perceptual_loss=True,  # VGG16-based perceptual loss
)
history = trainer.train(train_embeddings, train_images, num_epochs=100)
```

Progressive upsampling with skip connections. Perceptual loss from VGG16 intermediate activations.

#### VAEDecoder + VAETrainer

```python
from smote_image_synthesis.decoders.vae_decoder import VAEDecoder
from smote_image_synthesis.decoders.vae_trainer import VAETrainer

decoder = VAEDecoder(embedding_dim=512, image_shape=(3,64,64), latent_dim=256)
trainer = VAETrainer(decoder, beta=1.0, learning_rate=1e-3)
history = trainer.train(train_embeddings, train_images, num_epochs=200)
```

Reparameterisation trick. Loss = `reconstruction + β × KL(q(z|x) ‖ N(0,I))`. Beta-VAE supported via `beta > 1`.

#### GANDecoder + GANTrainer

```python
from smote_image_synthesis.decoders.gan_decoder import GANDecoder
from smote_image_synthesis.decoders.gan_trainer import GANTrainer

decoder = GANDecoder(embedding_dim=512, image_shape=(3,64,64))
trainer = GANTrainer(decoder, generator_lr=2e-4, discriminator_lr=2e-4)
history = trainer.train(train_embeddings, train_images, num_epochs=100)
```

Spectral normalisation, self-attention, feature matching loss. Separate generator/discriminator training loop (not joint with encoder).

#### DiffusionDecoder + DiffusionTrainer

```python
from smote_image_synthesis.decoders.diffusion_decoder import DiffusionDecoder
from smote_image_synthesis.decoders.diffusion_trainer import DiffusionTrainer

decoder = DiffusionDecoder(
    embedding_dim=512,
    image_shape=(3,64,64),
    num_timesteps=1000,
    noise_schedule='cosine',  # 'linear' | 'cosine' | 'quadratic'
)
trainer = DiffusionTrainer(decoder, learning_rate=1e-4)
history = trainer.train(train_embeddings, train_images, num_epochs=100)
```

U-Net with embedding conditioning. DDPM training (`predict noise ε`), DDIM sampling at inference. EMA of model weights. Slowest but highest quality.

**Decoder comparison:**

| Decoder | Speed | Visual Quality | Diversity | Notes |
|---------|-------|----------------|-----------|-------|
| Autoencoder | Fast | Good | Low | Best for prototyping |
| VAE | Medium | Good | High | Good for interpolation experiments |
| GAN | Medium | High | Medium | Sharp, may mode-collapse |
| DCGAN (frontier) | Medium | High | Medium | Class-conditional; trained jointly with encoder |
| Diffusion | Slow | Highest | High | Best quality; slow inference |

---

### QualityAssessor

`smote_image_synthesis/quality/assessor.py`

```python
from smote_image_synthesis.quality.assessor import QualityAssessor

assessor = QualityAssessor(
    metrics=['fid', 'lpips', 'ssim', 'ms_ssim', 'psnr', 'mse', 'mae'],
    compute_diversity=True,
    device=None,
)

results = assessor.evaluate_quality(
    synthetic_images,   # torch.Tensor [N, C, H, W]
    real_images,        # torch.Tensor [N, C, H, W]
    return_detailed=False,
)
```

**Metric reference:**

| Metric | Range | Better | Requirement | Notes |
|--------|-------|--------|-------------|-------|
| FID | 0 → ∞ | Lower | ≥1000 images | Inception V3 activation distance |
| LPIPS | 0 → 1 | Lower | Any | VGG perceptual distance |
| SSIM | 0 → 1 | Higher | Any | Structural similarity (luminance, contrast, structure) |
| MS-SSIM | 0 → 1 | Higher | Any | Multi-scale SSIM |
| PSNR | 0 → ∞ dB | Higher | Any | `10 log₁₀(1/MSE)` |
| MSE | 0 → ∞ | Lower | Any | Mean squared pixel error |
| MAE | 0 → ∞ | Lower | Any | Mean absolute pixel error |
| Diversity | 0 → 1 | Higher | Any | Pairwise distance + intra-class variance |

FID requires ~1000 images for a stable estimate. For smaller test sets use PSNR/SSIM.

---

### QualityReporter

`smote_image_synthesis/quality/reporter.py`

```python
from smote_image_synthesis.quality.reporter import QualityReporter

reporter = QualityReporter(
    output_dir='./quality_reports',
    report_format='html',   # 'html' | 'json' | 'text'
)

path = reporter.generate_comprehensive_report(
    quality_results=results,
    synthetic_images=syn_images,
    real_images=real_images,
    report_name='experiment_01',
)
```

Generates:
- `experiment_01.html` — full report with metric tables, image comparison grids, distribution plots, diversity histogram
- `experiment_01_metrics.csv` — flat metric export
- `experiment_01_metrics.json` — JSON for downstream parsing
- Plots: `_comparison.png`, `_distribution.png`, `_diversity.png`, `_metrics.png`

---

### Data Models

`smote_image_synthesis/data/models.py`

```python
from smote_image_synthesis.data.models import (
    EmbeddingData, SyntheticSample,
    PipelineConfig, EncoderConfig, DecoderConfig, SMOTEConfig, QualityConfig,
)
```

**`EmbeddingData`** — validated embedding container with serialisation:

```python
emb_data = EmbeddingData(
    embeddings=np.array(...),   # [N, D] float32
    labels=np.array(...),       # [N] int
    metadata={'source': 'resnet18', 'normalized': True},
)
emb_data.validate()   # checks: shape match, no NaN/Inf, ≥1 sample per class
```

**`SyntheticSample`** — tracks one synthetic with provenance:

```python
sample = SyntheticSample(
    embedding=np.array(...),
    label=1,
    parent_indices=(23, 47),
    interpolation_t=0.43,
    method='slerp',
    cluster_id=2,
)
```

**`PipelineConfig`** — hierarchical config with JSON serialisation:

```python
config = PipelineConfig(
    config_name='frontier_v1',
    encoder_config={
        'architecture': 'resnet18',
        'embedding_dim': 512,
        'pretrained': True,
        'normalize_output': True,
    },
    decoder_config={
        'decoder_type': 'dcgan',
        'base_channels': 256,
        'num_classes': 5,
        'use_self_attention': True,
    },
    smote_config={
        'k_neighbors': 5,
        'use_slerp': True,
        'density_weighted_t': True,
        'use_cluster_constraints': True,
    },
    quality_config={
        'metrics': ['psnr', 'ssim'],
        'compute_diversity': True,
    },
)
config.save_config('config.json')
loaded = PipelineConfig.load_config('config.json')
```

**`ImagePreprocessor`** (`data/preprocessor.py`):

```python
from smote_image_synthesis.data.preprocessor import ImagePreprocessor

prep = ImagePreprocessor(
    target_size=(64, 64),
    normalize=True,       # ImageNet stats
    augment=True,
)
images, labels, class_map = prep.load_from_directory('my_dataset/')
tensor = prep.process_image(pil_image)
```

---

### EMA

`smote_image_synthesis/pipeline.py` — class `_EMA`

Maintains a shadow copy of all learnable parameters of a model. Applied to both encoder and decoder at end of `_train_end_to_end()`.

```python
from smote_image_synthesis.pipeline import _EMA

ema = _EMA(model, decay=0.9999)

# Called every generator step (inside training loop)
ema.update(model)
# shadow[n] = decay × shadow[n] + (1 - decay) × param[n]

# After training: swap model to use smoothed weights
backup = ema.apply(model)

# Optionally revert (e.g. to continue training)
_EMA.restore(model, backup)
```

Decay `0.9999` means each parameter update contributes ~0.01% to the shadow. After 10000 steps the shadow is a running average over ~100 recent parameter values. Applied at the very end of training — inference always uses EMA weights.

---

### Repulsion Loss

`SynthesisPipeline._compute_repulsion_loss()` — static method

Prevents per-class mode collapse by penalising embeddings within the same class that are too close together.

```
For each class c in batch:
    For each pair (i, j) of class-c embeddings:
        d_ij = ‖e_i - e_j‖₂
        violation = max(0, margin - d_ij)
        L_repulse += violation²

L_repulse = mean over all same-class pairs
```

Applied during **Phase 2 only** (GAN phase) as:

```
G_loss += lambda_repulse × L_repulse(embeddings, labels)
```

Default `lambda_repulse=0.01`, `margin=0.3`. Only upper-triangle pairs to avoid double-counting.

---

### Adaptive λ_adv

λ_adv controls how strongly the adversarial loss contributes to the generator update. Too high early → unstable training. Too low late → generator ignores discriminator. The solution: adapt λ_adv based on whether the GAN is actually making progress.

**State variables:**

```python
W_ema = 0.0           # EMA of Wasserstein distance estimate
W_history = []        # per-epoch W_ema values
lam_adv_current = 0.05
W_window = 10         # look-back window for delta
```

**W-distance estimate** (per epoch): `W ≈ mean(D(real)) - mean(D(fake))` — the WGAN objective.

**Update rule** (applied end of each GAN-phase epoch):

```python
W_ema = 0.99 × W_ema + 0.01 × W_current

if len(W_history) >= 10:
    dW = W_history[-1] - W_history[-10]
    if dW < -0.01:   # W dropping → GAN improving
        lam_adv += 0.005   (clipped at 0.50)
    elif dW > +0.01: # W rising → GAN struggling
        lam_adv -= 0.005   (clipped at 0.01)
else:
    # Linear warmup
    frac = (global_epoch - recon_end) / (total - recon_end)
    lam_adv = 0.05 + 0.15 × frac
```

Ramp range: 0.05 → 0.20 during warmup, then ±0.005/epoch adaptive. Hard bounds `[0.01, 0.50]`.

---

## Training Details

### Phase schedule

```
Global epoch range     Phase                   Active losses
──────────────────     ──────────────────────  ─────────────────────────────────
0 → 30%               Reconstruction only      MSE + 0.5×L1 [+ 0.05×VGG]
30% → 100%            WGAN-GP + adversarial     above + λ_adv×G_adv + 0.1×FM + 0.01×repulse
```

Phase boundary is `recon_epochs = max(1, int(total_epochs × 0.3))`.

`test_frontier.py` uses `0.35` (35% recon phase) for a slightly longer warm-up.

### Discriminator training (Phase 2)

5 discriminator steps per 1 generator step (`n_critic=5`) — standard WGAN-GP.

```
D_loss = -E[D(real)] + E[D(fake)] + 10 × GP

GP (gradient penalty):
  α ~ Uniform(0,1)
  x̃ = α·real + (1-α)·fake
  GP = E[(‖∇_{x̃} D(x̃)‖₂ - 1)²]
```

### Generator training (Phase 2)

```
G_loss = MSE(recon, real)
       + 0.5  × L1(recon, real)
       + 0.05 × VGG(recon, real)          [if perceptual loss available]
       + λ_adv × (-E[D(fake)])            [adversarial]
       + 0.1  × FM_loss                   [feature matching, 3 scales]
       + 0.01 × repulsion_loss            [intra-class diversity]
```

### Multi-scale feature matching

At 3 discriminator depths (evenly spaced: early/mid/late), computes:

```
FM_loss = Σᵢ (wᵢ / Σw) × L1(feat_i(fake), feat_i(real).detach())
weights = [0.1, 0.3, 0.6]   ← texture to semantics progression
```

### LR schedule

`CosineAnnealingLR` on the generator/encoder joint optimiser:

```
lr(t) = η_min + 0.5 × (lr₀ - η_min) × (1 + cos(π × t/T_max))
T_max = total_epochs,  η_min = 1e-5
```

`last_epoch = start_epoch - 1` aligns the schedule correctly for segmented / resumed training — the LR curve is continuous across checkpoint boundaries.

### Gradient clipping

`torch.nn.utils.clip_grad_norm_(gen_params, max_norm=1.0)` on every generator step.

---

## Configuration System

All hyperparameters can be saved/loaded as JSON via `PipelineConfig`. See [Data Models](#data-models).

For the CLI, `run_pipeline.py` passes values directly to constructor calls — no separate config file needed. To reproduce a run, inspect the logged constructor arguments.

---

## Testing

```bash
# Full suite
python -m pytest tests/ -v

# Specific test files
python -m pytest tests/test_slerp_and_ema.py -v          # SLERP geometry + EMA
python -m pytest tests/test_patent_features.py -v         # vMF, density-t, clusters
python -m pytest tests/test_integration.py -v             # end-to-end pipeline
python -m pytest tests/test_resnet_encoder.py -v          # encoder shapes, L2-norm
python -m pytest tests/test_data_models.py -v             # EmbeddingData, PipelineConfig
python -m pytest tests/test_preprocessor.py -v            # image loading, augmentation

# Coverage report
python -m pytest tests/ --cov=smote_image_synthesis --cov-report=html
open htmlcov/index.html
```

| Test file | What it validates |
|-----------|------------------|
| `test_slerp_and_ema.py` | SLERP output lives on hypersphere, near-parallel fallback, EMA shadow update correctness, apply/restore roundtrip |
| `test_patent_features.py` | vMF samples are unit-normed and class-separated, density-weighted t biases toward 0.5 in sparse regions, cluster constraints restrict k-NN, ancestry metadata populated |
| `test_integration.py` | `fit → generate → evaluate` end-to-end on 100 synthetic images, output shapes, metric keys present |
| `test_resnet_encoder.py` | Output shape `[N, embedding_dim]`, L2-norm when `normalize_output=True`, freeze/unfreeze, OOM handling |
| `test_data_models.py` | `EmbeddingData.validate()` catches NaN/shape mismatch, `PipelineConfig` JSON roundtrip |
| `test_preprocessor.py` | Resize, normalise, augmentation, class-folder loading |
| `test_base_encoder.py` | Abstract interface contract |

---

## Installation

```bash
git clone <repo-url>
cd smoteforimages
pip install -r requirements.txt

# Editable install (makes smote_image_synthesis importable from anywhere)
pip install -e .
```

**Core requirements** (`requirements.txt`):

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
scikit-learn>=1.0.0
scikit-image>=0.19.0
imbalanced-learn>=0.8.0
Pillow>=8.3.0
matplotlib>=3.4.0
scipy>=1.7.0
pandas>=1.3.0
seaborn>=0.11.0
opencv-python>=4.5.0
tqdm>=4.62.0
tensorboard>=2.10.0
```

**Optional** (install separately for full quality metrics):

```bash
pip install lpips    # for LPIPS metric
```

**System requirements:**

| | Minimum | Recommended |
|--|---------|-------------|
| Python | 3.8 | 3.10+ |
| PyTorch | 1.9 | 2.0+ |
| RAM | 8 GB | 16 GB |
| GPU VRAM | — (CPU works, slow) | 6 GB+ CUDA |
| Disk | 2 GB | 10 GB |

---

## Project Structure

```
smoteforimages/
│
├── run_pipeline.py              ← main CLI: give it a folder, get synthetic images
├── test_frontier.py             ← frontier reference run on CIFAR-10 cats/dogs
├── demo_pipeline.py             ← demo with synthetic test data (no dataset needed)
├── train_cats_dogs.py           ← training example on cats/dogs
├── run_app.py                   ← interactive menu
├── requirements.txt
├── setup.py
│
├── smote_image_synthesis/       ← core library (installable)
│   ├── __init__.py              exports: SynthesisPipeline + config dataclasses
│   ├── pipeline.py              SynthesisPipeline, _EMA, discriminator builder,
│   │                            _compute_repulsion_loss, _train_end_to_end
│   ├── error_handling.py        ErrorRecoveryManager (GPU OOM, NaN, checkpoint)
│   ├── experiment_tracking.py   ExperimentTracker (TensorBoard)
│   │
│   ├── encoders/
│   │   ├── base.py              ImageEncoder abstract base
│   │   └── resnet_encoder.py    ResNetEncoder — resnet18/50/101, L2Normalize
│   │
│   ├── decoders/
│   │   ├── base.py              BaseDecoder abstract interface
│   │   ├── dcgan_decoder.py     DCGANDecoder + SelfAttention2d   ← FRONTIER
│   │   ├── autoencoder_decoder.py + autoencoder_trainer.py
│   │   ├── vae_decoder.py       + vae_trainer.py
│   │   ├── gan_decoder.py       + gan_trainer.py
│   │   └── diffusion_decoder.py + diffusion_trainer.py
│   │
│   ├── smote/
│   │   └── constrained_smote.py  SLERP, vMF, density-t, clusters, outliers, ancestry
│   │
│   ├── quality/
│   │   ├── assessor.py           FID · LPIPS · SSIM · PSNR · diversity
│   │   └── reporter.py           HTML/JSON/CSV reports + plots
│   │
│   └── data/
│       ├── models.py             EmbeddingData, SyntheticSample, PipelineConfig
│       └── preprocessor.py       ImagePreprocessor
│
├── tests/
│   ├── test_slerp_and_ema.py
│   ├── test_patent_features.py
│   ├── test_integration.py
│   ├── test_resnet_encoder.py
│   ├── test_data_models.py
│   ├── test_preprocessor.py
│   └── test_base_encoder.py
│
├── notebooks/
│   ├── 01_basic_pipeline_usage.ipynb
│   ├── 02_decoder_architectures.ipynb
│   └── 03_custom_dataset_integration.ipynb
│
├── docs/
│   ├── API_REFERENCE.md
│   ├── PATENT_TECHNICAL_DISCLOSURE.md
│   └── patent/patent_application.tex
│
├── api/                         REST API (FastAPI + WebSocket) — see api/README.md
├── frontier_output/             saved weights + comparison grids from test_frontier.py
├── checkpoints/                 training checkpoints
└── data/                        datasets (CIFAR-10 auto-downloaded by test_frontier.py)
```

---

## Documentation Index

| Document | Contents |
|----------|---------|
| **This file** | Full architecture, every component with math, run procedure, training details |
| [smote_image_synthesis/README.md](smote_image_synthesis/README.md) | Library API reference with method signatures |
| [api/README.md](api/README.md) | REST endpoints, WebSocket protocol, file storage layout |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | Python class/method reference |
| [docs/PATENT_TECHNICAL_DISCLOSURE.md](docs/PATENT_TECHNICAL_DISCLOSURE.md) | Novelty disclosure for SLERP-SMOTE + vMF-SMOTE |
| [REAL_IMAGE_USAGE_GUIDE.md](REAL_IMAGE_USAGE_GUIDE.md) | Step-by-step for real image workflows |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Feature completion checklist |
| `notebooks/01_basic_pipeline_usage.ipynb` | API walkthrough with code |
| `notebooks/02_decoder_architectures.ipynb` | Side-by-side decoder comparison |
| `notebooks/03_custom_dataset_integration.ipynb` | Custom dataset integration guide |
