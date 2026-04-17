# smote_image_synthesis — Library Reference

Core Python library for SMOTE-based synthetic image generation.

→ [Project root README](../README.md) for architecture overview and run instructions.

---

## Module Map

```
smote_image_synthesis/
├── __init__.py              # exports SynthesisPipeline + all config dataclasses
├── pipeline.py              # SynthesisPipeline, _EMA, _build_discriminator
├── error_handling.py        # ErrorRecoveryManager
├── experiment_tracking.py   # ExperimentTracker (TensorBoard)
├── encoders/
│   ├── base.py              # ImageEncoder abstract base
│   └── resnet_encoder.py    # ResNetEncoder (resnet18/50/101)
├── decoders/
│   ├── base.py              # BaseDecoder abstract interface
│   ├── autoencoder_decoder.py   + autoencoder_trainer.py
│   ├── vae_decoder.py           + vae_trainer.py
│   ├── gan_decoder.py           + gan_trainer.py
│   ├── dcgan_decoder.py         # frontier decoder — used by run_pipeline.py
│   └── diffusion_decoder.py     + diffusion_trainer.py
├── smote/
│   └── constrained_smote.py # ConstrainedSMOTE — all interpolation methods
└── quality/
    ├── assessor.py          # QualityAssessor — FID, LPIPS, SSIM, PSNR, diversity
    └── reporter.py          # QualityReporter — HTML/JSON/CSV reports
```

---

## Public API

```python
from smote_image_synthesis import (
    SynthesisPipeline,
    PipelineConfig, EncoderConfig, DecoderConfig, SMOTEConfig, QualityConfig,
    EmbeddingData, SyntheticSample,
)
```

---

## SynthesisPipeline

`pipeline.py` — main orchestrator.

### Constructor

```python
SynthesisPipeline(
    encoder,           # ResNetEncoder (or any ImageEncoder)
    decoder,           # DCGANDecoder (or any BaseDecoder)
    smote,             # ConstrainedSMOTE
    quality_assessor,  # QualityAssessor | None (creates default if None)
)
```

Validates that `encoder.get_embedding_dim() == decoder.get_embedding_dim()` on construction.

### fit()

```python
pipeline.fit(
    images,                  # torch.Tensor [N, C, H, W] — on target device
    labels,                  # np.ndarray [N] — integer class indices
    train_decoder=True,
    decoder_epochs=100,      # epochs for THIS call
    start_epoch=0,           # global epoch offset (for segmented/resumed training)
    total_epochs=0,          # global total (0 = same as decoder_epochs)
)
```

**For `DCGANDecoder` and `AutoencoderDecoder`:** calls `_train_end_to_end()` — joint encoder+decoder optimisation with WGAN-GP, feature matching, repulsion loss, EMA.

**For `VAEDecoder`, `GANDecoder`, `DiffusionDecoder`:** fits SMOTE on encoder embeddings first, then calls the decoder's standalone trainer.

After decoder training, always fits SMOTE on the current encoder embeddings.

### generate_synthetic_images()

```python
syn_images, syn_labels = pipeline.generate_synthetic_images(
    n_samples=500,
    target_classes=None,     # list[int] or None for all classes
    return_metadata=False,
)
```

Returns `torch.Tensor [N, C, H, W]` and `np.ndarray [N]`. Passes SMOTE-generated embeddings through the decoder in batches of 32.

### evaluate_quality()

```python
metrics = pipeline.evaluate_quality(synthetic_images, real_images)
# dict: {metric_name: float}
```

### save_pipeline() / load_pipeline()

```python
pipeline.save_pipeline("output/pipeline")
# writes: output/pipeline_encoder.pth, output/pipeline_decoder.pth

pipeline.load_pipeline("output/pipeline")
```

---

## _EMA (Exponential Moving Average)

`pipeline.py` — also importable directly.

```python
from smote_image_synthesis.pipeline import _EMA

ema = _EMA(model, decay=0.9999)

# During training (call each generator step)
ema.update(model)

# After training
backup = ema.apply(model)    # swap to EMA weights; returns original
_EMA.restore(model, backup)  # optionally revert
```

Used internally for both encoder and decoder. Applied at end of `_train_end_to_end()`.

---

## ResNetEncoder

`encoders/resnet_encoder.py`

```python
from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder

encoder = ResNetEncoder(
    architecture='resnet18',  # 'resnet18' | 'resnet50' | 'resnet101'
    embedding_dim=512,
    pretrained=True,
    freeze_backbone=False,    # True = only train Linear projection head
    normalize_output=True,    # L2-normalise → required for SLERP
    dropout_rate=0.1,
    device=None,
)

embs = encoder.encode(images)           # [N, 512]
embs = encoder.encode_batch(images, 32) # memory-safe batched encode
dim  = encoder.get_embedding_dim()      # 512
```

`normalize_output=True` is the default for frontier use. When enabled, set `normalize_embeddings=False` on `ConstrainedSMOTE` to avoid double-normalising.

---

## DCGANDecoder

`decoders/dcgan_decoder.py`

```python
from smote_image_synthesis.decoders.dcgan_decoder import DCGANDecoder

decoder = DCGANDecoder(
    embedding_dim=512,
    image_shape=(3, 64, 64),
    base_channels=256,       # 256 → 4×4 to H×W upsample path
    num_classes=0,           # 0 = unconditional; >0 = class-conditional
    class_embed_dim=64,      # dim of class embedding concatenated to z
    use_self_attention=True, # SAGAN non-local attention at 16×16
    device=None,
)
```

**Generator network layers:**

```
[z (512) | class_emb (64)] → Linear(576, ch0×16) → BN+ReLU → Reshape(ch0, 4, 4)
→ DeConvBlock  4→8   (ch0   → ch0//2)
→ DeConvBlock  8→16  (ch0//2 → ch0//4)
→ SelfAttention2d         ← at 16×16
→ DeConvBlock  16→32 (ch0//4 → ch0//8)
→ DeConvBlock  32→64 (ch0//8 → ch0//16)    [if image_size=64]
→ ConvTranspose2d → Tanh → [3, 64, 64]
```

`SelfAttention2d` — SAGAN (Zhang et al. 2019). Residual term scaled by `gamma=0` at init; learns gradually. `Q`, `K`, `V` projections with `in_ch//8` bottleneck.

Trained via `SynthesisPipeline._train_end_to_end()` — not via a standalone trainer.

---

## ConstrainedSMOTE

`smote/constrained_smote.py`

```python
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE

smote = ConstrainedSMOTE(
    k_neighbors=5,
    sampling_strategy='auto',       # 'auto' | 'minority' | dict {cls: n}

    # ── Interpolation ─────────────────────────────────────────────────────
    use_slerp=True,                 # geodesic SLERP on unit sphere
    use_vmf=False,                  # vMF distribution sampling (overrides slerp)
    vmf_concentration_scale=1.0,    # κ scale: >1 tighter, <1 broader

    # ── Frontier improvements ─────────────────────────────────────────────
    density_weighted_t=True,        # bias t toward sparse geodesic regions
    use_cluster_constraints=True,   # SLERP within-cluster only
    use_outlier_detection=True,     # IsolationForest; retry up to 3×
    track_ancestry=True,            # record provenance per synthetic sample
    boundary_detection_method='isolation',  # 'density'|'svm'|'isolation'

    # ── Clustering ────────────────────────────────────────────────────────
    use_clustering=True,
    clustering_method='kmeans',     # 'kmeans'|'dbscan'|'hierarchical'|'gmm'
    n_clusters=None,                # auto-determined from class size if None

    normalize_embeddings=False,     # False when encoder already L2-normalises
    random_state=42,
)
```

**Key methods:**

```python
smote.fit(embeddings_np, labels)           # np.ndarray [N, D], np.ndarray [N]

# Standard imbalanced-learn interface
syn_embs, syn_labs = smote.fit_resample(embeddings_np, labels)

# Direct generation with count control
syn_embs, syn_labs = smote.generate_synthetic(n_samples=500)

# With provenance metadata
syn_embs, syn_labs, metadata = smote.generate_synthetic(n_samples=500, return_metadata=True)
# metadata: list[dict] with keys 'parent_indices', 't', 'method', 'cluster_id'

# Inspect embedding space health
is_valid, report = smote.validate_embedding_space(test_embeddings)
```

**SLERP formula:**

```
θ = arccos(clamp(e1 · e2, -1, 1))
e_syn = sin((1-t)θ)/sinθ · e1  +  sin(tθ)/sinθ · e2
(linear fallback when θ < 1e-6)
```

**vMF sampling:** `p(x; μ, κ) ∝ exp(κ · μᵀx)`. MLE fit per class via approximate κ estimator. Samples drawn via Wood's rejection algorithm.

**Density-weighted t:** per-point k-NN density estimate d(x). Low-density → `t ~ Beta(2,2)` peaked at 0.5 (gap-filling). High-density → `t ~ Uniform(0,1)`.

---

## QualityAssessor

`quality/assessor.py`

```python
from smote_image_synthesis.quality.assessor import QualityAssessor

assessor = QualityAssessor(
    metrics=['fid', 'lpips', 'ssim', 'ms_ssim', 'psnr', 'mse', 'mae'],
    compute_diversity=True,
    device=None,
)

results = assessor.evaluate_quality(
    synthetic_images,    # torch.Tensor [N, C, H, W], range [-1,1] or [0,1]
    real_images,
    return_detailed=False,
)
```

FID requires ~1000 images for a statistically stable estimate. For smaller batches use PSNR/SSIM instead.

---

## QualityReporter

`quality/reporter.py`

```python
from smote_image_synthesis.quality.reporter import QualityReporter

reporter = QualityReporter(output_dir='./reports', report_format='html')
path = reporter.generate_comprehensive_report(
    quality_results=results,
    synthetic_images=syn_images,
    real_images=real_images,
    report_name='exp_01',
)
# Writes: exp_01.html, exp_01_metrics.csv, exp_01_metrics.json
# Plots: comparison grid, metric bar chart, distribution plots, diversity histogram
```

---

## Data Models

`data/models.py`

```python
from smote_image_synthesis.data.models import EmbeddingData, SyntheticSample, PipelineConfig

# Validated embedding container
emb_data = EmbeddingData(
    embeddings=np.array(...),   # [N, D]
    labels=np.array(...),       # [N]
    metadata={'source': 'resnet18', 'normalized': True},
)
emb_data.validate()  # checks shapes, NaNs, class counts

# Synthetic sample tracker
sample = SyntheticSample(
    embedding=np.array(...),
    label=1,
    parent_indices=(23, 47),
    interpolation_t=0.43,
    method='slerp',
)

# Hierarchical config with JSON serialisation
config = PipelineConfig(
    config_name='exp_01',
    encoder_config={'architecture': 'resnet18', 'embedding_dim': 512, 'normalize_output': True},
    decoder_config={'decoder_type': 'dcgan', 'base_channels': 256, 'num_classes': 5},
    smote_config={'k_neighbors': 5, 'use_slerp': True, 'density_weighted_t': True},
    quality_config={'metrics': ['psnr', 'ssim'], 'compute_diversity': True},
)
config.save_config('config.json')
loaded = PipelineConfig.load_config('config.json')
```

---

## ImagePreprocessor

`data/preprocessor.py`

```python
from smote_image_synthesis.data.preprocessor import ImagePreprocessor

prep = ImagePreprocessor(
    target_size=(64, 64),
    normalize=True,       # ImageNet mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
    augment=True,         # random hflip, color jitter
)

# Load folder dataset (class subdirs)
images, labels, class_map = prep.load_from_directory('my_dataset/')

# Process single PIL image
tensor = prep.process_image(pil_image)
```

---

## ErrorRecoveryManager

`error_handling.py`

Wraps pipeline operations. On `torch.cuda.OutOfMemoryError` clears CUDA cache and retries on CPU. On NaN loss logs a warning and skips the batch. On repeated failures falls back to the last good checkpoint.

```python
from smote_image_synthesis.error_handling import ErrorRecoveryManager

with ErrorRecoveryManager(checkpoint_dir='checkpoints/') as mgr:
    pipeline.fit(images, labels, decoder_epochs=100)
```

---

## ExperimentTracker

`experiment_tracking.py`

```python
from smote_image_synthesis.experiment_tracking import ExperimentTracker

tracker = ExperimentTracker('frontier_v1', log_dir='runs/')
tracker.log_metrics({'g_loss': 0.084, 'd_loss': 0.21, 'lambda_adv': 0.18}, step=200)
tracker.log_images('synthetic_grid', syn_images[:16], step=200)
tracker.close()
# View: tensorboard --logdir runs/
```
