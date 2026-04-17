# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

SMOTE for Images: synthetic image generation by applying SMOTE in **embedding space** (not pixel space) using geodesic SLERP interpolation on the unit hypersphere. The key constraint is that embeddings must be L2-normalized before any interpolation.

## Commands

### Python Library / Pipeline

```bash
pip install -r requirements.txt
# or editable install
pip install -e .

# Reference run (CIFAR-10 cats & dogs, all frontier features enabled)
python test_frontier.py

# General CLI for any dataset (class subdirs under --data-dir)
python run_pipeline.py --data-dir my_dataset/ --epochs 200 --image-size 64

# Interactive menu
python run_app.py

# Demo without real data
python demo_pipeline.py --n-samples 200 --generate-report
```

### REST API

```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Docs: http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm install
npm run dev   # http://localhost:3000
```

### Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/test_patent_features.py -v    # vMF, density-weighted t, cluster constraints
python -m pytest tests/test_slerp_and_ema.py -v      # SLERP geometry + EMA correctness
python -m pytest tests/test_integration.py -v        # Full end-to-end pipeline
python -m pytest tests/ --cov=smote_image_synthesis --cov-report=html
```

## Architecture

### Data Flow

```
Images
  → ResNetEncoder (backbone + linear projection → L2 normalize → unit hypersphere)
  → Embeddings [N, 512]
      ├─→ ConstrainedSMOTE  →  Synthetic embeddings
      └─→ Joint E2E Training (Phase 1: MSE/L1/VGG, Phase 2: WGAN-GP)
  → DCGANDecoder  →  Images
  → QualityAssessor  →  FID / LPIPS / SSIM / PSNR / diversity
```

### Key Modules

**`smote_image_synthesis/pipeline.py`** — `SynthesisPipeline` orchestrates everything. `_EMA` class wraps encoder+decoder with weight averaging (decay=0.9999). Two-phase training: 0–30% epochs uses reconstruction losses only; 30–100% adds WGAN-GP with adaptive `λ_adv` controlled by EMA-smoothed Wasserstein distance.

**`smote_image_synthesis/smote/constrained_smote.py`** — All interpolation variants live here. SLERP is the default; vMF overrides it when enabled. Density-weighted `t` biases interpolation toward sparse regions using k-NN density estimates. IsolationForest filters outliers with up to 3 retries. Ancestry tracking records `(parent_indices, t, method, cluster)` per synthetic.

**`smote_image_synthesis/encoders/resnet_encoder.py`** — ResNet18/50/101 backbone + linear head → 512-dim L2-normalized embedding. L2 normalization is **required** for SLERP geometry to be valid.

**`smote_image_synthesis/decoders/dcgan_decoder.py`** — The frontier decoder. Generator: Linear(576) → DeConvBlocks. Self-attention (SAGAN) at 16×16. Class conditioning via 64-dim class embedding concatenated to `z`. Discriminator uses spectral normalization on all Conv2d layers and projection discriminator (Miyata & Koyama 2018). Training: 5 discriminator steps per generator step.

**`smote_image_synthesis/quality/`** — `assessor.py` computes metrics (FID requires ~1000 images); `reporter.py` generates HTML/JSON/CSV + image grids.

**`api/`** — FastAPI backend. `services/training_runner.py` runs training in a background thread. WebSocket at `/ws/training/{run_id}` streams `{epoch, loss, phase}` JSON. `services/pipeline_manager.py` manages run lifecycle state.

**`frontend/src/lib/use-training-ws.ts`** — React hook consuming the WebSocket for live training progress.

### Decoder Variants

Five decoder types exist (`autoencoder_decoder.py`, `vae_decoder.py`, `gan_decoder.py`, `diffusion_decoder.py`, `dcgan_decoder.py`). Each has a standalone `*_trainer.py`. The frontier decoder is `DCGANDecoder` — use this for new work.

### Extending

- **New encoder**: subclass `encoders/base.py:ImageEncoder`
- **New decoder**: subclass `decoders/base.py:BaseDecoder`
- **New metric**: add to `quality/assessor.py:QualityAssessor`
- **New interpolation**: add method to `smote/constrained_smote.py:ConstrainedSMOTE`

## Frontier Features (all on by default in `test_frontier.py`)

| Flag | Default | Purpose |
|------|---------|---------|
| `normalize_output=True` | on | L2-normalize → unit hypersphere (required for SLERP) |
| `use_slerp=True` | on | Geodesic interpolation instead of linear |
| `use_vmf=False` | off | vMF distribution sampling (overrides SLERP) |
| `density_weighted_t=True` | on | t biased toward sparse regions |
| `use_cluster_constraints=True` | on | SLERP within K-means clusters only |
| `use_outlier_detection=True` | on | IsolationForest + retry (3×) |
| `track_ancestry=True` | on | Record parent indices, t, method, cluster |
| `use_self_attention=True` | on | SAGAN at 16×16 in decoder |

## `run_pipeline.py` Key Flags

```
--data-dir          Root folder with class subdirectories (required)
--output-dir        Output location (default: pipeline_output)
--image-size        Resize to N×N (default: 64)
--epochs            Training epochs (default: 150)
--architecture      resnet18|resnet50 (default: resnet18)
--balance-to        majority|mean|integer (default: majority)
--resume            Resume from checkpoint directory
--save-every        Checkpoint interval in epochs (default: 0 = end only)
```

## Output Structure

```
output_dir/
├── real_samples.png / synthetic_samples.png / comparison.png
├── class_balance.png
├── quality_metrics.json
├── pipeline_encoder.pth
└── pipeline_decoder.pth
```
