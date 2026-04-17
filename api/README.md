# api/ — FastAPI Backend

REST API and WebSocket server for the SMOTE image synthesis pipeline.

Back to [project root](../README.md)

---

## Start the server

```bash
# From project root:
python start_chlorophyll_api.py

# Or manually:
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- REST API: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/health`

---

## Endpoints

### Datasets

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/datasets/upload` | Upload a zipped dataset (class subdirs inside zip) |
| `GET` | `/api/datasets` | List all uploaded datasets |
| `DELETE` | `/api/datasets/{dataset_id}` | Delete a dataset |

**Upload format**: zip file containing `class_name/image.jpg` paths. The API unpacks to `api/static/datasets/<dataset_id>/`.

### Pipeline

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/pipeline/configure` | Create a new run (returns `run_id`) |
| `POST` | `/api/pipeline/train/{run_id}` | Start background training |
| `GET` | `/api/pipeline/status/{run_id}` | Poll training status |
| `POST` | `/api/pipeline/generate/{run_id}` | Generate synthetic images after training |
| `GET` | `/api/pipeline/generate/{run_id}/results` | List generated image URLs |

**Configure body** (`PipelineConfigRequest`):
```json
{
  "dataset_id": "string",
  "image_size": 64,
  "epochs": 100,
  "architecture": "resnet18",
  "embedding_dim": 512,
  "base_channels": 256,
  "batch_size": 32,
  "balance_to": "majority"
}
```

**Status response**:
```json
{
  "run_id": "string",
  "status": "pending|training|done|failed",
  "epoch": 42,
  "total_epochs": 100,
  "phase": "reconstruction|adversarial",
  "losses": {"gen": 0.32, "disc": 0.18}
}
```

### Quality

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/quality/evaluate/{run_id}` | Get quality metrics for a completed run |

### WebSocket — Live Training Updates

```
ws://localhost:8000/ws/training/{run_id}
```

Messages (JSON):
```json
{ "type": "progress", "data": { "epoch": 10, "total": 100, "phase": "adversarial", "losses": {...} } }
{ "type": "complete", "data": { "message": "Training complete" } }
{ "type": "error",    "data": { "message": "..." } }
```

---

## File Storage

```
api/static/
    datasets/
        <dataset_id>/
            class_a/   img1.jpg  img2.jpg  ...
            class_b/   img1.jpg  ...
    outputs/
        <run_id>/
            synthetic_samples.png
            real_samples.png
            comparison.png
            quality_metrics.json
            pipeline_encoder.pth
            pipeline_decoder.pth
```

Directories are created automatically on server startup (see `api/config.py`).

---

## Configuration (`api/config.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_IMAGE_SIZE` | 64 | Pixels |
| `DEFAULT_EMBEDDING_DIM` | 512 | Encoder output |
| `DEFAULT_BATCH_SIZE` | 32 | Training batch |
| `DEFAULT_EPOCHS` | 100 | Training epochs |
| `DEFAULT_BASE_CHANNELS` | 256 | DCGAN channels |
| `DEFAULT_K_NEIGHBORS` | 5 | SMOTE neighbors |
| `DEFAULT_LR` | 2e-4 | Learning rate |
| `DEVICE` | auto | CUDA if available, else CPU |

---

## Module Structure

```
api/
├── main.py                  # FastAPI app, CORS, router registration
├── config.py                # paths, defaults, device detection
├── models/
│   ├── requests.py          # Pydantic request models
│   └── responses.py         # Pydantic response models
├── routes/
│   ├── datasets.py          # /api/datasets
│   ├── pipeline.py          # /api/pipeline
│   ├── quality.py           # /api/quality
│   └── docs.py              # /api/docs (patent + API reference)
├── services/
│   ├── pipeline_manager.py  # singleton run lifecycle manager
│   ├── training_runner.py   # background thread for pipeline.fit()
│   ├── dataset_service.py   # zip upload + file I/O
│   └── quality_service.py   # wraps QualityAssessor
└── ws/
    └── training_ws.py       # WebSocket endpoint
```

---

## CORS

Allowed origins: `http://localhost:3000`, `http://127.0.0.1:3000` (Next.js dev server).

To add production origins, edit `allow_origins` in `api/main.py`.
