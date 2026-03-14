#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Static compile check"
python -m compileall -q smote_image_synthesis tests

echo "[2/4] Core unit/integration tests"
pytest -q tests test_basic_structure.py

echo "[3/4] Pipeline smoke test"
python - <<'PY'
import numpy as np
import torch
from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder
from smote_image_synthesis.decoders.autoencoder_decoder import AutoencoderDecoder
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE
from smote_image_synthesis.quality.assessor import QualityAssessor
from smote_image_synthesis.pipeline import SynthesisPipeline

torch.manual_seed(7)
np.random.seed(7)
images = torch.rand(12, 3, 64, 64)
labels = np.array([0] * 8 + [1] * 4)

pipeline = SynthesisPipeline(
    encoder=ResNetEncoder(architecture='resnet18', embedding_dim=128, pretrained=False, device=torch.device('cpu')),
    decoder=AutoencoderDecoder(embedding_dim=128, image_shape=(3, 64, 64), device=torch.device('cpu')),
    smote=ConstrainedSMOTE(k_neighbors=3, use_clustering=False, normalize_embeddings=False),
    quality_assessor=QualityAssessor(metrics=['mse', 'mae'], device=torch.device('cpu')),
)

pipeline.fit(images, labels, train_decoder=False)
synthetic_images, synthetic_labels = pipeline.generate_synthetic_images(4)
assert len(synthetic_images) == len(synthetic_labels)
results = pipeline.evaluate_quality(synthetic_images[:2], images[:2])
assert 'mse' in results
print('pipeline-smoke-ok')
PY

echo "[4/4] Deep audit report"
python scripts/full_repo_audit.py --output audit_reports/latest_audit.md --skip-tests

echo "All checks completed."
