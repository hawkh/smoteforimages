"""API configuration — paths, device detection, defaults."""

import os
import torch
from pathlib import Path

# Root of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Static file directories
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATASETS_DIR = STATIC_DIR / "datasets"
OUTPUTS_DIR = STATIC_DIR / "outputs"

# Docs
DOCS_DIR = PROJECT_ROOT / "docs"
PATENT_MD = DOCS_DIR / "PATENT_TECHNICAL_DISCLOSURE.md"
PATENT_TEX = DOCS_DIR / "patent" / "patent_application.tex"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defaults
DEFAULT_IMAGE_SIZE = 64
DEFAULT_EMBEDDING_DIM = 512
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
DEFAULT_BASE_CHANNELS = 256
DEFAULT_K_NEIGHBORS = 5
DEFAULT_LR = 2e-4

# Ensure directories exist
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
