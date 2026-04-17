"""SMOTE Image Synthesis API — FastAPI application."""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure project root is on sys.path so `smote_image_synthesis` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.config import STATIC_DIR
from api.routes import datasets, pipeline, quality, docs
from api.ws import training_ws


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    print(f"SMOTE API starting — static dir: {STATIC_DIR}")
    yield
    print("SMOTE API shutting down")


app = FastAPI(
    title="SMOTE Image Synthesis API",
    version="1.0.0",
    description="API for class-conditional synthetic image generation via SLERP/vMF-SMOTE",
    lifespan=lifespan,
)

# CORS — allow Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Include routers
app.include_router(datasets.router)
app.include_router(pipeline.router)
app.include_router(quality.router)
app.include_router(docs.router)
app.include_router(training_ws.router)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    import torch
    return {
        "status": "ok",
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "cuda_available": torch.cuda.is_available(),
    }
