import pytest
import numpy as np
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE

def test_filter_by_distance_correctness():
    # Setup deterministic dummy data
    np.random.seed(42)
    N_ORIGINAL = 100
    N_SYNTHETIC = 50
    DIM = 64
    N_CLASSES = 3

    smote = ConstrainedSMOTE(max_distance_threshold=15.0)

    # Original embeddings
    smote.embeddings = np.random.randn(N_ORIGINAL, DIM)
    smote.labels = np.random.randint(0, N_CLASSES, N_ORIGINAL)

    # Synthetic embeddings
    synth_emb = np.random.randn(N_SYNTHETIC, DIM)
    synth_labels = np.random.randint(0, N_CLASSES, N_SYNTHETIC)

    # Run the optimized function
    filtered_emb, filtered_labels = smote._filter_by_distance(synth_emb, synth_labels)

    # Basic assertions
    assert filtered_emb.ndim == 2
    assert filtered_emb.shape[1] == DIM
    assert len(filtered_emb) <= N_SYNTHETIC
    assert len(filtered_emb) == len(filtered_labels)

def test_generate_synthetic():
    np.random.seed(42)
    N_ORIGINAL = 50
    DIM = 16
    N_CLASSES = 2

    embeddings = np.random.randn(N_ORIGINAL, DIM)
    labels = np.random.randint(0, N_CLASSES, N_ORIGINAL)
    # Ensure imbalance
    labels[:40] = 0
    labels[40:] = 1

    smote = ConstrainedSMOTE(max_distance_threshold=100.0)
    smote.fit(embeddings, labels)

    synth_emb, synth_labels = smote.generate_synthetic()

    assert synth_emb.shape[0] > 0
    assert synth_emb.shape[1] == DIM
    assert len(synth_emb) == len(synth_labels)
