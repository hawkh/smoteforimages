import numpy as np
import pytest
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE

def test_filter_by_distance():
    smote = ConstrainedSMOTE(max_distance_threshold=1.0)

    # Mock data
    smote.embeddings = np.array([
        [0.0, 0.0],
        [10.0, 10.0]
    ])
    smote.labels = np.array([0, 1])

    synthetic_embeddings = np.array([
        [0.5, 0.5],   # Close to 0
        [5.0, 5.0],   # Far from both
        [10.2, 10.2]  # Close to 1
    ])
    synthetic_labels = np.array([0, 0, 1])

    filtered_emb, filtered_lbl = smote._filter_by_distance(synthetic_embeddings, synthetic_labels)

    assert len(filtered_emb) == 2
    assert np.array_equal(filtered_lbl, np.array([0, 1]))
    assert np.allclose(filtered_emb[0], [0.5, 0.5])
    assert np.allclose(filtered_emb[1], [10.2, 10.2])

def test_filter_by_distance_no_threshold():
    smote = ConstrainedSMOTE(max_distance_threshold=None)
    synthetic_embeddings = np.array([[0.5, 0.5]])
    synthetic_labels = np.array([0])

    filtered_emb, filtered_lbl = smote._filter_by_distance(synthetic_embeddings, synthetic_labels)

    assert len(filtered_emb) == 1
    assert np.array_equal(filtered_emb, synthetic_embeddings)
    assert np.array_equal(filtered_lbl, synthetic_labels)
