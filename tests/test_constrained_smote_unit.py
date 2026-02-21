
import unittest
import numpy as np
import sys
from pathlib import Path

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE

class TestConstrainedSMOTEUnit(unittest.TestCase):
    def setUp(self):
        self.smote = ConstrainedSMOTE(
            k_neighbors=2,
            sampling_strategy='auto',
            max_distance_threshold=2.0,
            use_clustering=False,
            normalize_embeddings=False
        )

        # Setup mock embeddings
        # Class 0: Center at (0, 0)
        # Class 1: Center at (10, 10)
        self.embeddings = np.array([
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], # Class 0
            [10.0, 10.0], [10.1, 10.1], [10.2, 10.0] # Class 1
        ])
        self.labels = np.array([0, 0, 0, 1, 1, 1])

        self.smote.embeddings = self.embeddings
        self.smote.labels = self.labels
        self.smote.is_fitted = True

    def test_filter_by_distance_valid(self):
        # Samples close to original class 0
        synthetic_embeddings = np.array([
            [0.05, 0.05], # Very close to (0,0) -> Should be kept
            [0.5, 0.5],   # Dist to (0,0) is ~0.7 -> Should be kept (thresh 2.0)
            [10.05, 10.05] # Very close to (10,10) class 1 -> Should be kept if label is 1
        ])
        synthetic_labels = np.array([0, 0, 1])

        filtered_emb, filtered_lbl = self.smote._filter_by_distance(
            synthetic_embeddings, synthetic_labels
        )

        self.assertEqual(len(filtered_emb), 3)
        np.testing.assert_array_equal(filtered_emb, synthetic_embeddings)
        np.testing.assert_array_equal(filtered_lbl, synthetic_labels)

    def test_filter_by_distance_invalid(self):
        # Samples far from original class 0
        synthetic_embeddings = np.array([
            [5.0, 5.0],   # Far from both (dist to 0,0 is ~7) -> Should be removed
            [0.05, 0.05], # Close -> Kept
        ])
        synthetic_labels = np.array([0, 0])

        filtered_emb, filtered_lbl = self.smote._filter_by_distance(
            synthetic_embeddings, synthetic_labels
        )

        self.assertEqual(len(filtered_emb), 1)
        np.testing.assert_array_equal(filtered_emb, np.array([[0.05, 0.05]]))
        np.testing.assert_array_equal(filtered_lbl, np.array([0]))

    def test_filter_by_distance_mixed(self):
        # Mix of valid and invalid for different classes
        synthetic_embeddings = np.array([
            [0.1, 0.1],   # Class 0, valid
            [5.0, 5.0],   # Class 0, invalid
            [10.1, 10.1], # Class 1, valid
            [0.0, 0.0]    # Class 1, invalid (close to class 0 but labeled 1, far from class 1)
        ])
        synthetic_labels = np.array([0, 0, 1, 1])

        filtered_emb, filtered_lbl = self.smote._filter_by_distance(
            synthetic_embeddings, synthetic_labels
        )

        self.assertEqual(len(filtered_emb), 2)
        # Expected: index 0 and 2
        expected_emb = np.array([[0.1, 0.1], [10.1, 10.1]])
        expected_lbl = np.array([0, 1])

        np.testing.assert_array_almost_equal(filtered_emb, expected_emb)
        np.testing.assert_array_equal(filtered_lbl, expected_lbl)

    def test_filter_by_distance_empty(self):
        synthetic_embeddings = np.array([])
        synthetic_labels = np.array([])

        filtered_emb, filtered_lbl = self.smote._filter_by_distance(
            synthetic_embeddings, synthetic_labels
        )

        self.assertEqual(len(filtered_emb), 0)
        self.assertEqual(len(filtered_lbl), 0)

    def test_filter_by_distance_no_threshold(self):
        self.smote.max_distance_threshold = None
        synthetic_embeddings = np.array([[100.0, 100.0]])
        synthetic_labels = np.array([0])

        filtered_emb, filtered_lbl = self.smote._filter_by_distance(
            synthetic_embeddings, synthetic_labels
        )

        self.assertEqual(len(filtered_emb), 1)

if __name__ == '__main__':
    unittest.main()
