
import unittest
import numpy as np
import sys
from pathlib import Path

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE

class TestConstrainedSMOTEFiltering(unittest.TestCase):
    """Test the filtering logic in ConstrainedSMOTE."""

    def setUp(self):
        """Set up test fixtures."""
        self.embedding_dim = 64
        self.n_samples = 100
        self.n_synthetic = 200
        self.n_classes = 3

        # Create base embeddings and labels
        self.embeddings = np.random.randn(self.n_samples, self.embedding_dim)
        self.labels = np.random.randint(0, self.n_classes, self.n_samples)

        # Create synthetic data
        self.synthetic_embeddings = np.random.randn(self.n_synthetic, self.embedding_dim)
        self.synthetic_labels = np.random.randint(0, self.n_classes, self.n_synthetic)

    def test_filter_by_distance_no_threshold(self):
        """Test filtering when no threshold is set."""
        smote = ConstrainedSMOTE(max_distance_threshold=None)
        smote.embeddings = self.embeddings
        smote.labels = self.labels

        filtered_embeddings, filtered_labels = smote._filter_by_distance(
            self.synthetic_embeddings, self.synthetic_labels
        )

        # Should return all samples
        self.assertEqual(len(filtered_embeddings), self.n_synthetic)
        np.testing.assert_array_equal(filtered_embeddings, self.synthetic_embeddings)
        np.testing.assert_array_equal(filtered_labels, self.synthetic_labels)

    def test_filter_by_distance_with_threshold(self):
        """Test filtering when threshold is set."""
        # Create a specific scenario where we know which samples should be filtered

        # Original: one point at origin for class 0
        embeddings = np.zeros((1, 2))
        labels = np.array([0])

        # Synthetic:
        # 1. At (0.1, 0) - distance 0.1
        # 2. At (1.0, 0) - distance 1.0
        # 3. At (2.0, 0) - distance 2.0
        synthetic_embeddings = np.array([
            [0.1, 0.0],
            [1.0, 0.0],
            [2.0, 0.0]
        ])
        synthetic_labels = np.array([0, 0, 0])

        smote = ConstrainedSMOTE(max_distance_threshold=0.5)
        smote.embeddings = embeddings
        smote.labels = labels

        filtered_embeddings, filtered_labels = smote._filter_by_distance(
            synthetic_embeddings, synthetic_labels
        )

        # Only the first one (dist 0.1) should remain
        self.assertEqual(len(filtered_embeddings), 1)
        np.testing.assert_array_equal(filtered_embeddings, synthetic_embeddings[[0]])
        np.testing.assert_array_equal(filtered_labels, synthetic_labels[[0]])

    def test_filter_by_distance_multiple_classes(self):
        """Test filtering with multiple classes."""
        # Class 0: Origin
        # Class 1: At (10, 10)
        embeddings = np.array([
            [0.0, 0.0],
            [10.0, 10.0]
        ])
        labels = np.array([0, 1])

        # Synthetic:
        # 1. Class 0, near origin (valid)
        # 2. Class 0, far (invalid)
        # 3. Class 1, near (10,10) (valid)
        # 4. Class 1, far (invalid)
        # 5. Class 1, near origin (but wrong class, so invalid relative to class 1)
        synthetic_embeddings = np.array([
            [0.1, 0.1],      # valid for 0
            [5.0, 5.0],      # invalid for 0
            [10.1, 10.1],    # valid for 1
            [15.0, 15.0],    # invalid for 1
            [0.1, 0.1]       # invalid for 1 (far from 10,10)
        ])
        synthetic_labels = np.array([0, 0, 1, 1, 1])

        smote = ConstrainedSMOTE(max_distance_threshold=1.0)
        smote.embeddings = embeddings
        smote.labels = labels

        filtered_embeddings, filtered_labels = smote._filter_by_distance(
            synthetic_embeddings, synthetic_labels
        )

        # Expected: indices 0 and 2
        expected_indices = [0, 2]

        self.assertEqual(len(filtered_embeddings), 2)
        np.testing.assert_array_equal(filtered_embeddings, synthetic_embeddings[expected_indices])
        np.testing.assert_array_equal(filtered_labels, synthetic_labels[expected_indices])

    def test_empty_result(self):
        """Test when no samples satisfy the condition."""
        embeddings = np.zeros((1, 2))
        labels = np.array([0])

        synthetic_embeddings = np.array([[10.0, 10.0]])
        synthetic_labels = np.array([0])

        smote = ConstrainedSMOTE(max_distance_threshold=0.1)
        smote.embeddings = embeddings
        smote.labels = labels

        filtered_embeddings, filtered_labels = smote._filter_by_distance(
            synthetic_embeddings, synthetic_labels
        )

        self.assertEqual(len(filtered_embeddings), 0)
        self.assertEqual(len(filtered_labels), 0)

if __name__ == '__main__':
    unittest.main()
