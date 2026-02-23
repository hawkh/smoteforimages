import unittest
import numpy as np
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE

class TestConstrainedSMOTE(unittest.TestCase):
    def setUp(self):
        self.n_original = 100
        self.n_synthetic = 50
        self.dim = 32
        self.n_classes = 2

        np.random.seed(42)

        self.embeddings = np.random.rand(self.n_original, self.dim).astype(np.float32)
        self.labels = np.random.randint(0, self.n_classes, self.n_original)

        self.synthetic_embeddings = np.random.rand(self.n_synthetic, self.dim).astype(np.float32)
        self.synthetic_labels = np.random.randint(0, self.n_classes, self.n_synthetic)

        self.smote = ConstrainedSMOTE(max_distance_threshold=0.5)
        # Mocking fitted state
        self.smote.embeddings = self.embeddings
        self.smote.labels = self.labels
        self.smote.is_fitted = True

    def test_filter_by_distance_basic(self):
        """Test basic filtering logic."""
        # Create a synthetic sample that is exactly the same as an original sample
        # It should pass any reasonable threshold > 0
        idx = 0
        label = self.labels[idx]
        original_embedding = self.embeddings[idx]

        syn_emb = original_embedding.reshape(1, -1)
        syn_label = np.array([label])

        filtered_emb, filtered_lbl = self.smote._filter_by_distance(syn_emb, syn_label)

        self.assertEqual(len(filtered_emb), 1)
        self.assertTrue(np.allclose(filtered_emb, syn_emb))

    def test_filter_by_distance_threshold(self):
        """Test threshold filtering."""
        # Create a synthetic sample that is far away
        # It should be filtered out given max_distance_threshold=0.5

        # Case 1: Close sample (distance 0.1)
        # Construct vector with norm 0.1
        idx = 0
        label = self.labels[idx]

        # Random vector
        v = np.random.randn(self.dim)
        v = v / np.linalg.norm(v) # unit vector

        close_sample = self.embeddings[idx] + v * 0.1

        # Case 2: Far sample (distance 10)
        far_sample = self.embeddings[idx] + v * 10.0

        syn_emb = np.vstack([close_sample, far_sample])
        syn_lbl = np.array([label, label])

        filtered_emb, filtered_lbl = self.smote._filter_by_distance(syn_emb, syn_lbl)

        self.assertEqual(len(filtered_emb), 1)
        self.assertTrue(np.allclose(filtered_emb[0], close_sample))

    def test_filter_no_threshold(self):
        """Test behavior when max_distance_threshold is None."""
        self.smote.max_distance_threshold = None
        filtered_emb, filtered_lbl = self.smote._filter_by_distance(self.synthetic_embeddings, self.synthetic_labels)

        self.assertEqual(len(filtered_emb), len(self.synthetic_embeddings))
        self.assertTrue(np.array_equal(filtered_emb, self.synthetic_embeddings))

    def test_filter_empty_input(self):
        """Test with empty input."""
        empty_emb = np.array([]).reshape(0, self.dim)
        empty_lbl = np.array([])

        filtered_emb, filtered_lbl = self.smote._filter_by_distance(empty_emb, empty_lbl)

        self.assertEqual(len(filtered_emb), 0)
        self.assertEqual(len(filtered_lbl), 0)

if __name__ == '__main__':
    unittest.main()
