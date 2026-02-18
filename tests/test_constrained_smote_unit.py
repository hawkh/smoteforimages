
import unittest
import numpy as np
import sys
from pathlib import Path

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE
from sklearn.metrics import pairwise_distances

class TestConstrainedSMOTEUnit(unittest.TestCase):
    def setUp(self):
        self.n_samples = 50
        self.n_features = 32
        self.n_classes = 3
        self.embeddings = np.random.randn(self.n_samples, self.n_features)
        self.labels = np.random.randint(0, self.n_classes, self.n_samples)

        # Ensure at least one sample per class to avoid issues
        for i in range(self.n_classes):
            self.labels[i] = i

    def test_basic_flow(self):
        smote = ConstrainedSMOTE(
            k_neighbors=2,
            sampling_strategy='auto',
            use_clustering=False,
            normalize_embeddings=False,
            random_state=42
        )
        smote.fit(self.embeddings, self.labels)

        synthetic_embeddings, synthetic_labels = smote.generate_synthetic()

        # Check output types
        self.assertIsInstance(synthetic_embeddings, np.ndarray)
        self.assertIsInstance(synthetic_labels, np.ndarray)

        # Check dimensions
        self.assertEqual(synthetic_embeddings.shape[1], self.n_features)
        self.assertEqual(len(synthetic_embeddings), len(synthetic_labels))

    def test_validate_embedding_space(self):
        smote = ConstrainedSMOTE(k_neighbors=2)
        smote.fit(self.embeddings, self.labels)

        is_valid = smote.validate_embedding_space(self.embeddings)
        self.assertTrue(is_valid)

        # Test invalid
        invalid_embeddings = self.embeddings.copy()
        invalid_embeddings[0, 0] = np.nan
        is_valid = smote.validate_embedding_space(invalid_embeddings)
        self.assertFalse(is_valid)

    def test_filter_by_distance_logic(self):
        """
        Verify that _filter_by_distance correctly filters samples.
        We'll manually construct a case.
        """
        # Create a simple 2D case
        embeddings = np.array([
            [0.0, 0.0], # Class 0
            [10.0, 10.0] # Class 1
        ])
        labels = np.array([0, 1])

        smote = ConstrainedSMOTE(
            k_neighbors=1,
            max_distance_threshold=1.0,
            min_samples_per_class=1
        )
        smote.fit(embeddings, labels)

        # Synthetic samples
        # 1. Close to Class 0 (dist=0.5) -> Should be kept
        # 2. Far from Class 0 (dist=2.0) -> Should be dropped
        # 3. Close to Class 1 (dist=0.5) -> Should be kept
        # 4. Far from Class 1 (dist=2.0) -> Should be dropped

        synthetic_embeddings = np.array([
            [0.5, 0.0],   # Close to 0
            [2.0, 0.0],   # Far from 0
            [10.5, 10.0], # Close to 1
            [12.0, 12.0]  # Far from 1
        ])
        synthetic_labels = np.array([0, 0, 1, 1])

        filtered_emb, filtered_lbl = smote._filter_by_distance(synthetic_embeddings, synthetic_labels)

        self.assertEqual(len(filtered_emb), 2)

        # Check contents
        # We expect [0.5, 0.0] and [10.5, 10.0]
        expected_embeddings = np.array([
            [0.5, 0.0],
            [10.5, 10.0]
        ])

        # Sort to compare safely
        # But filter_by_distance might preserve order or not.
        # In my implementation, I process by label group.
        # Class 0: [0.5, 0.0] (kept)
        # Class 1: [10.5, 10.0] (kept)

        # We can check if expected are in filtered
        for exp in expected_embeddings:
            found = False
            for res in filtered_emb:
                if np.allclose(exp, res):
                    found = True
                    break
            self.assertTrue(found, f"Expected {exp} not found in result")

if __name__ == "__main__":
    unittest.main()
