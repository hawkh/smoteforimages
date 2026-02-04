
import unittest
import numpy as np
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE

class TestConstrainedSMOTEOptimization(unittest.TestCase):
    def test_filter_by_distance_logic(self):
        """Test that _filter_by_distance correctly filters points."""

        # Setup
        dim = 2
        # Original points at (0,0) and (10,10) for class 0
        embeddings = np.array([[0.0, 0.0], [10.0, 10.0]])
        labels = np.array([0, 0])

        smote = ConstrainedSMOTE(max_distance_threshold=1.0)
        smote.embeddings = embeddings
        smote.labels = labels
        smote.is_fitted = True

        # Synthetic points:
        # 1. (0.5, 0.5) -> dist to (0,0) is sqrt(0.5^2+0.5^2) = 0.707 < 1.0 (KEEP)
        # 2. (2.0, 2.0) -> dist to (0,0) is sqrt(8) ~ 2.8 > 1.0. Dist to (10,10) large. (DROP)
        # 3. (10.5, 10.5) -> dist to (10,10) is 0.707 < 1.0 (KEEP)

        syn_embeddings = np.array([
            [0.5, 0.5],
            [2.0, 2.0],
            [10.5, 10.5]
        ])
        syn_labels = np.array([0, 0, 0])

        filtered_emb, filtered_lbl = smote._filter_by_distance(syn_embeddings, syn_labels)

        self.assertEqual(len(filtered_emb), 2)
        # We expect points 1 and 3 to be kept.
        # Check coordinates close to expected
        self.assertTrue(np.allclose(filtered_emb[0], [0.5, 0.5]) or np.allclose(filtered_emb[0], [10.5, 10.5]))
        self.assertTrue(np.allclose(filtered_emb[1], [0.5, 0.5]) or np.allclose(filtered_emb[1], [10.5, 10.5]))

    def test_filter_by_distance_multi_class(self):
        """Test with multiple classes."""
        # Class 0 at (0,0), Class 1 at (10,10)
        embeddings = np.array([[0.0, 0.0], [10.0, 10.0]])
        labels = np.array([0, 1])

        smote = ConstrainedSMOTE(max_distance_threshold=1.0)
        smote.embeddings = embeddings
        smote.labels = labels
        smote.is_fitted = True

        # Synthetic:
        # 1. Class 0 at (0.5, 0.5) -> Valid for Class 0
        # 2. Class 0 at (10.5, 10.5) -> Invalid for Class 0 (nearest Class 0 is (0,0) -> dist large)
        # 3. Class 1 at (0.5, 0.5) -> Invalid for Class 1 (nearest Class 1 is (10,10) -> dist large)
        # 4. Class 1 at (10.5, 10.5) -> Valid for Class 1

        syn_embeddings = np.array([
            [0.5, 0.5],
            [10.5, 10.5],
            [0.5, 0.5],
            [10.5, 10.5]
        ])
        syn_labels = np.array([0, 0, 1, 1])

        filtered_emb, filtered_lbl = smote._filter_by_distance(syn_embeddings, syn_labels)

        self.assertEqual(len(filtered_emb), 2)

        # Check we have one class 0 and one class 1
        self.assertEqual(np.sum(filtered_lbl == 0), 1)
        self.assertEqual(np.sum(filtered_lbl == 1), 1)

        # Check actual values
        # The class 0 point should be near (0,0) i.e., (0.5, 0.5)
        self.assertTrue(np.allclose(filtered_emb[filtered_lbl==0][0], [0.5, 0.5]))
        # The class 1 point should be near (10,10) i.e., (10.5, 10.5)
        self.assertTrue(np.allclose(filtered_emb[filtered_lbl==1][0], [10.5, 10.5]))

if __name__ == '__main__':
    unittest.main()
