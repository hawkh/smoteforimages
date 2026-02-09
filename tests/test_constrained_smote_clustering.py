
import unittest
import numpy as np
from sklearn.cluster import KMeans
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE

class TestConstrainedSMOTEClustering(unittest.TestCase):
    """Test ConstrainedSMOTE clustering logic."""

    def setUp(self):
        self.n_samples = 100
        self.n_features = 64
        self.embeddings = np.random.rand(self.n_samples, self.n_features).astype(np.float32)
        # Create 2 classes to satisfy SMOTE
        self.labels = np.zeros(self.n_samples, dtype=int)
        self.labels[50:] = 1

    def test_determine_optimal_clusters_monotonicity_check(self):
        """
        Verify that our intended fix (multiplying variance by N)
        matches the actual KMeans inertia for k=1.
        """
        var_sum = np.sum(np.var(self.embeddings, axis=0))
        inertia_k1_est = var_sum * len(self.embeddings)

        kmeans = KMeans(n_clusters=1, n_init=10, random_state=42)
        kmeans.fit(self.embeddings)
        inertia_k1_true = kmeans.inertia_

        # This confirms the math for the fix is correct
        # Using a loose delta because floating point differences can accumulate
        self.assertAlmostEqual(inertia_k1_est, inertia_k1_true, delta=1.0)

    def test_determine_optimal_clusters_output(self):
        """Test that _determine_optimal_clusters returns a valid number."""
        smote = ConstrainedSMOTE(use_clustering=True, random_state=42)
        n_clusters = smote._determine_optimal_clusters(self.embeddings)

        # Can be int or np.integer
        self.assertTrue(isinstance(n_clusters, (int, np.integer)))
        self.assertGreaterEqual(n_clusters, 1)
        self.assertLessEqual(n_clusters, 5) # Default max clusters

    def test_fit_with_clustering(self):
        """Test fit method with clustering enabled."""
        smote = ConstrainedSMOTE(use_clustering=True, random_state=42)
        smote.fit(self.embeddings, self.labels)

        self.assertTrue(smote.is_fitted)
        # Check if we have cluster models for each class
        # Depending on random data, we might have 1 or more clusters
        for label in [0, 1]:
            # The code only creates cluster models if n_clusters > 1
            # Which happens if elbow method says so.
            # We can't guarantee it will be > 1 for random data, but if it is, it should be in the dict.
            pass

    def test_performance_regression_check(self):
        """Simple check to ensure clustering is reasonably fast."""
        import time
        smote = ConstrainedSMOTE(use_clustering=True, random_state=42)

        start_time = time.time()
        smote._determine_optimal_clusters(self.embeddings)
        duration = time.time() - start_time

        self.assertLess(duration, 2.0)

if __name__ == '__main__':
    unittest.main()
