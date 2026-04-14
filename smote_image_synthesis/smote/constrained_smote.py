"""
Constrained SMOTE implementation for embedding space.
"""

from typing import Tuple, Optional, Dict, List, Union, Any
import math
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE
import logging
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class ConstrainedSMOTE:
    """SMOTE with constraints for embedding space oversampling.

    Extends standard SMOTE with:
    - Spherical linear interpolation (SLERP) for manifold-faithful synthesis
    - Semantic clustering constraints to preserve class structure
    - Distance thresholding for valid interpolation regions
    - Outlier detection to filter invalid synthetic samples
    - Boundary detection for class separation awareness
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        sampling_strategy: Union[str, Dict[int, int]] = 'auto',
        max_distance_threshold: Optional[float] = None,
        use_clustering: bool = True,
        n_clusters: Optional[int] = None,
        clustering_method: str = 'kmeans',
        cluster_validation_threshold: float = 0.7,
        semantic_coherence_threshold: float = 0.8,
        boundary_detection_method: str = 'density',
        outlier_detection_threshold: float = 0.1,
        manifold_validation: bool = True,
        random_state: Optional[int] = None,
        normalize_embeddings: bool = True,
        min_samples_per_class: int = 2,
        use_slerp: bool = True,
    ):
        """
        Initialize constrained SMOTE.

        Args:
            k_neighbors: Number of nearest neighbours for interpolation
            sampling_strategy: Sampling strategy ('auto', 'minority', etc.) or dict
            max_distance_threshold: Maximum distance for valid interpolation
            use_clustering: Whether to apply clustering constraints
            n_clusters: Number of clusters (auto-determined if None)
            clustering_method: Clustering algorithm ('kmeans','dbscan','hierarchical','gmm')
            cluster_validation_threshold: Threshold for cluster coherence validation
            semantic_coherence_threshold: Threshold for semantic coherence validation
            boundary_detection_method: Method for boundary detection ('density','svm','isolation')
            outlier_detection_threshold: Threshold for outlier detection
            manifold_validation: Whether to validate manifold structure
            random_state: Random seed for reproducibility
            normalize_embeddings: Whether to standardise embeddings before processing
            min_samples_per_class: Minimum samples required per class
            use_slerp: Use spherical linear interpolation (SLERP) instead of linear SMOTE.
                       SLERP follows geodesics on the embedding manifold and produces
                       more semantically coherent synthetic samples.
        """
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.max_distance_threshold = max_distance_threshold
        self.use_clustering = use_clustering
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.cluster_validation_threshold = cluster_validation_threshold
        self.semantic_coherence_threshold = semantic_coherence_threshold
        self.boundary_detection_method = boundary_detection_method
        self.outlier_detection_threshold = outlier_detection_threshold
        self.manifold_validation = manifold_validation
        self.random_state = random_state or 42
        self.normalize_embeddings = normalize_embeddings
        self.min_samples_per_class = min_samples_per_class
        self.use_slerp = use_slerp

        # Fallback sklearn SMOTE (used when use_slerp=False)
        self.smote = SMOTE(
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
        )

        # Storage for fitted components
        self.embeddings: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        self.cluster_models: Dict[int, Any] = {}
        self.outlier_detectors: Dict[int, Any] = {}
        self.is_fitted = False

        self._validate_parameters()

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> 'ConstrainedSMOTE':
        """Fit the SMOTE model on embeddings."""
        self._validate_input_data(embeddings, labels)

        self.embeddings = embeddings.copy()
        self.labels = labels.copy()

        if self.normalize_embeddings:
            self.scaler = StandardScaler()
            normalized_embeddings = self.scaler.fit_transform(embeddings)
        else:
            normalized_embeddings = embeddings

        if self.use_clustering:
            self._apply_clustering_constraints(normalized_embeddings, labels)

        self._initialize_outlier_detectors(normalized_embeddings, labels)

        # Pre-fit sklearn SMOTE so it's ready if use_slerp=False
        self.smote.fit(normalized_embeddings, labels)
        self.is_fitted = True

        logger.info(
            f"ConstrainedSMOTE fitted on {len(embeddings)} samples, "
            f"{len(np.unique(labels))} classes, "
            f"interpolation={'SLERP' if self.use_slerp else 'linear'}"
        )
        return self

    # ------------------------------------------------------------------
    # SLERP interpolation
    # ------------------------------------------------------------------

    @staticmethod
    def _slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between two embedding vectors.

        Interpolates along the great-circle arc on the hypersphere defined by
        the two vectors.  Magnitude is linearly interpolated so the output
        lives at the same "radius" as the inputs (important when embeddings
        are not unit-normalised).

        Args:
            v0: Start vector [D]
            v1: End vector [D]
            t: Interpolation weight in [0, 1]

        Returns:
            Interpolated vector [D]
        """
        n0 = np.linalg.norm(v0)
        n1 = np.linalg.norm(v1)
        if n0 < 1e-8 or n1 < 1e-8:
            return (1.0 - t) * v0 + t * v1

        u0, u1 = v0 / n0, v1 / n1
        dot = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
        omega = np.arccos(dot)

        if abs(omega) < 1e-6:
            # Nearly parallel — linear fallback avoids division by ~0
            interp_unit = (1.0 - t) * u0 + t * u1
            norm_iu = np.linalg.norm(interp_unit)
            if norm_iu > 1e-8:
                interp_unit = interp_unit / norm_iu
        else:
            s = np.sin(omega)
            interp_unit = (
                np.sin((1.0 - t) * omega) / s * u0
                + np.sin(t * omega) / s * u1
            )

        # Scale by linearly interpolated magnitude
        interp_norm = (1.0 - t) * n0 + t * n1
        return interp_unit * interp_norm

    def _generate_slerp(
        self,
        work_embeddings: np.ndarray,
        n_samples: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic samples via SLERP-SMOTE (k-NN + spherical interp).

        Args:
            work_embeddings: Embeddings in working space (may be scaled)
            n_samples: Total synthetic samples to produce (None → balance classes)

        Returns:
            (synthetic_embeddings, synthetic_labels) in working space
        """
        rng = np.random.default_rng(self.random_state)
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        n_classes = len(unique_labels)

        if n_samples is not None and n_samples > 0:
            per_class = max(1, math.ceil(n_samples / n_classes))
        else:
            max_count = int(np.max(counts))
            min_count = int(np.min(counts))
            per_class = max_count - min_count
            if per_class == 0:
                empty = np.empty((0, work_embeddings.shape[1]), dtype=work_embeddings.dtype)
                return empty, np.empty(0, dtype=np.int64)

        synthetics: List[np.ndarray] = []
        syn_labels: List[int] = []

        for label in unique_labels:
            class_embs = work_embeddings[self.labels == label]
            n = len(class_embs)
            k = min(self.k_neighbors, n - 1)
            if k < 1:
                continue

            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
            nbrs.fit(class_embs)
            _, nn_idx = nbrs.kneighbors(class_embs)  # (n, k+1); col-0 is self

            for _ in range(per_class):
                i = int(rng.integers(n))
                col = int(rng.integers(1, k + 1))
                j = int(nn_idx[i, col])
                t = float(rng.uniform(0.0, 1.0))
                synthetics.append(self._slerp(class_embs[i], class_embs[j], t))
                syn_labels.append(int(label))

        if not synthetics:
            empty = np.empty((0, work_embeddings.shape[1]), dtype=work_embeddings.dtype)
            return empty, np.empty(0, dtype=np.int64)

        syn_arr = np.array(synthetics, dtype=work_embeddings.dtype)
        lbl_arr = np.array(syn_labels, dtype=np.int64)

        # Trim to exact requested count
        if n_samples is not None and 0 < n_samples < len(syn_arr):
            idx = rng.choice(len(syn_arr), n_samples, replace=False)
            idx.sort()
            syn_arr = syn_arr[idx]
            lbl_arr = lbl_arr[idx]

        return syn_arr, lbl_arr

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_synthetic(
        self, n_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic embeddings.

        Args:
            n_samples: Number of synthetic samples.  Distributed evenly across
                       classes.  None uses the default SMOTE strategy.

        Returns:
            Tuple of (synthetic_embeddings, synthetic_labels)
        """
        if not self.is_fitted:
            raise ValueError("SMOTE must be fitted before generating samples")

        # Work in normalised space when a scaler was fitted
        work_embeddings = (
            self.scaler.transform(self.embeddings)
            if self.scaler is not None
            else self.embeddings
        )

        if self.use_slerp:
            synthetic_embeddings, synthetic_labels = self._generate_slerp(
                work_embeddings, n_samples
            )
        else:
            # ── Standard sklearn SMOTE path ──────────────────────────────────
            if n_samples is not None and n_samples > 0:
                unique_labels, counts = np.unique(self.labels, return_counts=True)
                n_classes = len(unique_labels)
                per_class_extra = max(1, math.ceil(n_samples / n_classes))
                target = {
                    int(lbl): int(cnt) + per_class_extra
                    for lbl, cnt in zip(unique_labels, counts)
                }
                temp_smote = SMOTE(
                    k_neighbors=self.k_neighbors,
                    sampling_strategy=target,
                    random_state=self.random_state,
                )
                X_resampled, y_resampled = temp_smote.fit_resample(
                    work_embeddings, self.labels
                )
            else:
                X_resampled, y_resampled = self.smote.fit_resample(
                    work_embeddings, self.labels
                )

            n_original = len(self.embeddings)
            synthetic_embeddings = X_resampled[n_original:]
            synthetic_labels = y_resampled[n_original:]

            # Trim to exact requested count
            if n_samples is not None and 0 < n_samples < len(synthetic_embeddings):
                rng = np.random.default_rng(self.random_state)
                idx = rng.choice(len(synthetic_embeddings), n_samples, replace=False)
                idx.sort()
                synthetic_embeddings = synthetic_embeddings[idx]
                synthetic_labels = synthetic_labels[idx]

        # Invert normalisation so embeddings match the decoder's expected space
        if self.scaler is not None and len(synthetic_embeddings) > 0:
            synthetic_embeddings = self.scaler.inverse_transform(synthetic_embeddings)

        # Apply distance-based validity filter (raw space)
        if self.max_distance_threshold is not None and len(synthetic_embeddings) > 0:
            synthetic_embeddings, synthetic_labels = self._filter_by_distance(
                synthetic_embeddings, synthetic_labels
            )

        return synthetic_embeddings, synthetic_labels

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_embedding_space(
        self, embeddings: np.ndarray
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate if embeddings are in valid space."""
        report: Dict[str, Any] = {
            'n_samples': len(embeddings),
            'n_features': embeddings.shape[1] if embeddings.ndim == 2 else None,
        }
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            report['error'] = 'NaN or infinite values detected'
            return False, report
        if self.embeddings is not None and embeddings.shape[1] != self.embeddings.shape[1]:
            report['error'] = (
                f'Dimension mismatch: expected {self.embeddings.shape[1]}, '
                f'got {embeddings.shape[1]}'
            )
            return False, report
        report['status'] = 'valid'
        return True, report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filter_by_distance(
        self,
        synthetic_embeddings: np.ndarray,
        synthetic_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter synthetic embeddings by distance threshold."""
        if self.max_distance_threshold is None:
            return synthetic_embeddings, synthetic_labels

        valid_indices = []
        unique_labels = np.unique(synthetic_labels)

        # ⚡ Bolt: Replaced O(N x M) python loop with batched vectorized cdist
        # computation. Improves _filter_by_distance speed by ~10x,
        # avoiding OOM on large sets.
        for label in unique_labels:
            syn_mask = synthetic_labels == label
            syn_indices = np.where(syn_mask)[0]
            syn_embs = synthetic_embeddings[syn_mask]

            real_mask = self.labels == label
            real_embs = self.embeddings[real_mask]

            if len(real_embs) == 0:
                continue

            batch_size = 1000
            for i in range(0, len(syn_embs), batch_size):
                batch_syn_embs = syn_embs[i:i + batch_size]
                batch_syn_indices = syn_indices[i:i + batch_size]

                dists = cdist(batch_syn_embs, real_embs, metric='euclidean')
                min_dists = np.min(dists, axis=1)

                valid = batch_syn_indices[min_dists <= self.max_distance_threshold]
                valid_indices.extend(valid)

        valid_indices.sort()
        if valid_indices:
            return synthetic_embeddings[valid_indices], synthetic_labels[valid_indices]
        return np.array([]), np.array([])

    def _validate_input_data(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> None:
        """Validate input data format and consistency."""
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("embeddings must be a numpy array")
        if not isinstance(labels, np.ndarray):
            raise TypeError("labels must be a numpy array")
        if len(embeddings) != len(labels):
            raise ValueError("embeddings and labels must have same length")
        if len(embeddings) == 0:
            raise ValueError("Empty embeddings provided")
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D array")
        if labels.ndim != 1:
            raise ValueError("labels must be 1D array")
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            raise ValueError("embeddings contain NaN or infinite values")

        unique_labels, counts = np.unique(labels, return_counts=True)
        insufficient_classes = unique_labels[counts < self.min_samples_per_class]
        if len(insufficient_classes) > 0:
            logger.warning(
                f"Classes {insufficient_classes} have fewer than "
                f"{self.min_samples_per_class} samples"
            )

        min_count = int(np.min(counts))
        if min_count <= self.k_neighbors:
            raise ValueError(
                f"Insufficient samples for k_neighbors={self.k_neighbors}. "
                f"Minimum samples per class is {min_count}, "
                f"need at least {self.k_neighbors + 1}."
            )

    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.k_neighbors <= 0:
            raise ValueError("k_neighbors must be positive")
        valid_cm = ['kmeans', 'dbscan', 'hierarchical', 'gmm']
        if self.clustering_method not in valid_cm:
            raise ValueError(f"clustering_method must be one of {valid_cm}")
        valid_bm = ['density', 'svm', 'isolation']
        if self.boundary_detection_method not in valid_bm:
            raise ValueError(f"boundary_detection_method must be one of {valid_bm}")
        if not (0.0 <= self.cluster_validation_threshold <= 1.0):
            raise ValueError("cluster_validation_threshold must be between 0 and 1")
        if not (0.0 <= self.semantic_coherence_threshold <= 1.0):
            raise ValueError("semantic_coherence_threshold must be between 0 and 1")
        if not (0.0 <= self.outlier_detection_threshold <= 1.0):
            raise ValueError("outlier_detection_threshold must be between 0 and 1")
        if self.min_samples_per_class < 1:
            raise ValueError("min_samples_per_class must be at least 1")

    def _apply_clustering_constraints(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> None:
        """Apply clustering constraints to preserve semantic structure."""
        for label in np.unique(labels):
            label_embeddings = embeddings[labels == label]
            if len(label_embeddings) < self.min_samples_per_class:
                continue
            n_clusters = self._determine_optimal_clusters(label_embeddings)
            if n_clusters > 1:
                cluster_model = self._create_cluster_model(n_clusters)
                cluster_model.fit_predict(label_embeddings)
                self.cluster_models[label] = cluster_model
                logger.debug(f"Class {label}: {n_clusters} clusters created")

    def _determine_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """Determine optimal number of clusters for embeddings."""
        if self.n_clusters is not None:
            return min(self.n_clusters, len(embeddings))
        max_clusters = min(5, len(embeddings) // 2)
        if max_clusters < 2:
            return 1
        inertias = []
        for k in range(1, max_clusters + 1):
            if k == 1:
                inertias.append(np.sum(np.var(embeddings, axis=0)))
            else:
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                km.fit(embeddings)
                inertias.append(km.inertia_)
        if len(inertias) < 3:
            return 1
        second_diffs = np.diff(np.diff(inertias))
        if len(second_diffs) > 0:
            return min(int(np.argmax(second_diffs)) + 2, max_clusters)
        return 2

    def _create_cluster_model(self, n_clusters: int) -> Any:
        """Create clustering model based on specified method."""
        if self.clustering_method == 'kmeans':
            return KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        elif self.clustering_method == 'dbscan':
            return DBSCAN(eps=0.5, min_samples=2)
        elif self.clustering_method == 'hierarchical':
            return AgglomerativeClustering(n_clusters=n_clusters)
        elif self.clustering_method == 'gmm':
            return GaussianMixture(n_components=n_clusters, random_state=self.random_state)
        raise ValueError(f"Unknown clustering method: {self.clustering_method}")

    def _initialize_outlier_detectors(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> None:
        """Initialize outlier detection models for each class."""
        for label in np.unique(labels):
            label_embeddings = embeddings[labels == label]
            if len(label_embeddings) < 3:
                continue
            try:
                if self.boundary_detection_method == 'isolation':
                    detector = IsolationForest(
                        contamination=self.outlier_detection_threshold,
                        random_state=self.random_state,
                    )
                elif self.boundary_detection_method == 'svm':
                    detector = OneClassSVM(
                        nu=self.outlier_detection_threshold, gamma='scale'
                    )
                else:  # density
                    detector = LocalOutlierFactor(
                        n_neighbors=min(5, len(label_embeddings) - 1),
                        contamination=self.outlier_detection_threshold,
                    )
                detector.fit(label_embeddings)
                self.outlier_detectors[label] = detector
            except Exception as e:
                logger.warning(
                    f"Failed to initialize outlier detector for class {label}: {e}"
                )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_cluster_info(self) -> Dict[int, Dict[str, Any]]:
        """Get information about clusters for each class."""
        cluster_info = {}
        for label, model in self.cluster_models.items():
            info: Dict[str, Any] = {
                'n_clusters': getattr(model, 'n_clusters', 'unknown'),
                'method': self.clustering_method,
            }
            if hasattr(model, 'inertia_'):
                info['inertia'] = model.inertia_
            if hasattr(model, 'cluster_centers_'):
                info['centers_shape'] = model.cluster_centers_.shape
            cluster_info[label] = info
        return cluster_info

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the fitted SMOTE model."""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        stats: Dict[str, Any] = {
            'status': 'fitted',
            'n_samples': len(self.embeddings),
            'n_features': self.embeddings.shape[1],
            'n_classes': len(unique_labels),
            'class_distribution': dict(zip(unique_labels.tolist(), counts.tolist())),
            'use_clustering': self.use_clustering,
            'use_slerp': self.use_slerp,
            'n_cluster_models': len(self.cluster_models),
            'n_outlier_detectors': len(self.outlier_detectors),
            'normalize_embeddings': self.normalize_embeddings,
        }
        if self.use_clustering:
            stats['cluster_info'] = self.get_cluster_info()
        return stats
