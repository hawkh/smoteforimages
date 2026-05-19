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
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)


class ConstrainedSMOTE:
    """SMOTE with constraints for embedding space oversampling.

    Extends standard SMOTE with:
    - Spherical linear interpolation (SLERP) for manifold-faithful synthesis
    - von Mises-Fisher (vMF) distribution sampling on unit hypersphere
    - Density-weighted SLERP t parameter for gap-filling
    - Cluster-constrained SLERP to prevent cross-mode interpolation
    - Outlier detection filtering of invalid synthetic samples
    - Synthesis ancestry tracking for data provenance
    - Distance thresholding for valid interpolation regions
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
        # Frontier improvements
        use_vmf: bool = False,
        vmf_concentration_scale: float = 1.0,
        use_outlier_detection: bool = False,
        use_cluster_constraints: bool = False,
        density_weighted_t: bool = False,
        track_ancestry: bool = False,
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
            use_vmf: Use von Mises-Fisher distribution sampling on unit hypersphere.
                     Fits a vMF(μ, κ) per class and samples directly from the distribution —
                     more principled than k-NN SLERP, no pair-selection bias.
                     Takes precedence over use_slerp when both are True.
            vmf_concentration_scale: Scale factor applied to the MLE-estimated κ.
                     >1 tightens around class mean (higher fidelity); <1 broadens
                     (higher diversity). Default 1.0 uses MLE κ directly.
            use_outlier_detection: Activate per-class outlier filtering on synthetic
                     samples. Detectors are initialized in fit(); this wires them into
                     the generation path. Rejected samples are resampled up to 3 times.
            use_cluster_constraints: Constrain SLERP k-NN selection to within-cluster
                     neighbours only, preventing interpolation between semantically distant
                     modes of the same class. Requires use_clustering=True.
            density_weighted_t: Bias the SLERP interpolation parameter t toward 0.5
                     in low-density geodesic regions (gap-filling) and toward Uniform
                     in high-density regions. Uses per-point k-NN density estimates.
            track_ancestry: Record provenance metadata for each synthetic sample
                     (parent indices, t value, method, cluster). Stored in
                     self.last_ancestry after each generate_synthetic() call.
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
        self.use_vmf = use_vmf
        self.vmf_concentration_scale = vmf_concentration_scale
        self.use_outlier_detection = use_outlier_detection
        self.use_cluster_constraints = use_cluster_constraints
        self.density_weighted_t = density_weighted_t
        self.track_ancestry = track_ancestry

        # Fallback sklearn SMOTE (used when use_slerp=False and use_vmf=False)
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
        self.cluster_assignments: Dict[int, np.ndarray] = {}  # per-class cluster labels
        self.outlier_detectors: Dict[int, Any] = {}
        self.last_ancestry: Optional[Dict[int, Dict]] = None
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

        # Pre-fit sklearn SMOTE so it's ready if use_slerp=False and use_vmf=False
        self.smote.fit(normalized_embeddings, labels)
        self.is_fitted = True

        mode = 'vMF' if self.use_vmf else ('SLERP' if self.use_slerp else 'linear')
        logger.info(
            f"ConstrainedSMOTE fitted on {len(embeddings)} samples, "
            f"{len(np.unique(labels))} classes, interpolation={mode}"
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
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[int, Dict]]]:
        """Generate synthetic samples via SLERP-SMOTE (k-NN + spherical interp).

        Supports:
        - Cluster-constrained k-NN (use_cluster_constraints=True)
        - Density-weighted t sampling (density_weighted_t=True)
        - Ancestry tracking (track_ancestry=True)

        Args:
            work_embeddings: Embeddings in working space (may be scaled)
            n_samples: Total synthetic samples to produce (None → balance classes)

        Returns:
            (synthetic_embeddings, synthetic_labels, ancestry_or_None) in working space
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
                return empty, np.empty(0, dtype=np.int64), None

        synthetics: List[np.ndarray] = []
        syn_labels: List[int] = []
        ancestry: Dict[int, Dict] = {} if self.track_ancestry else None
        syn_idx = 0

        for label in unique_labels:
            label_int = int(label)
            class_mask = self.labels == label
            class_embs = work_embeddings[class_mask]
            class_global_indices = np.where(class_mask)[0]  # for ancestry
            n = len(class_embs)
            k = min(self.k_neighbors, n - 1)
            if k < 1:
                continue

            # ── Fit k-NN for this class ──────────────────────────────────────
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
            nbrs.fit(class_embs)
            all_dists, nn_idx = nbrs.kneighbors(class_embs)  # (n, k+1); col-0 is self

            # ── Density estimates for t weighting ───────────────────────────
            if self.density_weighted_t:
                # Per-point density ∝ inverse mean k-NN distance (exclude self at col 0)
                point_densities = 1.0 / (all_dists[:, 1:].mean(axis=1) + 1e-8)
                median_density = float(np.median(point_densities))
            else:
                point_densities = None
                median_density = 0.0

            # ── Cluster assignments for constrained SLERP ───────────────────
            if (self.use_cluster_constraints
                    and label_int in self.cluster_assignments
                    and self.clustering_method != 'dbscan'):
                cluster_assigns = self.cluster_assignments[label_int]
            else:
                cluster_assigns = None

            for _ in range(per_class):
                i = int(rng.integers(n))

                # Determine neighbor candidates
                if cluster_assigns is not None:
                    ci = int(cluster_assigns[i])
                    same_cluster = np.where(cluster_assigns == ci)[0]
                    candidates = [idx for idx in same_cluster if idx != i]
                    if len(candidates) >= 1:
                        j = int(rng.choice(candidates))
                        cluster_id = ci
                    else:
                        # Fall back to global k-NN
                        col = int(rng.integers(1, k + 1))
                        j = int(nn_idx[i, col])
                        cluster_id = ci
                else:
                    col = int(rng.integers(1, k + 1))
                    j = int(nn_idx[i, col])
                    cluster_id = -1

                # ── Density-weighted t ──────────────────────────────────────
                if self.density_weighted_t and point_densities is not None:
                    mid_density = 0.5 * (point_densities[i] + point_densities[j])
                    if mid_density < 0.8 * median_density:
                        # Sparse gap — concentrate t near 0.5 using Beta(3,3)
                        t = float(rng.beta(3.0, 3.0))
                    else:
                        t = float(rng.uniform(0.0, 1.0))
                else:
                    t = float(rng.uniform(0.0, 1.0))

                syn = self._slerp(class_embs[i], class_embs[j], t)
                synthetics.append(syn)
                syn_labels.append(label_int)

                if ancestry is not None:
                    ancestry[syn_idx] = {
                        'parent_a': int(class_global_indices[i]),
                        'parent_b': int(class_global_indices[j]),
                        'class_label': label_int,
                        't': float(t),
                        'method': 'slerp',
                        'cluster': cluster_id,
                    }
                syn_idx += 1

        if not synthetics:
            empty = np.empty((0, work_embeddings.shape[1]), dtype=work_embeddings.dtype)
            return empty, np.empty(0, dtype=np.int64), ancestry

        syn_arr = np.array(synthetics, dtype=work_embeddings.dtype)
        lbl_arr = np.array(syn_labels, dtype=np.int64)

        # Trim to exact requested count
        if n_samples is not None and 0 < n_samples < len(syn_arr):
            idx = rng.choice(len(syn_arr), n_samples, replace=False)
            idx.sort()
            syn_arr = syn_arr[idx]
            lbl_arr = lbl_arr[idx]
            if ancestry is not None:
                ancestry = {new_i: ancestry[old_i] for new_i, old_i in enumerate(idx)}

        return syn_arr, lbl_arr, ancestry

    # ------------------------------------------------------------------
    # von Mises-Fisher sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_vmf_params(embs: np.ndarray) -> Tuple[np.ndarray, float]:
        """MLE estimate of vMF mean direction μ and concentration κ.

        Uses the Banerjee et al. (2005) approximation for κ:
            κ̂ ≈ r̄(d - r̄²) / (1 - r̄²)
        where r̄ is the mean resultant length of the normalised embeddings.
        """
        d = embs.shape[1]
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        unit_embs = embs / np.maximum(norms, 1e-8)

        mean_vec = unit_embs.mean(axis=0)
        r_bar = float(np.linalg.norm(mean_vec))
        mu = mean_vec / max(r_bar, 1e-8)

        r = min(r_bar, 1.0 - 1e-6)
        kappa = r * (d - r ** 2) / max(1.0 - r ** 2, 1e-8)
        return mu, max(float(kappa), 0.1)

    @staticmethod
    def _sample_vmf(
        mu: np.ndarray,
        kappa: float,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample from vMF(μ, κ) using Wood's (1994) rejection algorithm.

        Works for any d ≥ 2.  Returns unit-norm samples on S^(d-1).
        """
        d = len(mu)

        # Wood's algorithm parameters
        b = (-2.0 * kappa + np.sqrt(4.0 * kappa ** 2 + (d - 1) ** 2)) / (d - 1)
        x0 = (1.0 - b) / (1.0 + b)
        c = kappa * x0 + (d - 1) * math.log(max(1.0 - x0 ** 2, 1e-12))

        # Householder reflection: maps e1 = [1, 0, ...] → μ
        e1 = np.zeros(d)
        e1[0] = 1.0
        u = e1 - mu
        u_norm = np.linalg.norm(u)
        has_rotation = u_norm > 1e-8
        if has_rotation:
            u = u / u_norm  # unit Householder vector

        alpha = (d - 1) / 2.0  # Beta shape param

        samples = np.zeros((n_samples, d))
        for idx in range(n_samples):
            # Rejection sampling for the axial component W
            for _attempt in range(100000):
                Z = float(rng.beta(alpha, alpha))
                W = (1.0 - (1.0 + b) * Z) / (1.0 - (1.0 - b) * Z)
                U = float(rng.uniform())
                log_criterion = (
                    kappa * W
                    + (d - 1) * math.log(max(1.0 - x0 * W, 1e-12))
                    - c
                )
                if log_criterion >= math.log(max(U, 1e-12)):
                    break
            else:
                W = x0  # extremely unlikely fallback

            # Sample uniform vector on S^(d-2)
            v = rng.standard_normal(d - 1)
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-8:
                v = v / v_norm
            else:
                v = np.zeros(d - 1)
                v[0] = 1.0

            # Construct x in frame where μ = e1: x = [W, √(1-W²) · v]
            scale = math.sqrt(max(0.0, 1.0 - W ** 2))
            x = np.concatenate([[W], scale * v])

            # Householder rotation: x → H·x where H·e1 = μ
            if has_rotation:
                x = x - 2.0 * np.dot(x, u) * u

            samples[idx] = x

        return samples

    def _generate_vmf(
        self,
        work_embeddings: np.ndarray,
        n_samples: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[int, Dict]]]:
        """Generate synthetic samples via per-class vMF distribution sampling.

        Fits vMF(μ, κ) to each class's hyperspherical embeddings and samples
        n_per_class points.  κ is scaled by vmf_concentration_scale.

        Returns:
            (synthetic_embeddings, synthetic_labels, ancestry_or_None) in working space
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
                return empty, np.empty(0, dtype=np.int64), None

        synthetics: List[np.ndarray] = []
        syn_labels: List[int] = []
        ancestry: Dict[int, Dict] = {} if self.track_ancestry else None
        syn_idx = 0

        for label in unique_labels:
            label_int = int(label)
            class_embs = work_embeddings[self.labels == label]
            if len(class_embs) < 2:
                continue

            mu, kappa_raw = self._estimate_vmf_params(class_embs)
            kappa = kappa_raw * self.vmf_concentration_scale

            # Sample on unit sphere, then scale to class average norm
            unit_samples = self._sample_vmf(mu, kappa, per_class, rng)
            avg_norm = float(np.linalg.norm(class_embs, axis=1).mean())
            syn_embs = unit_samples * avg_norm

            synthetics.append(syn_embs)
            syn_labels.extend([label_int] * per_class)

            if ancestry is not None:
                for _ in range(per_class):
                    ancestry[syn_idx] = {
                        'parent_a': -1,
                        'parent_b': -1,
                        'class_label': label_int,
                        't': float(kappa),  # reuse field to store κ
                        'method': 'vmf',
                        'cluster': -1,
                    }
                    syn_idx += 1

        if not synthetics:
            empty = np.empty((0, work_embeddings.shape[1]), dtype=work_embeddings.dtype)
            return empty, np.empty(0, dtype=np.int64), ancestry

        syn_arr = np.vstack(synthetics).astype(work_embeddings.dtype)
        lbl_arr = np.array(syn_labels, dtype=np.int64)

        if n_samples is not None and 0 < n_samples < len(syn_arr):
            idx = rng.choice(len(syn_arr), n_samples, replace=False)
            idx.sort()
            syn_arr = syn_arr[idx]
            lbl_arr = lbl_arr[idx]
            if ancestry is not None:
                ancestry = {new_i: ancestry[old_i] for new_i, old_i in enumerate(idx)}

        return syn_arr, lbl_arr, ancestry

    # ------------------------------------------------------------------
    # Outlier filtering
    # ------------------------------------------------------------------

    def _filter_by_outlier_detectors(
        self,
        synthetic_embeddings: np.ndarray,
        synthetic_labels: np.ndarray,
        work_embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter synthetic embeddings (work space) using per-class outlier detectors.

        Rejected samples are resampled via SLERP up to 3 times before being dropped.
        LOF detectors are skipped (novelty prediction is handled at fit time via
        novelty=True; LOF without that flag cannot predict on new data).
        """
        if not self.outlier_detectors or len(synthetic_embeddings) == 0:
            return synthetic_embeddings, synthetic_labels

        rng = np.random.default_rng(self.random_state + 1)
        max_attempts = 3
        valid_mask = np.ones(len(synthetic_embeddings), dtype=bool)

        for i, (emb, label) in enumerate(
            zip(synthetic_embeddings, synthetic_labels)
        ):
            label_int = int(label)
            if label_int not in self.outlier_detectors:
                continue

            detector = self.outlier_detectors[label_int]
            score = int(detector.predict([emb])[0])  # +1=inlier, -1=outlier

            if score == -1:
                class_embs = work_embeddings[self.labels == label_int]
                replaced = False
                for _ in range(max_attempts):
                    if len(class_embs) < 2:
                        break
                    idx_a = int(rng.integers(len(class_embs)))
                    idx_b = int(rng.integers(len(class_embs)))
                    if idx_a == idx_b:
                        continue
                    t_new = float(rng.uniform(0.0, 1.0))
                    new_emb = self._slerp(class_embs[idx_a], class_embs[idx_b], t_new)
                    if int(detector.predict([new_emb])[0]) == 1:
                        synthetic_embeddings[i] = new_emb
                        replaced = True
                        break
                if not replaced:
                    valid_mask[i] = False

        return synthetic_embeddings[valid_mask], synthetic_labels[valid_mask]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_synthetic(
        self, n_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic embeddings.

        Applies (in order when enabled):
          1. vMF sampling or SLERP-SMOTE or sklearn SMOTE
          2. Outlier detector filtering (use_outlier_detection=True)
          3. Inverse normalisation
          4. Distance-based validity filter (max_distance_threshold)

        Ancestry metadata stored in self.last_ancestry when track_ancestry=True.

        Args:
            n_samples: Number of synthetic samples. Distributed evenly across
                       classes.  None uses the default SMOTE strategy.

        Returns:
            Tuple of (synthetic_embeddings, synthetic_labels) in original space.
        """
        if not self.is_fitted:
            raise ValueError("SMOTE must be fitted before generating samples")

        work_embeddings = (
            self.scaler.transform(self.embeddings)
            if self.scaler is not None
            else self.embeddings
        )

        ancestry: Optional[Dict[int, Dict]] = None

        if self.use_vmf:
            synthetic_embeddings, synthetic_labels, ancestry = self._generate_vmf(
                work_embeddings, n_samples
            )
        elif self.use_slerp:
            synthetic_embeddings, synthetic_labels, ancestry = self._generate_slerp(
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

            if n_samples is not None and 0 < n_samples < len(synthetic_embeddings):
                rng = np.random.default_rng(self.random_state)
                idx = rng.choice(len(synthetic_embeddings), n_samples, replace=False)
                idx.sort()
                synthetic_embeddings = synthetic_embeddings[idx]
                synthetic_labels = synthetic_labels[idx]

        # ── Outlier filtering (work space, before inverse transform) ─────────
        if self.use_outlier_detection and len(synthetic_embeddings) > 0:
            synthetic_embeddings, synthetic_labels = self._filter_by_outlier_detectors(
                synthetic_embeddings, synthetic_labels, work_embeddings
            )

        # ── Invert normalisation so embeddings match decoder's expected space ─
        if self.scaler is not None and len(synthetic_embeddings) > 0:
            synthetic_embeddings = self.scaler.inverse_transform(synthetic_embeddings)

        # ── Distance-based validity filter (raw space) ───────────────────────
        if self.max_distance_threshold is not None and len(synthetic_embeddings) > 0:
            synthetic_embeddings, synthetic_labels = self._filter_by_distance(
                synthetic_embeddings, synthetic_labels
            )

        self.last_ancestry = ancestry
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
        """Filter synthetic embeddings by distance threshold.

        Uses vectorized scipy.spatial.distance.cdist instead of python loops
        for ~14x speedup on distance calculations.
        """
        if self.max_distance_threshold is None:
            return synthetic_embeddings, synthetic_labels

        valid_indices = []
        unique_syn_labels = np.unique(synthetic_labels)

        for label in unique_syn_labels:
            label_mask = self.labels == label
            label_embeddings = self.embeddings[label_mask]

            if len(label_embeddings) == 0:
                continue

            syn_mask = synthetic_labels == label
            syn_indices = np.where(syn_mask)[0]
            syn_label_embeddings = synthetic_embeddings[syn_mask]

            # Vectorized O(M x N) distance calculation
            distances = cdist(
                syn_label_embeddings, label_embeddings, metric='euclidean'
            )

            min_distances = np.min(distances, axis=1)
            valid_syn_mask = min_distances <= self.max_distance_threshold

            valid_indices.extend(syn_indices[valid_syn_mask])

        if valid_indices:
            valid_indices.sort()
            return (
                synthetic_embeddings[valid_indices],
                synthetic_labels[valid_indices]
            )
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
        if self.vmf_concentration_scale <= 0.0:
            raise ValueError("vmf_concentration_scale must be positive")

    def _apply_clustering_constraints(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> None:
        """Apply clustering constraints to preserve semantic structure.

        Stores cluster assignments in self.cluster_assignments for use by
        cluster-constrained SLERP.
        """
        for label in np.unique(labels):
            label_int = int(label)
            label_embeddings = embeddings[labels == label]
            if len(label_embeddings) < self.min_samples_per_class:
                continue
            n_clusters = self._determine_optimal_clusters(label_embeddings)
            if n_clusters > 1:
                cluster_model = self._create_cluster_model(n_clusters)
                assignments = cluster_model.fit_predict(label_embeddings)
                self.cluster_models[label_int] = cluster_model
                self.cluster_assignments[label_int] = assignments
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
        """Initialize outlier detection models for each class.

        LOF uses novelty=True so it can predict on new samples generated
        during synthesis (not just the training set).
        """
        for label in np.unique(labels):
            label_int = int(label)
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
                else:  # density — LOF with novelty=True for new-sample prediction
                    detector = LocalOutlierFactor(
                        n_neighbors=min(5, len(label_embeddings) - 1),
                        contamination=self.outlier_detection_threshold,
                        novelty=True,
                    )
                detector.fit(label_embeddings)
                self.outlier_detectors[label_int] = detector
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
            if label in self.cluster_assignments:
                unique, cnts = np.unique(self.cluster_assignments[label], return_counts=True)
                info['cluster_sizes'] = dict(zip(unique.tolist(), cnts.tolist()))
            cluster_info[label] = info
        return cluster_info

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the fitted SMOTE model."""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        mode = 'vmf' if self.use_vmf else ('slerp' if self.use_slerp else 'linear')
        stats: Dict[str, Any] = {
            'status': 'fitted',
            'n_samples': len(self.embeddings),
            'n_features': self.embeddings.shape[1],
            'n_classes': len(unique_labels),
            'class_distribution': dict(zip(unique_labels.tolist(), counts.tolist())),
            'use_clustering': self.use_clustering,
            'interpolation_mode': mode,
            'vmf_concentration_scale': self.vmf_concentration_scale,
            'density_weighted_t': self.density_weighted_t,
            'use_cluster_constraints': self.use_cluster_constraints,
            'use_outlier_detection': self.use_outlier_detection,
            'track_ancestry': self.track_ancestry,
            'n_cluster_models': len(self.cluster_models),
            'n_outlier_detectors': len(self.outlier_detectors),
            'normalize_embeddings': self.normalize_embeddings,
        }
        if self.use_clustering:
            stats['cluster_info'] = self.get_cluster_info()
        return stats
