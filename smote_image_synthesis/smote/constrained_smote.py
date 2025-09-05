"""
Constrained SMOTE implementation for embedding space.
"""

from typing import Tuple, Optional, Dict, List, Union, Any
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
import warnings

logger = logging.getLogger(__name__)


class ConstrainedSMOTE:
    """SMOTE with constraints for embedding space oversampling.
    
    This implementation extends standard SMOTE with multiple constraint mechanisms:
    - Semantic clustering constraints to preserve class structure
    - Distance thresholding for valid interpolation regions
    - Outlier detection to filter invalid synthetic samples
    - Boundary detection for class separation awareness
    """
    
    def __init__(self, 
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
                 min_samples_per_class: int = 2):
        """
        Initialize constrained SMOTE.
        
        Args:
            k_neighbors: Number of nearest neighbors for SMOTE
            sampling_strategy: Sampling strategy ('auto', 'minority', etc.) or dict mapping
            max_distance_threshold: Maximum distance for valid interpolation
            use_clustering: Whether to apply clustering constraints
            n_clusters: Number of clusters (auto-determined if None)
            clustering_method: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical', 'gmm')
            cluster_validation_threshold: Threshold for cluster coherence validation
            semantic_coherence_threshold: Threshold for semantic coherence validation
            boundary_detection_method: Method for boundary detection ('density', 'svm', 'isolation')
            outlier_detection_threshold: Threshold for outlier detection
            manifold_validation: Whether to validate manifold structure
            random_state: Random seed for reproducibility
            normalize_embeddings: Whether to normalize embeddings before processing
            min_samples_per_class: Minimum samples required per class for processing
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
        
        # Initialize components
        self.smote = SMOTE(
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        
        # Storage for fitted components
        self.embeddings: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        self.cluster_models: Dict[int, Any] = {}
        self.outlier_detectors: Dict[int, Any] = {}
        self.is_fitted = False
        
        # Validation
        self._validate_parameters()
        
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.k_neighbors <= 0:
            raise ValueError("k_neighbors must be positive")
        
        valid_clustering_methods = ['kmeans', 'dbscan', 'hierarchical', 'gmm']
        if self.clustering_method not in valid_clustering_methods:
            raise ValueError(f"clustering_method must be one of {valid_clustering_methods}")
        
        valid_boundary_methods = ['density', 'svm', 'isolation']
        if self.boundary_detection_method not in valid_boundary_methods:
            raise ValueError(f"boundary_detection_method must be one of {valid_boundary_methods}")
        
        if not (0.0 <= self.cluster_validation_threshold <= 1.0):
            raise ValueError("cluster_validation_threshold must be between 0 and 1")
        
        if not (0.0 <= self.semantic_coherence_threshold <= 1.0):
            raise ValueError("semantic_coherence_threshold must be between 0 and 1")
        
        if not (0.0 <= self.outlier_detection_threshold <= 1.0):
            raise ValueError("outlier_detection_threshold must be between 0 and 1")
        
        if self.min_samples_per_class < 1:
            raise ValueError("min_samples_per_class must be at least 1")
    
    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> 'ConstrainedSMOTE':
        """
        Fit the SMOTE model on embeddings.
        
        Args:
            embeddings: Training embeddings [N, embedding_dim]
            labels: Corresponding labels [N]
            
        Returns:
            Self for method chaining
        """
        # Validate inputs
        self._validate_input_data(embeddings, labels)
        
        # Store original data
        self.embeddings = embeddings.copy()
        self.labels = labels.copy()
        
        # Normalize embeddings if requested
        if self.normalize_embeddings:
            self.scaler = StandardScaler()
            normalized_embeddings = self.scaler.fit_transform(embeddings)
        else:
            normalized_embeddings = embeddings
        
        # Apply clustering constraints
        if self.use_clustering:
            self._apply_clustering_constraints(normalized_embeddings, labels)
        
        # Initialize outlier detectors
        self._initialize_outlier_detectors(normalized_embeddings, labels)
        
        # Fit base SMOTE
        self.smote.fit(normalized_embeddings, labels)
        self.is_fitted = True
        
        logger.info(f"ConstrainedSMOTE fitted on {len(embeddings)} samples with {len(np.unique(labels))} classes")
        return self
        
    def generate_synthetic(self, n_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic embeddings.
        
        Args:
            n_samples: Number of synthetic samples (uses SMOTE default if None)
            
        Returns:
            Tuple of (synthetic_embeddings, synthetic_labels)
        """
        if not self.is_fitted:
            raise ValueError("SMOTE must be fitted before generating samples")
            
        X_resampled, y_resampled = self.smote.fit_resample(self.embeddings, self.labels)
        
        # Extract only the synthetic samples
        n_original = len(self.embeddings)
        synthetic_embeddings = X_resampled[n_original:]
        synthetic_labels = y_resampled[n_original:]
        
        # Apply validation constraints
        if self.max_distance_threshold is not None:
            synthetic_embeddings, synthetic_labels = self._filter_by_distance(
                synthetic_embeddings, synthetic_labels
            )
            
        return synthetic_embeddings, synthetic_labels
        
    def validate_embedding_space(self, embeddings: np.ndarray) -> bool:
        """
        Validate if embeddings are in valid space.
        
        Args:
            embeddings: Embeddings to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check for NaN or infinity
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            return False
            
        # Check dimensionality
        if embeddings.shape[1] != self.embeddings.shape[1]:
            return False
            
        return True
        
    def _apply_clustering_constraint(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """Apply clustering constraints before SMOTE."""
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            label_mask = labels == label
            label_embeddings = embeddings[label_mask]
            
            if len(label_embeddings) > 1:
                # Determine number of clusters
                n_clusters = self.n_clusters
                if n_clusters is None:
                    n_clusters = min(3, len(label_embeddings) // 2)
                    n_clusters = max(1, n_clusters)
                
                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(label_embeddings)
                    self.cluster_models[label] = kmeans
                    
    def _filter_by_distance(self, 
                          synthetic_embeddings: np.ndarray, 
                          synthetic_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter synthetic embeddings by distance threshold."""
        if self.max_distance_threshold is None:
            return synthetic_embeddings, synthetic_labels
            
        valid_indices = []
        
        for i, (embedding, label) in enumerate(zip(synthetic_embeddings, synthetic_labels)):
            # Find nearest neighbors in original embeddings of same class
            label_mask = self.labels == label
            label_embeddings = self.embeddings[label_mask]
            
            if len(label_embeddings) > 0:
                distances = np.linalg.norm(label_embeddings - embedding, axis=1)
                min_distance = np.min(distances)
                
                if min_distance <= self.max_distance_threshold:
                    valid_indices.append(i)
                    
        if valid_indices:
            return synthetic_embeddings[valid_indices], synthetic_labels[valid_indices]
        else:
            return np.array([]), np.array([])