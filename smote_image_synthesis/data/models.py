"""
Data models for SMOTE image synthesis pipeline.

This module contains the core data structures used throughout the pipeline,
including embedding data, synthetic samples, and configuration models.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import json
import pickle
from pathlib import Path


@dataclass
class EmbeddingData:
    """
    Data model for storing image embeddings with metadata and validation.
    
    Attributes:
        embedding: The embedding vector as numpy array
        label: Class label for the embedding
        source_image_id: Unique identifier for the source image
        metadata: Additional metadata dictionary
        timestamp: When the embedding was created
    """
    embedding: np.ndarray
    label: int
    source_image_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the embedding data for correctness.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check embedding validity
        if self.embedding is None:
            errors.append("Embedding cannot be None")
        elif not isinstance(self.embedding, np.ndarray):
            errors.append("Embedding must be a numpy array")
        elif self.embedding.size == 0:
            errors.append("Embedding cannot be empty")
        elif np.any(np.isnan(self.embedding)):
            errors.append("Embedding contains NaN values")
        elif np.any(np.isinf(self.embedding)):
            errors.append("Embedding contains infinite values")
        
        # Check label validity
        if not isinstance(self.label, int):
            errors.append("Label must be an integer")
        elif self.label < 0:
            errors.append("Label must be non-negative")
        
        # Check source image ID
        if not isinstance(self.source_image_id, str):
            errors.append("Source image ID must be a string")
        elif not self.source_image_id.strip():
            errors.append("Source image ID cannot be empty")
        
        # Check metadata
        if not isinstance(self.metadata, dict):
            errors.append("Metadata must be a dictionary")
        
        # Check timestamp
        if not isinstance(self.timestamp, datetime):
            errors.append("Timestamp must be a datetime object")
        
        return len(errors) == 0, errors
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embedding."""
        return self.embedding.shape[0] if self.embedding is not None else 0
    
    def normalize_embedding(self) -> 'EmbeddingData':
        """
        Create a new EmbeddingData with L2-normalized embedding.
        
        Returns:
            New EmbeddingData instance with normalized embedding
        """
        if self.embedding is None or self.embedding.size == 0:
            return self
        
        norm = np.linalg.norm(self.embedding)
        if norm == 0:
            return self
        
        normalized_embedding = self.embedding / norm
        
        return EmbeddingData(
            embedding=normalized_embedding,
            label=self.label,
            source_image_id=self.source_image_id,
            metadata=self.metadata.copy(),
            timestamp=self.timestamp
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of the embedding data
        """
        return {
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'label': self.label,
            'source_image_id': self.source_image_id,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingData':
        """
        Create EmbeddingData from dictionary.
        
        Args:
            data: Dictionary containing embedding data
            
        Returns:
            EmbeddingData instance
        """
        embedding = np.array(data['embedding']) if data['embedding'] is not None else None
        timestamp = datetime.fromisoformat(data['timestamp'])
        
        return cls(
            embedding=embedding,
            label=data['label'],
            source_image_id=data['source_image_id'],
            metadata=data.get('metadata', {}),
            timestamp=timestamp
        )
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save embedding data to file using pickle for numpy array preservation.
        
        Args:
            filepath: Path to save the embedding data
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'EmbeddingData':
        """
        Load embedding data from file.
        
        Args:
            filepath: Path to load the embedding data from
            
        Returns:
            EmbeddingData instance
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def save_metadata_json(self, filepath: str) -> None:
        """
        Save only metadata to JSON file (excluding embedding array).
        
        Args:
            filepath: Path to save the metadata JSON
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        metadata_dict = {
            'label': self.label,
            'source_image_id': self.source_image_id,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'embedding_shape': self.embedding.shape if self.embedding is not None else None,
            'embedding_dtype': str(self.embedding.dtype) if self.embedding is not None else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata_dict, f, indent=2)


@dataclass
class SyntheticSample:
    """
    Data model for synthetic samples generated by the pipeline.
    
    Attributes:
        synthetic_embedding: The generated synthetic embedding
        generated_image: The decoded synthetic image (optional)
        parent_embeddings: IDs of source embeddings used in interpolation
        interpolation_weights: Weights used for interpolation
        quality_score: Quality assessment score
        decoder_type: Type of decoder used for generation
        generation_timestamp: When the sample was generated
        generation_metadata: Additional generation metadata
    """
    synthetic_embedding: np.ndarray
    parent_embeddings: List[str]
    interpolation_weights: np.ndarray
    quality_score: float
    decoder_type: str
    generation_timestamp: datetime = field(default_factory=datetime.now)
    generated_image: Optional[np.ndarray] = None
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the synthetic sample data.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check synthetic embedding
        if self.synthetic_embedding is None:
            errors.append("Synthetic embedding cannot be None")
        elif not isinstance(self.synthetic_embedding, np.ndarray):
            errors.append("Synthetic embedding must be a numpy array")
        elif self.synthetic_embedding.size == 0:
            errors.append("Synthetic embedding cannot be empty")
        elif np.any(np.isnan(self.synthetic_embedding)):
            errors.append("Synthetic embedding contains NaN values")
        elif np.any(np.isinf(self.synthetic_embedding)):
            errors.append("Synthetic embedding contains infinite values")
        
        # Check parent embeddings
        if not isinstance(self.parent_embeddings, list):
            errors.append("Parent embeddings must be a list")
        elif len(self.parent_embeddings) == 0:
            errors.append("Must have at least one parent embedding")
        elif not all(isinstance(pid, str) for pid in self.parent_embeddings):
            errors.append("All parent embedding IDs must be strings")
        
        # Check interpolation weights
        if self.interpolation_weights is None:
            errors.append("Interpolation weights cannot be None")
        elif not isinstance(self.interpolation_weights, np.ndarray):
            errors.append("Interpolation weights must be a numpy array")
        elif len(self.interpolation_weights) != len(self.parent_embeddings):
            errors.append("Number of weights must match number of parent embeddings")
        elif not np.isclose(np.sum(self.interpolation_weights), 1.0, atol=1e-6):
            errors.append("Interpolation weights must sum to 1.0")
        elif np.any(self.interpolation_weights < 0):
            errors.append("Interpolation weights must be non-negative")
        
        # Check quality score
        if not isinstance(self.quality_score, (int, float)):
            errors.append("Quality score must be a number")
        elif self.quality_score < 0:
            errors.append("Quality score must be non-negative")
        
        # Check decoder type
        if not isinstance(self.decoder_type, str):
            errors.append("Decoder type must be a string")
        elif not self.decoder_type.strip():
            errors.append("Decoder type cannot be empty")
        
        # Check generated image if present
        if self.generated_image is not None:
            if not isinstance(self.generated_image, np.ndarray):
                errors.append("Generated image must be a numpy array")
            elif self.generated_image.ndim not in [2, 3]:
                errors.append("Generated image must be 2D or 3D array")
        
        # Check metadata
        if not isinstance(self.generation_metadata, dict):
            errors.append("Generation metadata must be a dictionary")
        
        return len(errors) == 0, errors
    
    def get_interpolation_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the interpolation used to create this sample.
        
        Returns:
            Dictionary with interpolation details
        """
        return {
            'num_parents': len(self.parent_embeddings),
            'parent_ids': self.parent_embeddings.copy(),
            'weights': self.interpolation_weights.tolist(),
            'max_weight': float(np.max(self.interpolation_weights)),
            'min_weight': float(np.min(self.interpolation_weights)),
            'weight_entropy': float(-np.sum(self.interpolation_weights * np.log(self.interpolation_weights + 1e-10)))
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of the synthetic sample
        """
        return {
            'synthetic_embedding': self.synthetic_embedding.tolist(),
            'parent_embeddings': self.parent_embeddings,
            'interpolation_weights': self.interpolation_weights.tolist(),
            'quality_score': self.quality_score,
            'decoder_type': self.decoder_type,
            'generation_timestamp': self.generation_timestamp.isoformat(),
            'generated_image': self.generated_image.tolist() if self.generated_image is not None else None,
            'generation_metadata': self.generation_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyntheticSample':
        """
        Create SyntheticSample from dictionary.
        
        Args:
            data: Dictionary containing synthetic sample data
            
        Returns:
            SyntheticSample instance
        """
        synthetic_embedding = np.array(data['synthetic_embedding'])
        interpolation_weights = np.array(data['interpolation_weights'])
        generation_timestamp = datetime.fromisoformat(data['generation_timestamp'])
        generated_image = np.array(data['generated_image']) if data['generated_image'] is not None else None
        
        return cls(
            synthetic_embedding=synthetic_embedding,
            parent_embeddings=data['parent_embeddings'],
            interpolation_weights=interpolation_weights,
            quality_score=data['quality_score'],
            decoder_type=data['decoder_type'],
            generation_timestamp=generation_timestamp,
            generated_image=generated_image,
            generation_metadata=data.get('generation_metadata', {})
        )
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save synthetic sample to file using pickle.
        
        Args:
            filepath: Path to save the synthetic sample
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'SyntheticSample':
        """
        Load synthetic sample from file.
        
        Args:
            filepath: Path to load the synthetic sample from
            
        Returns:
            SyntheticSample instance
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)


@dataclass
class EncoderConfig:
    """Configuration for image encoder."""
    architecture: str = "resnet50"
    embedding_dim: int = 512
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout_rate: float = 0.1
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate encoder configuration."""
        errors = []
        
        valid_architectures = ["resnet18", "resnet50", "resnet101", "efficientnet", "vit"]
        if self.architecture not in valid_architectures:
            errors.append(f"Architecture must be one of {valid_architectures}")
        
        if self.embedding_dim <= 0:
            errors.append("Embedding dimension must be positive")
        
        if not isinstance(self.pretrained, bool):
            errors.append("Pretrained must be a boolean")
        
        if not isinstance(self.freeze_backbone, bool):
            errors.append("Freeze backbone must be a boolean")
        
        if not (0.0 <= self.dropout_rate <= 1.0):
            errors.append("Dropout rate must be between 0.0 and 1.0")
        
        return len(errors) == 0, errors


@dataclass
class SMOTEConfig:
    """Configuration for SMOTE algorithm."""
    k_neighbors: int = 5
    sampling_strategy: str = "auto"
    random_state: Optional[int] = None
    use_clustering: bool = False
    cluster_method: str = "kmeans"
    n_clusters: Optional[int] = None
    distance_threshold: float = 0.5
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate SMOTE configuration."""
        errors = []
        
        if self.k_neighbors <= 0:
            errors.append("k_neighbors must be positive")
        
        valid_strategies = ["auto", "minority", "majority", "not minority", "not majority", "all"]
        if self.sampling_strategy not in valid_strategies:
            errors.append(f"Sampling strategy must be one of {valid_strategies}")
        
        if self.random_state is not None and self.random_state < 0:
            errors.append("Random state must be non-negative")
        
        if not isinstance(self.use_clustering, bool):
            errors.append("Use clustering must be a boolean")
        
        valid_cluster_methods = ["kmeans", "hierarchical", "dbscan"]
        if self.cluster_method not in valid_cluster_methods:
            errors.append(f"Cluster method must be one of {valid_cluster_methods}")
        
        if self.n_clusters is not None and self.n_clusters <= 0:
            errors.append("Number of clusters must be positive")
        
        if not (0.0 <= self.distance_threshold <= 1.0):
            errors.append("Distance threshold must be between 0.0 and 1.0")
        
        return len(errors) == 0, errors


@dataclass
class DecoderConfig:
    """Configuration for image decoder."""
    decoder_type: str = "autoencoder"
    image_shape: Tuple[int, int, int] = (64, 64, 3)
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10
    use_perceptual_loss: bool = True
    perceptual_loss_weight: float = 1.0
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate decoder configuration."""
        errors = []
        
        valid_decoders = ["autoencoder", "vae", "gan", "diffusion"]
        if self.decoder_type not in valid_decoders:
            errors.append(f"Decoder type must be one of {valid_decoders}")
        
        if len(self.image_shape) != 3:
            errors.append("Image shape must be 3D (height, width, channels)")
        
        if any(dim <= 0 for dim in self.image_shape):
            errors.append("All image dimensions must be positive")
        
        if self.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        if self.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        if self.num_epochs <= 0:
            errors.append("Number of epochs must be positive")
        
        if self.early_stopping_patience <= 0:
            errors.append("Early stopping patience must be positive")
        
        if not isinstance(self.use_perceptual_loss, bool):
            errors.append("Use perceptual loss must be a boolean")
        
        if self.perceptual_loss_weight < 0:
            errors.append("Perceptual loss weight must be non-negative")
        
        return len(errors) == 0, errors


@dataclass
class QualityConfig:
    """Configuration for quality assessment."""
    metrics: List[str] = field(default_factory=lambda: ["fid", "lpips", "ssim"])
    fid_batch_size: int = 50
    compute_diversity: bool = True
    diversity_sample_size: int = 1000
    quality_threshold: float = 0.7
    generate_report: bool = True
    save_comparisons: bool = True
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate quality configuration."""
        errors = []
        
        valid_metrics = ["fid", "lpips", "ssim", "psnr", "ms_ssim"]
        for metric in self.metrics:
            if metric not in valid_metrics:
                errors.append(f"Invalid metric '{metric}'. Must be one of {valid_metrics}")
        
        if len(self.metrics) == 0:
            errors.append("At least one quality metric must be specified")
        
        if self.fid_batch_size <= 0:
            errors.append("FID batch size must be positive")
        
        if not isinstance(self.compute_diversity, bool):
            errors.append("Compute diversity must be a boolean")
        
        if self.diversity_sample_size <= 0:
            errors.append("Diversity sample size must be positive")
        
        if not (0.0 <= self.quality_threshold <= 1.0):
            errors.append("Quality threshold must be between 0.0 and 1.0")
        
        if not isinstance(self.generate_report, bool):
            errors.append("Generate report must be a boolean")
        
        if not isinstance(self.save_comparisons, bool):
            errors.append("Save comparisons must be a boolean")
        
        return len(errors) == 0, errors


@dataclass
class PipelineConfig:
    """
    Main configuration class for the SMOTE image synthesis pipeline.
    
    Attributes:
        encoder_config: Configuration for the image encoder
        smote_config: Configuration for the SMOTE algorithm
        decoder_config: Configuration for the image decoder
        quality_config: Configuration for quality assessment
        config_name: Name identifier for this configuration
        creation_timestamp: When this configuration was created
        output_dir: Directory for saving outputs
        seed: Random seed for reproducibility
    """
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    smote_config: SMOTEConfig = field(default_factory=SMOTEConfig)
    decoder_config: DecoderConfig = field(default_factory=DecoderConfig)
    quality_config: QualityConfig = field(default_factory=QualityConfig)
    config_name: str = "default_config"
    creation_timestamp: datetime = field(default_factory=datetime.now)
    output_dir: str = "./outputs"
    seed: Optional[int] = None
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the entire pipeline configuration for consistency.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        all_errors = []
        
        # Validate individual components
        encoder_valid, encoder_errors = self.encoder_config.validate()
        if not encoder_valid:
            all_errors.extend([f"Encoder: {error}" for error in encoder_errors])
        
        smote_valid, smote_errors = self.smote_config.validate()
        if not smote_valid:
            all_errors.extend([f"SMOTE: {error}" for error in smote_errors])
        
        decoder_valid, decoder_errors = self.decoder_config.validate()
        if not decoder_valid:
            all_errors.extend([f"Decoder: {error}" for error in decoder_errors])
        
        quality_valid, quality_errors = self.quality_config.validate()
        if not quality_valid:
            all_errors.extend([f"Quality: {error}" for error in quality_errors])
        
        # Cross-component validation
        if self.encoder_config.embedding_dim != self.decoder_config.image_shape[0] * self.decoder_config.image_shape[1]:
            # This is just a warning - embedding dim doesn't need to match image size exactly
            pass
        
        # Validate config name
        if not isinstance(self.config_name, str) or not self.config_name.strip():
            all_errors.append("Config name must be a non-empty string")
        
        # Validate output directory
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            all_errors.append("Output directory must be a non-empty string")
        
        # Validate seed
        if self.seed is not None and self.seed < 0:
            all_errors.append("Seed must be non-negative")
        
        return len(all_errors) == 0, all_errors
    
    def get_consistency_warnings(self) -> List[str]:
        """
        Get warnings about potential configuration inconsistencies.
        
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check if decoder batch size is compatible with quality assessment
        if self.decoder_config.batch_size > self.quality_config.fid_batch_size:
            warnings.append("Decoder batch size is larger than FID batch size - may cause memory issues")
        
        # Check if SMOTE k_neighbors is reasonable for typical dataset sizes
        if self.smote_config.k_neighbors > 20:
            warnings.append("Large k_neighbors value may lead to over-smoothing in SMOTE")
        
        # Check if embedding dimension is reasonable
        if self.encoder_config.embedding_dim > 2048:
            warnings.append("Very high embedding dimension may cause memory and performance issues")
        
        # Check decoder-specific warnings
        if self.decoder_config.decoder_type == "gan" and self.decoder_config.learning_rate > 0.01:
            warnings.append("High learning rate for GAN decoder may cause training instability")
        
        if self.decoder_config.decoder_type == "diffusion" and self.decoder_config.num_epochs < 200:
            warnings.append("Diffusion models typically require more training epochs")
        
        return warnings
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            'encoder_config': {
                'architecture': self.encoder_config.architecture,
                'embedding_dim': self.encoder_config.embedding_dim,
                'pretrained': self.encoder_config.pretrained,
                'freeze_backbone': self.encoder_config.freeze_backbone,
                'dropout_rate': self.encoder_config.dropout_rate
            },
            'smote_config': {
                'k_neighbors': self.smote_config.k_neighbors,
                'sampling_strategy': self.smote_config.sampling_strategy,
                'random_state': self.smote_config.random_state,
                'use_clustering': self.smote_config.use_clustering,
                'cluster_method': self.smote_config.cluster_method,
                'n_clusters': self.smote_config.n_clusters,
                'distance_threshold': self.smote_config.distance_threshold
            },
            'decoder_config': {
                'decoder_type': self.decoder_config.decoder_type,
                'image_shape': self.decoder_config.image_shape,
                'learning_rate': self.decoder_config.learning_rate,
                'batch_size': self.decoder_config.batch_size,
                'num_epochs': self.decoder_config.num_epochs,
                'early_stopping_patience': self.decoder_config.early_stopping_patience,
                'use_perceptual_loss': self.decoder_config.use_perceptual_loss,
                'perceptual_loss_weight': self.decoder_config.perceptual_loss_weight
            },
            'quality_config': {
                'metrics': self.quality_config.metrics,
                'fid_batch_size': self.quality_config.fid_batch_size,
                'compute_diversity': self.quality_config.compute_diversity,
                'diversity_sample_size': self.quality_config.diversity_sample_size,
                'quality_threshold': self.quality_config.quality_threshold,
                'generate_report': self.quality_config.generate_report,
                'save_comparisons': self.quality_config.save_comparisons
            },
            'config_name': self.config_name,
            'creation_timestamp': self.creation_timestamp.isoformat(),
            'output_dir': self.output_dir,
            'seed': self.seed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """
        Create PipelineConfig from dictionary.
        
        Args:
            data: Dictionary containing configuration data
            
        Returns:
            PipelineConfig instance
        """
        encoder_config = EncoderConfig(**data['encoder_config'])
        smote_config = SMOTEConfig(**data['smote_config'])
        decoder_config = DecoderConfig(**data['decoder_config'])
        quality_config = QualityConfig(**data['quality_config'])
        
        creation_timestamp = datetime.fromisoformat(data['creation_timestamp'])
        
        return cls(
            encoder_config=encoder_config,
            smote_config=smote_config,
            decoder_config=decoder_config,
            quality_config=quality_config,
            config_name=data['config_name'],
            creation_timestamp=creation_timestamp,
            output_dir=data['output_dir'],
            seed=data['seed']
        )
    
    def save_config(self, filepath: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save the configuration file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'PipelineConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to load the configuration from
            
        Returns:
            PipelineConfig instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def create_variant(self, name: str, **kwargs) -> 'PipelineConfig':
        """
        Create a variant of this configuration with modified parameters.
        
        Args:
            name: Name for the new configuration variant
            **kwargs: Parameters to modify (nested dict structure supported)
            
        Returns:
            New PipelineConfig instance with modifications
        """
        config_dict = self.to_dict()
        config_dict['config_name'] = name
        config_dict['creation_timestamp'] = datetime.now().isoformat()
        
        # Apply modifications
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested keys like 'encoder_config.embedding_dim'
                parts = key.split('.')
                current = config_dict
                for part in parts[:-1]:
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[key] = value
        
        return self.__class__.from_dict(config_dict)