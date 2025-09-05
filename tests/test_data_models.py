"""
Unit tests for data models in the SMOTE image synthesis pipeline.
"""

import unittest
import numpy as np
import tempfile
import os
from datetime import datetime
from pathlib import Path

from smote_image_synthesis.data.models import EmbeddingData, SyntheticSample


class TestEmbeddingData(unittest.TestCase):
    """Test cases for EmbeddingData class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_embedding = np.array([1.0, 2.0, 3.0, 4.0])
        self.valid_label = 1
        self.valid_source_id = "image_001"
        self.valid_metadata = {"source_path": "/path/to/image.jpg", "preprocessing": "normalized"}
        
    def test_valid_embedding_data_creation(self):
        """Test creating valid EmbeddingData instance."""
        embedding_data = EmbeddingData(
            embedding=self.valid_embedding,
            label=self.valid_label,
            source_image_id=self.valid_source_id,
            metadata=self.valid_metadata
        )
        
        is_valid, errors = embedding_data.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
    def test_embedding_validation_none_embedding(self):
        """Test validation with None embedding."""
        embedding_data = EmbeddingData(
            embedding=None,
            label=self.valid_label,
            source_image_id=self.valid_source_id
        )
        
        is_valid, errors = embedding_data.validate()
        self.assertFalse(is_valid)
        self.assertIn("Embedding cannot be None", errors)
        
    def test_embedding_validation_empty_embedding(self):
        """Test validation with empty embedding."""
        embedding_data = EmbeddingData(
            embedding=np.array([]),
            label=self.valid_label,
            source_image_id=self.valid_source_id
        )
        
        is_valid, errors = embedding_data.validate()
        self.assertFalse(is_valid)
        self.assertIn("Embedding cannot be empty", errors)
        
    def test_embedding_validation_nan_values(self):
        """Test validation with NaN values in embedding."""
        embedding_data = EmbeddingData(
            embedding=np.array([1.0, np.nan, 3.0]),
            label=self.valid_label,
            source_image_id=self.valid_source_id
        )
        
        is_valid, errors = embedding_data.validate()
        self.assertFalse(is_valid)
        self.assertIn("Embedding contains NaN values", errors)
        
    def test_embedding_validation_inf_values(self):
        """Test validation with infinite values in embedding."""
        embedding_data = EmbeddingData(
            embedding=np.array([1.0, np.inf, 3.0]),
            label=self.valid_label,
            source_image_id=self.valid_source_id
        )
        
        is_valid, errors = embedding_data.validate()
        self.assertFalse(is_valid)
        self.assertIn("Embedding contains infinite values", errors)
        
    def test_label_validation_negative(self):
        """Test validation with negative label."""
        embedding_data = EmbeddingData(
            embedding=self.valid_embedding,
            label=-1,
            source_image_id=self.valid_source_id
        )
        
        is_valid, errors = embedding_data.validate()
        self.assertFalse(is_valid)
        self.assertIn("Label must be non-negative", errors)
        
    def test_label_validation_non_integer(self):
        """Test validation with non-integer label."""
        embedding_data = EmbeddingData(
            embedding=self.valid_embedding,
            label=1.5,
            source_image_id=self.valid_source_id
        )
        
        is_valid, errors = embedding_data.validate()
        self.assertFalse(is_valid)
        self.assertIn("Label must be an integer", errors)
        
    def test_source_id_validation_empty(self):
        """Test validation with empty source image ID."""
        embedding_data = EmbeddingData(
            embedding=self.valid_embedding,
            label=self.valid_label,
            source_image_id=""
        )
        
        is_valid, errors = embedding_data.validate()
        self.assertFalse(is_valid)
        self.assertIn("Source image ID cannot be empty", errors)
        
    def test_get_embedding_dim(self):
        """Test getting embedding dimension."""
        embedding_data = EmbeddingData(
            embedding=self.valid_embedding,
            label=self.valid_label,
            source_image_id=self.valid_source_id
        )
        
        self.assertEqual(embedding_data.get_embedding_dim(), 4)
        
    def test_normalize_embedding(self):
        """Test embedding normalization."""
        embedding_data = EmbeddingData(
            embedding=np.array([3.0, 4.0]),  # L2 norm = 5.0
            label=self.valid_label,
            source_image_id=self.valid_source_id
        )
        
        normalized = embedding_data.normalize_embedding()
        expected_norm = np.linalg.norm(normalized.embedding)
        
        self.assertAlmostEqual(expected_norm, 1.0, places=6)
        self.assertAlmostEqual(normalized.embedding[0], 0.6, places=6)
        self.assertAlmostEqual(normalized.embedding[1], 0.8, places=6)
        
    def test_to_dict_and_from_dict(self):
        """Test serialization to dictionary and back."""
        original = EmbeddingData(
            embedding=self.valid_embedding,
            label=self.valid_label,
            source_image_id=self.valid_source_id,
            metadata=self.valid_metadata
        )
        
        # Convert to dict and back
        data_dict = original.to_dict()
        reconstructed = EmbeddingData.from_dict(data_dict)
        
        # Check all fields match
        np.testing.assert_array_equal(original.embedding, reconstructed.embedding)
        self.assertEqual(original.label, reconstructed.label)
        self.assertEqual(original.source_image_id, reconstructed.source_image_id)
        self.assertEqual(original.metadata, reconstructed.metadata)
        
    def test_file_serialization(self):
        """Test saving and loading from file."""
        original = EmbeddingData(
            embedding=self.valid_embedding,
            label=self.valid_label,
            source_image_id=self.valid_source_id,
            metadata=self.valid_metadata
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_embedding.pkl")
            
            # Save and load
            original.save_to_file(filepath)
            loaded = EmbeddingData.load_from_file(filepath)
            
            # Check all fields match
            np.testing.assert_array_equal(original.embedding, loaded.embedding)
            self.assertEqual(original.label, loaded.label)
            self.assertEqual(original.source_image_id, loaded.source_image_id)
            self.assertEqual(original.metadata, loaded.metadata)


class TestSyntheticSample(unittest.TestCase):
    """Test cases for SyntheticSample class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_embedding = np.array([1.0, 2.0, 3.0, 4.0])
        self.valid_parents = ["parent_001", "parent_002"]
        self.valid_weights = np.array([0.6, 0.4])
        self.valid_quality = 0.85
        self.valid_decoder = "autoencoder"
        self.valid_image = np.random.rand(64, 64, 3)
        
    def test_valid_synthetic_sample_creation(self):
        """Test creating valid SyntheticSample instance."""
        sample = SyntheticSample(
            synthetic_embedding=self.valid_embedding,
            parent_embeddings=self.valid_parents,
            interpolation_weights=self.valid_weights,
            quality_score=self.valid_quality,
            decoder_type=self.valid_decoder,
            generated_image=self.valid_image
        )
        
        is_valid, errors = sample.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
    def test_validation_none_embedding(self):
        """Test validation with None synthetic embedding."""
        sample = SyntheticSample(
            synthetic_embedding=None,
            parent_embeddings=self.valid_parents,
            interpolation_weights=self.valid_weights,
            quality_score=self.valid_quality,
            decoder_type=self.valid_decoder
        )
        
        is_valid, errors = sample.validate()
        self.assertFalse(is_valid)
        self.assertIn("Synthetic embedding cannot be None", errors)
        
    def test_validation_empty_parents(self):
        """Test validation with empty parent embeddings list."""
        sample = SyntheticSample(
            synthetic_embedding=self.valid_embedding,
            parent_embeddings=[],
            interpolation_weights=self.valid_weights,
            quality_score=self.valid_quality,
            decoder_type=self.valid_decoder
        )
        
        is_valid, errors = sample.validate()
        self.assertFalse(is_valid)
        self.assertIn("Must have at least one parent embedding", errors)
        
    def test_validation_weights_not_sum_to_one(self):
        """Test validation with weights that don't sum to 1."""
        sample = SyntheticSample(
            synthetic_embedding=self.valid_embedding,
            parent_embeddings=self.valid_parents,
            interpolation_weights=np.array([0.3, 0.4]),  # Sum = 0.7
            quality_score=self.valid_quality,
            decoder_type=self.valid_decoder
        )
        
        is_valid, errors = sample.validate()
        self.assertFalse(is_valid)
        self.assertIn("Interpolation weights must sum to 1.0", errors)
        
    def test_validation_negative_weights(self):
        """Test validation with negative weights."""
        sample = SyntheticSample(
            synthetic_embedding=self.valid_embedding,
            parent_embeddings=self.valid_parents,
            interpolation_weights=np.array([-0.1, 1.1]),
            quality_score=self.valid_quality,
            decoder_type=self.valid_decoder
        )
        
        is_valid, errors = sample.validate()
        self.assertFalse(is_valid)
        self.assertIn("Interpolation weights must be non-negative", errors)
        
    def test_validation_mismatched_weights_parents(self):
        """Test validation with mismatched number of weights and parents."""
        sample = SyntheticSample(
            synthetic_embedding=self.valid_embedding,
            parent_embeddings=self.valid_parents,  # 2 parents
            interpolation_weights=np.array([1.0]),  # 1 weight
            quality_score=self.valid_quality,
            decoder_type=self.valid_decoder
        )
        
        is_valid, errors = sample.validate()
        self.assertFalse(is_valid)
        self.assertIn("Number of weights must match number of parent embeddings", errors)
        
    def test_validation_negative_quality_score(self):
        """Test validation with negative quality score."""
        sample = SyntheticSample(
            synthetic_embedding=self.valid_embedding,
            parent_embeddings=self.valid_parents,
            interpolation_weights=self.valid_weights,
            quality_score=-0.1,
            decoder_type=self.valid_decoder
        )
        
        is_valid, errors = sample.validate()
        self.assertFalse(is_valid)
        self.assertIn("Quality score must be non-negative", errors)
        
    def test_get_interpolation_info(self):
        """Test getting interpolation information."""
        sample = SyntheticSample(
            synthetic_embedding=self.valid_embedding,
            parent_embeddings=self.valid_parents,
            interpolation_weights=self.valid_weights,
            quality_score=self.valid_quality,
            decoder_type=self.valid_decoder
        )
        
        info = sample.get_interpolation_info()
        
        self.assertEqual(info['num_parents'], 2)
        self.assertEqual(info['parent_ids'], self.valid_parents)
        self.assertEqual(info['max_weight'], 0.6)
        self.assertEqual(info['min_weight'], 0.4)
        self.assertGreater(info['weight_entropy'], 0)
        
    def test_to_dict_and_from_dict(self):
        """Test serialization to dictionary and back."""
        original = SyntheticSample(
            synthetic_embedding=self.valid_embedding,
            parent_embeddings=self.valid_parents,
            interpolation_weights=self.valid_weights,
            quality_score=self.valid_quality,
            decoder_type=self.valid_decoder,
            generated_image=self.valid_image
        )
        
        # Convert to dict and back
        data_dict = original.to_dict()
        reconstructed = SyntheticSample.from_dict(data_dict)
        
        # Check all fields match
        np.testing.assert_array_equal(original.synthetic_embedding, reconstructed.synthetic_embedding)
        self.assertEqual(original.parent_embeddings, reconstructed.parent_embeddings)
        np.testing.assert_array_equal(original.interpolation_weights, reconstructed.interpolation_weights)
        self.assertEqual(original.quality_score, reconstructed.quality_score)
        self.assertEqual(original.decoder_type, reconstructed.decoder_type)
        np.testing.assert_array_equal(original.generated_image, reconstructed.generated_image)
        
    def test_file_serialization(self):
        """Test saving and loading from file."""
        original = SyntheticSample(
            synthetic_embedding=self.valid_embedding,
            parent_embeddings=self.valid_parents,
            interpolation_weights=self.valid_weights,
            quality_score=self.valid_quality,
            decoder_type=self.valid_decoder,
            generated_image=self.valid_image
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_sample.pkl")
            
            # Save and load
            original.save_to_file(filepath)
            loaded = SyntheticSample.load_from_file(filepath)
            
            # Check all fields match
            np.testing.assert_array_equal(original.synthetic_embedding, loaded.synthetic_embedding)
            self.assertEqual(original.parent_embeddings, loaded.parent_embeddings)
            np.testing.assert_array_equal(original.interpolation_weights, loaded.interpolation_weights)
            self.assertEqual(original.quality_score, loaded.quality_score)
            self.assertEqual(original.decoder_type, loaded.decoder_type)
            np.testing.assert_array_equal(original.generated_image, loaded.generated_image)


if __name__ == '__main__':
    unittest.main()


class TestPipelineConfig(unittest.TestCase):
    """Test cases for PipelineConfig and related configuration classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        from smote_image_synthesis.data.models import (
            PipelineConfig, EncoderConfig, SMOTEConfig, 
            DecoderConfig, QualityConfig
        )
        self.PipelineConfig = PipelineConfig
        self.EncoderConfig = EncoderConfig
        self.SMOTEConfig = SMOTEConfig
        self.DecoderConfig = DecoderConfig
        self.QualityConfig = QualityConfig
        
    def test_encoder_config_validation_valid(self):
        """Test valid encoder configuration."""
        config = self.EncoderConfig(
            architecture="resnet50",
            embedding_dim=512,
            pretrained=True,
            dropout_rate=0.1
        )
        
        is_valid, errors = config.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
    def test_encoder_config_validation_invalid_architecture(self):
        """Test encoder config with invalid architecture."""
        config = self.EncoderConfig(architecture="invalid_arch")
        
        is_valid, errors = config.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("Architecture must be one of" in error for error in errors))
        
    def test_encoder_config_validation_negative_embedding_dim(self):
        """Test encoder config with negative embedding dimension."""
        config = self.EncoderConfig(embedding_dim=-10)
        
        is_valid, errors = config.validate()
        self.assertFalse(is_valid)
        self.assertIn("Embedding dimension must be positive", errors)
        
    def test_smote_config_validation_valid(self):
        """Test valid SMOTE configuration."""
        config = self.SMOTEConfig(
            k_neighbors=5,
            sampling_strategy="auto",
            use_clustering=True,
            distance_threshold=0.5
        )
        
        is_valid, errors = config.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
    def test_smote_config_validation_invalid_k_neighbors(self):
        """Test SMOTE config with invalid k_neighbors."""
        config = self.SMOTEConfig(k_neighbors=0)
        
        is_valid, errors = config.validate()
        self.assertFalse(is_valid)
        self.assertIn("k_neighbors must be positive", errors)
        
    def test_smote_config_validation_invalid_sampling_strategy(self):
        """Test SMOTE config with invalid sampling strategy."""
        config = self.SMOTEConfig(sampling_strategy="invalid")
        
        is_valid, errors = config.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("Sampling strategy must be one of" in error for error in errors))
        
    def test_decoder_config_validation_valid(self):
        """Test valid decoder configuration."""
        config = self.DecoderConfig(
            decoder_type="autoencoder",
            image_shape=(64, 64, 3),
            learning_rate=0.001,
            batch_size=32
        )
        
        is_valid, errors = config.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
    def test_decoder_config_validation_invalid_decoder_type(self):
        """Test decoder config with invalid decoder type."""
        config = self.DecoderConfig(decoder_type="invalid_decoder")
        
        is_valid, errors = config.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("Decoder type must be one of" in error for error in errors))
        
    def test_decoder_config_validation_invalid_image_shape(self):
        """Test decoder config with invalid image shape."""
        config = self.DecoderConfig(image_shape=(64, 64))  # Missing channel dimension
        
        is_valid, errors = config.validate()
        self.assertFalse(is_valid)
        self.assertIn("Image shape must be 3D (height, width, channels)", errors)
        
    def test_quality_config_validation_valid(self):
        """Test valid quality configuration."""
        config = self.QualityConfig(
            metrics=["fid", "lpips", "ssim"],
            fid_batch_size=50,
            compute_diversity=True,
            quality_threshold=0.7
        )
        
        is_valid, errors = config.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
    def test_quality_config_validation_invalid_metrics(self):
        """Test quality config with invalid metrics."""
        config = self.QualityConfig(metrics=["invalid_metric"])
        
        is_valid, errors = config.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("Invalid metric" in error for error in errors))
        
    def test_quality_config_validation_empty_metrics(self):
        """Test quality config with empty metrics list."""
        config = self.QualityConfig(metrics=[])
        
        is_valid, errors = config.validate()
        self.assertFalse(is_valid)
        self.assertIn("At least one quality metric must be specified", errors)
        
    def test_pipeline_config_creation_default(self):
        """Test creating default pipeline configuration."""
        config = self.PipelineConfig()
        
        is_valid, errors = config.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Check default values
        self.assertEqual(config.config_name, "default_config")
        self.assertEqual(config.output_dir, "./outputs")
        self.assertIsNone(config.seed)
        
    def test_pipeline_config_validation_invalid_components(self):
        """Test pipeline config validation with invalid components."""
        config = self.PipelineConfig(
            encoder_config=self.EncoderConfig(embedding_dim=-10),  # Invalid
            smote_config=self.SMOTEConfig(k_neighbors=0),  # Invalid
            config_name=""  # Invalid
        )
        
        is_valid, errors = config.validate()
        self.assertFalse(is_valid)
        
        # Should have errors from encoder, smote, and config name
        self.assertTrue(any("Encoder:" in error for error in errors))
        self.assertTrue(any("SMOTE:" in error for error in errors))
        self.assertIn("Config name must be a non-empty string", errors)
        
    def test_pipeline_config_consistency_warnings(self):
        """Test pipeline config consistency warnings."""
        config = self.PipelineConfig(
            decoder_config=self.DecoderConfig(batch_size=100),  # Large batch size
            quality_config=self.QualityConfig(fid_batch_size=50),  # Smaller FID batch
            smote_config=self.SMOTEConfig(k_neighbors=25)  # Large k_neighbors
        )
        
        warnings = config.get_consistency_warnings()
        
        # Should have warnings about batch size and k_neighbors
        self.assertTrue(any("batch size" in warning.lower() for warning in warnings))
        self.assertTrue(any("k_neighbors" in warning for warning in warnings))
        
    def test_pipeline_config_to_dict_and_from_dict(self):
        """Test pipeline config serialization to dictionary and back."""
        original = self.PipelineConfig(
            config_name="test_config",
            output_dir="./test_outputs",
            seed=42
        )
        
        # Convert to dict and back
        config_dict = original.to_dict()
        reconstructed = self.PipelineConfig.from_dict(config_dict)
        
        # Check all fields match
        self.assertEqual(original.config_name, reconstructed.config_name)
        self.assertEqual(original.output_dir, reconstructed.output_dir)
        self.assertEqual(original.seed, reconstructed.seed)
        self.assertEqual(original.encoder_config.architecture, reconstructed.encoder_config.architecture)
        self.assertEqual(original.smote_config.k_neighbors, reconstructed.smote_config.k_neighbors)
        
    def test_pipeline_config_file_serialization(self):
        """Test saving and loading pipeline config from file."""
        original = self.PipelineConfig(
            config_name="file_test_config",
            output_dir="./file_test_outputs",
            seed=123
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_config.json")
            
            # Save and load
            original.save_config(filepath)
            loaded = self.PipelineConfig.load_config(filepath)
            
            # Check all fields match
            self.assertEqual(original.config_name, loaded.config_name)
            self.assertEqual(original.output_dir, loaded.output_dir)
            self.assertEqual(original.seed, loaded.seed)
            self.assertEqual(original.encoder_config.architecture, loaded.encoder_config.architecture)
            
    def test_pipeline_config_create_variant(self):
        """Test creating configuration variants."""
        base_config = self.PipelineConfig(
            config_name="base_config",
            seed=42
        )
        
        # Create variant with modified parameters
        variant = base_config.create_variant(
            "variant_config",
            seed=123,
            output_dir="./variant_outputs"
        )
        
        # Check variant has new values
        self.assertEqual(variant.config_name, "variant_config")
        self.assertEqual(variant.seed, 123)
        self.assertEqual(variant.output_dir, "./variant_outputs")
        
        # Check base config is unchanged
        self.assertEqual(base_config.config_name, "base_config")
        self.assertEqual(base_config.seed, 42)
        
    def test_pipeline_config_create_variant_nested(self):
        """Test creating configuration variants with nested parameter changes."""
        base_config = self.PipelineConfig()
        
        # Create variant with nested parameter changes
        variant = base_config.create_variant(
            "nested_variant"
        )
        
        # Check variant was created
        self.assertEqual(variant.config_name, "nested_variant")
        self.assertNotEqual(variant.creation_timestamp, base_config.creation_timestamp)