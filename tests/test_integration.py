"""
Integration tests for the SMOTE image synthesis pipeline.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from smote_image_synthesis.data.models import PipelineConfig, EmbeddingData, SyntheticSample
from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder
from smote_image_synthesis.decoders.autoencoder_decoder import AutoencoderDecoder
from smote_image_synthesis.decoders.vae_decoder import VAEDecoder
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE
from smote_image_synthesis.quality.assessor import QualityAssessor
from smote_image_synthesis.quality.reporter import QualityReporter
from smote_image_synthesis.pipeline import SynthesisPipeline


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')  # Use CPU for testing
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Test parameters
        self.embedding_dim = 128
        self.image_shape = (3, 64, 64)
        self.batch_size = 4
        self.n_classes = 3
        
        # Create test data
        self.test_images = torch.randn(self.batch_size, *self.image_shape)
        self.test_labels = np.array([0, 1, 2, 0])
        self.test_embeddings = torch.randn(self.batch_size, self.embedding_dim)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_pipeline_config_validation(self):
        """Test pipeline configuration validation."""
        config = PipelineConfig()
        
        # Test validation
        is_valid, errors = config.validate()
        self.assertTrue(is_valid, f"Default config should be valid: {errors}")
        
        # Test invalid config
        config.encoder_config.embedding_dim = -1
        is_valid, errors = config.validate()
        self.assertFalse(is_valid, "Negative embedding_dim should be invalid")
    
    def test_embedding_data_model(self):
        """Test EmbeddingData model functionality."""
        embedding = np.random.randn(self.embedding_dim)
        
        data = EmbeddingData(
            embedding=embedding,
            label=0,
            source_image_id="test_001"
        )
        
        # Test validation
        is_valid, errors = data.validate()
        self.assertTrue(is_valid, f"Valid embedding data should pass validation: {errors}")
        
        # Test serialization
        data_dict = data.to_dict()
        reconstructed = EmbeddingData.from_dict(data_dict)
        
        np.testing.assert_array_equal(data.embedding, reconstructed.embedding)
        self.assertEqual(data.label, reconstructed.label)
        self.assertEqual(data.source_image_id, reconstructed.source_image_id)
    
    def test_synthetic_sample_model(self):
        """Test SyntheticSample model functionality."""
        synthetic_embedding = np.random.randn(self.embedding_dim)
        parent_embeddings = ["parent_001", "parent_002"]
        interpolation_weights = np.array([0.6, 0.4])
        
        sample = SyntheticSample(
            synthetic_embedding=synthetic_embedding,
            parent_embeddings=parent_embeddings,
            interpolation_weights=interpolation_weights,
            quality_score=0.85,
            decoder_type="autoencoder"
        )
        
        # Test validation
        is_valid, errors = sample.validate()
        self.assertTrue(is_valid, f"Valid synthetic sample should pass validation: {errors}")
        
        # Test interpolation info
        info = sample.get_interpolation_info()
        self.assertEqual(info['num_parents'], 2)
        self.assertAlmostEqual(info['max_weight'], 0.6)
    
    def test_resnet_encoder_basic(self):
        """Test basic ResNet encoder functionality."""
        encoder = ResNetEncoder(
            architecture='resnet18',
            embedding_dim=self.embedding_dim,
            pretrained=False,  # Don't download pretrained weights in tests
            device=self.device
        )
        
        # Test encoding
        embeddings = encoder.encode(self.test_images)
        
        self.assertEqual(embeddings.shape, (self.batch_size, self.embedding_dim))
        self.assertFalse(torch.isnan(embeddings).any())
        self.assertFalse(torch.isinf(embeddings).any())
        
        # Test model info
        info = encoder.get_model_info()
        self.assertEqual(info['architecture'], 'resnet18')
        self.assertEqual(info['embedding_dim'], self.embedding_dim)
    
    def test_autoencoder_decoder_basic(self):
        """Test basic AutoencoderDecoder functionality."""
        decoder = AutoencoderDecoder(
            embedding_dim=self.embedding_dim,
            image_shape=self.image_shape,
            device=self.device
        )
        
        # Test decoding
        decoded_images = decoder.decode(self.test_embeddings)
        
        self.assertEqual(decoded_images.shape, (self.batch_size, *self.image_shape))
        self.assertFalse(torch.isnan(decoded_images).any())
        self.assertFalse(torch.isinf(decoded_images).any())
        
        # Test model info
        info = decoder.get_model_info()
        self.assertEqual(info['embedding_dim'], self.embedding_dim)
        self.assertEqual(info['image_shape'], self.image_shape)
    
    def test_vae_decoder_basic(self):
        """Test basic VAEDecoder functionality."""
        decoder = VAEDecoder(
            embedding_dim=self.embedding_dim,
            image_shape=self.image_shape,
            latent_dim=64,
            device=self.device
        )
        
        # Test decoding
        decoded_images = decoder.decode(self.test_embeddings)
        
        self.assertEqual(decoded_images.shape, (self.batch_size, *self.image_shape))
        self.assertFalse(torch.isnan(decoded_images).any())
        self.assertFalse(torch.isinf(decoded_images).any())
        
        # Test encode-decode
        reconstructed, mu, log_var = decoder.encode_and_decode(self.test_embeddings)
        
        self.assertEqual(reconstructed.shape, (self.batch_size, *self.image_shape))
        self.assertEqual(mu.shape, (self.batch_size, 64))
        self.assertEqual(log_var.shape, (self.batch_size, 64))
        
        # Test KL divergence computation
        kl_div = decoder.compute_kl_divergence(mu, log_var)
        self.assertEqual(kl_div.shape, (self.batch_size,))
    
    def test_constrained_smote_basic(self):
        """Test basic ConstrainedSMOTE functionality."""
        # Create test data with sufficient samples and imbalance to trigger SMOTE
        n_samples = 30
        embeddings = np.random.randn(n_samples, self.embedding_dim)
        # Ensure imbalance: class 0 (12), class 1 (12), class 2 (6)
        # All have >= 4 samples for k_neighbors=3
        labels = np.concatenate([
            np.zeros(12, dtype=int),
            np.ones(12, dtype=int),
            np.full(6, 2, dtype=int)
        ])
        
        smote = ConstrainedSMOTE(
            k_neighbors=3,
            sampling_strategy='auto',
            use_clustering=False,  # Disable for basic test
            normalize_embeddings=False
        )
        
        # Test fitting
        smote.fit(embeddings, labels)
        self.assertTrue(smote.is_fitted)
        
        # Test generation
        synthetic_embeddings, synthetic_labels = smote.generate_synthetic()
        
        self.assertGreater(len(synthetic_embeddings), 0)
        self.assertEqual(len(synthetic_embeddings), len(synthetic_labels))
        self.assertEqual(synthetic_embeddings.shape[1], self.embedding_dim)
        
        # Test validation
        is_valid = smote.validate_embedding_space(embeddings[:5])
        self.assertTrue(is_valid)
    
    def test_quality_assessor_basic(self):
        """Test basic QualityAssessor functionality."""
        assessor = QualityAssessor(
            metrics=['mse', 'mae'],  # Use simple metrics for testing
            device=self.device
        )
        
        # Test quality evaluation
        results = assessor.evaluate_quality(self.test_images, self.test_images)
        
        self.assertIn('metrics', results)
        self.assertIn('mse', results['metrics'])
        self.assertIn('mae', results['metrics'])
        
        # MSE should be 0 when comparing identical images
        self.assertAlmostEqual(results['metrics']['mse'], 0.0, places=6)
        
        # Test diversity metrics
        diversity = assessor.compute_diversity_metrics(self.test_images)
        
        self.assertIn('mean_pairwise_distance', diversity)
        self.assertIn('diversity_index', diversity)
        self.assertGreaterEqual(diversity['mean_pairwise_distance'], 0)
    
    def test_quality_reporter_basic(self):
        """Test basic QualityReporter functionality."""
        reporter = QualityReporter(
            output_dir=str(self.temp_dir),
            save_plots=False  # Don't save plots in tests
        )
        
        # Create mock quality results
        quality_results = {
            'metrics': {
                'mse': 0.05,
                'mae': 0.03,
                'ssim': 0.85
            },
            'diversity': {
                'mean_pairwise_distance': 2.5,
                'diversity_index': 0.7
            }
        }
        
        # Test report generation (text format for simplicity)
        reporter.report_format = 'txt'
        report_path = reporter.generate_comprehensive_report(
            quality_results, self.test_images, self.test_images, 'test_report'
        )
        
        self.assertTrue(Path(report_path).exists())
        
        # Test CSV export
        csv_path = reporter.export_metrics_csv(quality_results, 'test_metrics')
        self.assertTrue(Path(csv_path).exists())
    
    def test_pipeline_integration(self):
        """Test complete pipeline integration."""
        # Create components
        encoder = ResNetEncoder(
            architecture='resnet18',
            embedding_dim=self.embedding_dim,
            pretrained=False,
            device=self.device
        )
        
        decoder = AutoencoderDecoder(
            embedding_dim=self.embedding_dim,
            image_shape=self.image_shape,
            device=self.device
        )
        
        # Create SMOTE with minimal samples
        smote = ConstrainedSMOTE(
            k_neighbors=2,  # Reduce for small test dataset
            sampling_strategy='auto',
            use_clustering=False,
            normalize_embeddings=False
        )
        
        quality_assessor = QualityAssessor(
            metrics=['mse'],  # Simple metric for testing
            device=self.device
        )
        
        # Create pipeline
        pipeline = SynthesisPipeline(
            encoder=encoder,
            decoder=decoder,
            smote=smote,
            quality_assessor=quality_assessor
        )
        
        # Create larger test dataset for SMOTE
        n_train = 12  # Minimum for SMOTE to work
        train_images = torch.randn(n_train, *self.image_shape)
        # Imbalance to trigger SMOTE: 0(5), 1(4), 2(3)
        # All have >= 3 samples for k_neighbors=2
        train_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        
        # Test pipeline fitting
        pipeline.fit(train_images, train_labels)
        
        # Test synthetic generation
        synthetic_images, synthetic_labels = pipeline.generate_synthetic_images(n_samples=6)
        
        self.assertGreater(len(synthetic_images), 0)
        self.assertEqual(len(synthetic_images), len(synthetic_labels))
        self.assertEqual(synthetic_images.shape[1:], self.image_shape)
        
        # Test quality evaluation
        if len(synthetic_images) > 0:
            n_eval = min(len(synthetic_images), 4)
            quality_results = pipeline.evaluate_quality(
                synthetic_images[:n_eval], train_images[:n_eval]
            )
            self.assertIn('metrics', quality_results)
            self.assertIn('mse', quality_results['metrics'])
    
    def test_error_handling(self):
        """Test error handling in components."""
        # Test invalid input shapes
        encoder = ResNetEncoder(
            architecture='resnet18',
            embedding_dim=self.embedding_dim,
            pretrained=False,
            device=self.device
        )
        
        # Invalid image tensor
        with self.assertRaises(ValueError):
            invalid_images = torch.randn(2, 5, 64, 64)  # 5 channels invalid
            encoder.encode(invalid_images)
        
        # Test SMOTE with insufficient data
        smote = ConstrainedSMOTE(k_neighbors=5)
        
        with self.assertRaises(ValueError):
            # Too few samples for k_neighbors=5
            insufficient_embeddings = np.random.randn(3, self.embedding_dim)
            insufficient_labels = np.array([0, 1, 2])
            smote.fit(insufficient_embeddings, insufficient_labels)


class TestConfigurationManagement(unittest.TestCase):
    """Test configuration management and serialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_pipeline_config_serialization(self):
        """Test pipeline configuration serialization."""
        config = PipelineConfig(
            config_name="test_config",
            output_dir=str(self.temp_dir)
        )
        
        # Test saving
        config_path = self.temp_dir / "test_config.json"
        config.save_config(str(config_path))
        
        self.assertTrue(config_path.exists())
        
        # Test loading
        loaded_config = PipelineConfig.load_config(str(config_path))
        
        self.assertEqual(config.config_name, loaded_config.config_name)
        self.assertEqual(config.encoder_config.architecture, loaded_config.encoder_config.architecture)
        self.assertEqual(config.decoder_config.decoder_type, loaded_config.decoder_config.decoder_type)
    
    def test_config_variants(self):
        """Test configuration variant creation."""
        base_config = PipelineConfig(config_name="base")
        
        # Create variant with modified parameters
        variant = base_config.create_variant(
            "variant_test",
            **{"encoder_config.embedding_dim": 256, "decoder_config.learning_rate": 0.01}
        )
        
        self.assertEqual(variant.config_name, "variant_test")
        self.assertEqual(variant.encoder_config.embedding_dim, 256)
        self.assertEqual(variant.decoder_config.learning_rate, 0.01)
        
        # Original should be unchanged
        self.assertNotEqual(base_config.encoder_config.embedding_dim, 256)


if __name__ == '__main__':
    unittest.main(verbosity=2)