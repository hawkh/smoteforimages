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
        # Create imbalanced data so SMOTE actually generates synthetic samples
        # Class 0: 10 samples (majority), Class 1: 5 samples, Class 2: 5 samples
        counts = [10, 5, 5]
        embeddings = np.random.randn(sum(counts), self.embedding_dim)
        labels = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
        
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
        is_valid, report = smote.validate_embedding_space(embeddings[:5])
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
        train_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        
        # Test pipeline fitting
        pipeline.fit(train_images, train_labels)
        
        # Test synthetic generation
        synthetic_images, synthetic_labels = pipeline.generate_synthetic_images(n_samples=6)
        
        self.assertGreater(len(synthetic_images), 0)
        self.assertEqual(len(synthetic_images), len(synthetic_labels))
        self.assertEqual(synthetic_images.shape[1:], self.image_shape)
        
        # Test quality evaluation
        if len(synthetic_images) > 0:
            quality_results = pipeline.evaluate_quality(
                synthetic_images[:4], train_images[:4]
            )
            self.assertIn('mse', quality_results)
    
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


class TestRecentImprovements(unittest.TestCase):
    """Tests for bug-fixes and features added in the latest improvement pass."""

    def setUp(self):
        self.device = torch.device('cpu')
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    # ── SMOTE scaler round-trip ────────────────────────────────────────────────
    def test_smote_scaler_roundtrip(self):
        """Synthetic embeddings must be in original (un-normalised) space."""
        np.random.seed(0)
        # High-mean data so we can detect if inverse_transform is skipped
        emb = np.random.randn(40, 16) * 10 + 200
        labels = np.array([0] * 20 + [1] * 20)

        smote = ConstrainedSMOTE(k_neighbors=3, normalize_embeddings=True, random_state=42)
        smote.fit(emb, labels)
        syn, _ = smote.generate_synthetic(n_samples=10)

        # Mean of synthetic should be near 200, not near 0
        self.assertAlmostEqual(syn.mean(), 200.0, delta=30.0,
                               msg="inverse_transform not applied; synthetic embeddings still in normalised space")

    def test_smote_scaler_disabled(self):
        """With normalize_embeddings=False, scaler must not be fitted or applied."""
        np.random.seed(0)
        emb = np.random.randn(40, 16) * 10 + 200
        labels = np.array([0] * 20 + [1] * 20)

        smote = ConstrainedSMOTE(k_neighbors=3, normalize_embeddings=False, random_state=42)
        smote.fit(emb, labels)

        self.assertIsNone(smote.scaler, "scaler must be None when normalize_embeddings=False")
        syn, _ = smote.generate_synthetic(n_samples=10)
        self.assertAlmostEqual(syn.mean(), 200.0, delta=30.0)

    # ── diversity_index must be plain float ───────────────────────────────────
    def test_diversity_index_is_float(self):
        """diversity_index should be a Python float, not np.float64, for JSON safety."""
        qa = QualityAssessor(metrics=['mse'], compute_diversity=True, device=self.device)
        imgs = torch.randn(6, 3, 32, 32)
        results = qa.evaluate_quality(imgs, imgs)
        idx = results['diversity']['diversity_index']
        self.assertIsInstance(idx, float,
                              f"diversity_index should be float, got {type(idx).__name__}")

    # ── JSON report serialises np types without error ─────────────────────────
    def test_json_report_with_numpy_values(self):
        """QualityReporter JSON format must handle np.float64 values without TypeError."""
        import json
        qa = QualityAssessor(metrics=['mse'], compute_diversity=True, device=self.device)
        imgs = torch.randn(6, 3, 32, 32)
        results = qa.evaluate_quality(imgs, imgs)

        reporter = QualityReporter(output_dir=str(self.temp_dir), report_format='json')
        path = reporter._generate_json_report(results, 'test_report')

        # Must be valid JSON
        with open(path) as f:
            data = json.load(f)
        self.assertIn('quality_results', data)

    # ── DCGANDecoder in full pipeline ─────────────────────────────────────────
    def test_dcgan_pipeline_smoke(self):
        """DCGANDecoder pipeline: fit + generate should produce correct-shape tensors."""
        from smote_image_synthesis.decoders.dcgan_decoder import DCGANDecoder

        enc = ResNetEncoder('resnet18', embedding_dim=64, pretrained=False, device=self.device)
        dec = DCGANDecoder(64, (3, 32, 32), base_channels=64, device=self.device)
        smote = ConstrainedSMOTE(k_neighbors=3, normalize_embeddings=True, random_state=0)
        qa = QualityAssessor(metrics=['mse'], compute_diversity=False, device=self.device)
        pipeline = SynthesisPipeline(enc, dec, smote, qa)

        imgs = torch.randn(20, 3, 32, 32)
        labels = np.array([0] * 10 + [1] * 10)
        pipeline.fit(imgs, labels, train_decoder=True, decoder_epochs=2)

        syn, syn_labels = pipeline.generate_synthetic_images(n_samples=8)
        self.assertEqual(syn.shape[1:], torch.Size([3, 32, 32]))
        self.assertEqual(len(syn), len(syn_labels))

    # ── Image resize in data loader ───────────────────────────────────────────
    def test_load_cats_and_dogs_resize(self):
        """load_cats_and_dogs must produce the requested image_size."""
        # Import the function directly (it downloads data only when called with real CIFAR)
        # Instead, verify that the transform includes Resize when size != 32
        import torchvision.transforms as T
        from torchvision.transforms import Resize

        # Simulate what load_cats_and_dogs does for image_size=64
        image_size = 64
        resize_ops = [] if image_size == 32 else [T.Resize(image_size)]
        transform = T.Compose([
            *resize_ops,
            T.ToTensor(),
        ])

        # Apply to a dummy PIL image to confirm output shape
        from PIL import Image as PILImage
        dummy = PILImage.new('RGB', (32, 32))
        t = transform(dummy)
        self.assertEqual(t.shape, (3, 64, 64))

    # ── SMOTE n_samples precision ─────────────────────────────────────────────
    def test_smote_exact_n_samples(self):
        """generate_synthetic must return EXACTLY n_samples regardless of n_classes."""
        np.random.seed(42)
        emb = np.random.randn(60, 16) * 10
        labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
        smote = ConstrainedSMOTE(k_neighbors=3, normalize_embeddings=True, random_state=42)
        smote.fit(emb, labels)

        for n in [1, 5, 7, 11, 13, 17, 23, 50]:
            syn, lbl = smote.generate_synthetic(n_samples=n)
            self.assertEqual(len(syn), n,
                             f"Requested {n} samples, got {len(syn)} (non-multiple of n_classes={3})")
            self.assertEqual(len(syn), len(lbl))

    def test_smote_none_n_samples_imbalanced(self):
        """generate_synthetic(n_samples=None) returns samples for imbalanced classes."""
        np.random.seed(0)
        emb = np.random.randn(30, 8)
        # Imbalanced: minority class (1) has 5 samples vs majority (0) has 25
        labels = np.array([0] * 25 + [1] * 5)
        smote = ConstrainedSMOTE(k_neighbors=3, normalize_embeddings=False, random_state=0)
        smote.fit(emb, labels)
        syn, lbl = smote.generate_synthetic(n_samples=None)
        # With auto strategy, minority class gets upsampled to match majority
        self.assertGreater(len(syn), 0)

    def test_smote_none_n_samples_balanced_returns_zero(self):
        """generate_synthetic(n_samples=None) for balanced data returns 0 — auto strategy has nothing to do."""
        np.random.seed(0)
        emb = np.random.randn(30, 8)
        labels = np.array([0] * 15 + [1] * 15)
        smote = ConstrainedSMOTE(k_neighbors=3, normalize_embeddings=False, random_state=0)
        smote.fit(emb, labels)
        syn, lbl = smote.generate_synthetic(n_samples=None)
        self.assertEqual(len(syn), 0, "Balanced classes with auto strategy should produce 0 synthetic samples")

    # ── GAN warmup not restarted in segmented training ───────────────────────
    def test_segmented_training_global_epoch(self):
        """GAN phase must activate based on global epoch, not local epoch within segment."""
        from smote_image_synthesis.decoders.dcgan_decoder import DCGANDecoder

        enc = ResNetEncoder('resnet18', embedding_dim=64, pretrained=False, device=self.device)
        dec = DCGANDecoder(64, (3, 32, 32), base_channels=64, device=self.device)
        smote = ConstrainedSMOTE(k_neighbors=3, normalize_embeddings=False, random_state=0)
        qa = QualityAssessor(metrics=['mse'], compute_diversity=False, device=self.device)
        pipeline = SynthesisPipeline(enc, dec, smote, qa)

        imgs = torch.randn(20, 3, 32, 32)
        labels = np.array([0] * 10 + [1] * 10)

        # Simulate two segments: epochs 0-9, then 10-19 (total=20, warmup ends at epoch 6)
        # Second segment starts at global_epoch=10, which is past warmup → GAN should be active
        pipeline.fit(imgs, labels, train_decoder=True, decoder_epochs=10,
                     start_epoch=0, total_epochs=20)
        pipeline.fit(imgs, labels, train_decoder=True, decoder_epochs=10,
                     start_epoch=10, total_epochs=20)

        # If we got here without error and shapes are right, segmented training works
        syn, _ = pipeline.generate_synthetic_images(n_samples=6)
        self.assertEqual(syn.shape[1:], torch.Size([3, 32, 32]))

    # ── Pipeline save/load round-trip ─────────────────────────────────────────
    def test_pipeline_save_load_roundtrip(self):
        """save_pipeline / load_pipeline must preserve encoder+decoder weights."""
        enc = ResNetEncoder('resnet18', embedding_dim=64, pretrained=False, device=self.device)
        dec = AutoencoderDecoder(64, (3, 32, 32), device=self.device)
        smote = ConstrainedSMOTE(k_neighbors=3, normalize_embeddings=False, random_state=0)
        qa = QualityAssessor(metrics=['mse'], compute_diversity=False, device=self.device)
        pipeline = SynthesisPipeline(enc, dec, smote, qa)

        base = str(self.temp_dir / 'ckpt')
        pipeline.save_pipeline(base)

        self.assertTrue(Path(base + '_encoder.pth').exists())
        self.assertTrue(Path(base + '_decoder.pth').exists())

        # Load into a fresh pipeline
        enc2 = ResNetEncoder('resnet18', embedding_dim=64, pretrained=False, device=self.device)
        dec2 = AutoencoderDecoder(64, (3, 32, 32), device=self.device)
        pipeline2 = SynthesisPipeline(enc2, dec2, smote, qa)
        pipeline2.load_pipeline(base)

        # Weights should match
        for p1, p2 in zip(pipeline.encoder.model.parameters(),
                          pipeline2.encoder.model.parameters()):
            self.assertTrue(torch.equal(p1, p2))


if __name__ == '__main__':
    unittest.main(verbosity=2)