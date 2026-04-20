"""
Tests that directly verify patent-claimed features:
  - Self-attention (gamma=0 init, shape preservation, identity at gamma=0)
  - Class-conditional decoding (output differs with/without labels)
  - Discriminator (no BatchNorm, feature extraction list)
  - WGAN-GP gradient penalty (non-negative, finite)
  - L2-normalized encoder output (unit-norm)
  - Power-of-2 decoder validation
  - DCGANDecoder shape correctness
"""
import pytest
import torch
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from smote_image_synthesis.decoders.dcgan_decoder import DCGANDecoder
from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder
from smote_image_synthesis.pipeline import SynthesisPipeline
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE


# ───────────────────────────────────────────────────────────────────────
# Self-Attention Block (patent Section 4.4.3)
# ───────────────────────────────────────────────────────────────────────

class TestSelfAttention:

    @pytest.fixture
    def attn(self):
        from smote_image_synthesis.decoders.dcgan_decoder import SelfAttention2d
        return SelfAttention2d(in_channels=64)

    def test_gamma_starts_zero(self, attn):
        """gamma must be initialized to 0 (patent: zero-init for stable early training)."""
        assert attn.gamma.item() == 0.0

    def test_output_shape_matches_input(self, attn):
        """Self-attention is a residual block — output shape must match input."""
        x = torch.randn(2, 64, 16, 16)
        out = attn(x)
        assert out.shape == x.shape

    def test_identity_at_gamma_zero(self, attn):
        """With gamma=0, block must return input unchanged (F + 0*O = F)."""
        attn.eval()
        x = torch.randn(2, 64, 8, 8)
        with torch.no_grad():
            out = attn(x)
        torch.testing.assert_close(out, x)


# ───────────────────────────────────────────────────────────────────────
# Class-Conditional Decoder (patent Section 4.4.2)
# ───────────────────────────────────────────────────────────────────────

class TestClassConditionalDecoder:

    def test_conditioned_output_differs_from_unconditioned(self):
        """Class embedding must change the decoder output."""
        dec = DCGANDecoder(64, (3, 32, 32), base_channels=64,
                           num_classes=3, device=torch.device('cpu'))
        z = torch.randn(4, 64)
        labels = torch.tensor([0, 1, 2, 0])
        out_cond = dec.decode(z, labels)
        out_uncond = dec.decode(z, None)
        assert out_cond.shape == (4, 3, 32, 32)
        assert out_uncond.shape == (4, 3, 32, 32)
        assert not torch.allclose(out_cond, out_uncond), \
            "Class-conditional output must differ from unconditional"

    def test_null_labels_do_not_crash(self):
        """Decoder must handle labels=None gracefully (uses zero embedding)."""
        dec = DCGANDecoder(64, (3, 32, 32), base_channels=64,
                           num_classes=2, device=torch.device('cpu'))
        z = torch.randn(2, 64)
        out = dec.decode(z, None)
        assert out.shape == (2, 3, 32, 32)
        assert not torch.isnan(out).any()

    def test_decode_output_shape_64x64(self):
        """Decoder at 64x64 must produce correct spatial dimensions."""
        dec = DCGANDecoder(128, (3, 64, 64), base_channels=128,
                           device=torch.device('cpu'))
        z = torch.randn(2, 128)
        out = dec.decode(z)
        assert out.shape == (2, 3, 64, 64)


# ───────────────────────────────────────────────────────────────────────
# Power-of-2 Validation
# ───────────────────────────────────────────────────────────────────────

class TestDecoderValidation:

    def test_rejects_non_power_of_two_height(self):
        with pytest.raises(ValueError, match="power of 2"):
            DCGANDecoder(64, (3, 96, 96))

    def test_rejects_non_power_of_two_width(self):
        with pytest.raises(ValueError, match="power of 2"):
            DCGANDecoder(64, (3, 64, 48))

    def test_rejects_too_small_height(self):
        with pytest.raises(ValueError, match="power of 2"):
            DCGANDecoder(64, (3, 4, 4))


# ───────────────────────────────────────────────────────────────────────
# WGAN-GP Discriminator (patent Section 4.5)
# ───────────────────────────────────────────────────────────────────────

class TestDiscriminator:

    @pytest.fixture
    def disc(self):
        return SynthesisPipeline._build_discriminator((3, 32, 32), base_channels=32)

    def test_no_batchnorm(self, disc):
        """Discriminator must not use BatchNorm (WGAN-GP requirement, patent Section 4.5.3)."""
        for module in disc.modules():
            assert not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)), \
                f"Found forbidden {type(module).__name__} in discriminator"

    def test_get_features_returns_list(self, disc):
        """get_features must return a list of intermediate activation tensors."""
        x = torch.randn(4, 3, 32, 32)
        features = disc.get_features(x)
        assert isinstance(features, list)
        assert len(features) > 0
        for f in features:
            assert isinstance(f, torch.Tensor)
            assert f.shape[0] == 4

    def test_scalar_critic_output(self, disc):
        """Discriminator output must be a scalar per sample (no sigmoid)."""
        x = torch.randn(4, 3, 32, 32)
        out = disc(x)
        assert out.shape == (4, 1) or out.ndim == 1 and out.shape[0] == 4

    def test_gradient_penalty_non_negative_and_finite(self, disc):
        """WGAN-GP penalty = E[(||grad||_2 - 1)^2] must be non-negative and finite."""
        real = torch.randn(4, 3, 32, 32)
        fake = torch.randn(4, 3, 32, 32)
        alpha = torch.rand(4, 1, 1, 1)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interp = disc(interp)
        grads = torch.autograd.grad(
            outputs=d_interp.sum(), inputs=interp, create_graph=True
        )[0]
        gp = ((grads.norm(2, dim=(1, 2, 3)) - 1) ** 2).mean()
        assert gp.item() >= 0
        assert torch.isfinite(gp)


# ───────────────────────────────────────────────────────────────────────
# L2 Normalized Encoder Output (patent Section 4.2)
# ───────────────────────────────────────────────────────────────────────

class TestL2NormalizedEncoder:

    def test_unit_norm_embeddings(self):
        """With normalize_output=True, all embeddings must have L2 norm = 1."""
        enc = ResNetEncoder('resnet18', embedding_dim=128, pretrained=False,
                            normalize_output=True, device=torch.device('cpu'))
        imgs = torch.randn(8, 3, 64, 64)
        embs = enc.encode(imgs)
        norms = embs.norm(dim=1)
        torch.testing.assert_close(norms, torch.ones(8), atol=1e-5, rtol=0)

    def test_non_normalized_embeddings_are_not_unit_norm(self):
        """With normalize_output=False, embeddings should NOT all be unit norm."""
        enc = ResNetEncoder('resnet18', embedding_dim=128, pretrained=False,
                            normalize_output=False, device=torch.device('cpu'))
        imgs = torch.randn(8, 3, 64, 64)
        embs = enc.encode(imgs)
        norms = embs.norm(dim=1)
        # Extremely unlikely all 8 random outputs happen to be exactly 1.0
        assert not torch.allclose(norms, torch.ones(8), atol=1e-3)


# ───────────────────────────────────────────────────────────────────────
# Fast Distance Filtering
# ───────────────────────────────────────────────────────────────────────

class TestFastDistanceFiltering:
    def test_filter_by_distance_optimized(self):
        """_filter_by_distance should correctly and efficiently filter embeddings based on NN max distance threshold."""
        np.random.seed(42)
        n_real = 50
        n_syn = 50
        dim = 10
        threshold = 2.0

        real_emb = np.random.randn(n_real, dim)
        real_labels = np.random.randint(0, 3, n_real)

        syn_emb = np.random.randn(n_syn, dim)
        syn_labels = np.random.randint(0, 3, n_syn)

        smote = ConstrainedSMOTE(max_distance_threshold=threshold)
        smote.embeddings = real_emb
        smote.labels = real_labels

        filtered_emb, filtered_labels = smote._filter_by_distance(syn_emb, syn_labels)

        # Verify the length is less than or equal to original due to filtering
        assert len(filtered_emb) <= n_syn
        assert len(filtered_emb) == len(filtered_labels)

        # Ensure all kept samples meet the distance threshold property
        for syn_e, syn_l in zip(filtered_emb, filtered_labels):
            real_class_embs = real_emb[real_labels == syn_l]
            distances = np.linalg.norm(real_class_embs - syn_e, axis=1)
            assert np.min(distances) <= threshold

        # Check that we didn't just filter everything out if threshold is reasonable
        assert len(filtered_emb) > 0


# ───────────────────────────────────────────────────────────────────────
# Pipeline: class-conditional generation end-to-end
# ───────────────────────────────────────────────────────────────────────

class TestPipelineClassConditional:

    def test_generate_uses_class_labels(self):
        """generate_synthetic_images must produce correct class labels when conditioned."""
        enc = ResNetEncoder('resnet18', embedding_dim=64, pretrained=False,
                            normalize_output=True, device=torch.device('cpu'))
        dec = DCGANDecoder(64, (3, 32, 32), base_channels=64, num_classes=2,
                           device=torch.device('cpu'))
        smote = ConstrainedSMOTE(k_neighbors=3, normalize_embeddings=False,
                                 use_slerp=True, random_state=42)
        pipeline = SynthesisPipeline(enc, dec, smote)

        imgs = torch.randn(20, 3, 32, 32)
        labels = np.array([0] * 10 + [1] * 10)
        pipeline.fit(imgs, labels, train_decoder=False)

        syn, syn_labels = pipeline.generate_synthetic_images(n_samples=6)
        assert syn.shape[1:] == torch.Size([3, 32, 32])
        assert len(syn_labels) == 6
        assert set(syn_labels).issubset({0, 1})
