"""
Main pipeline orchestrator for SMOTE-based image synthesis.
"""

from typing import Optional, Dict, Any, Tuple
import torch
import numpy as np

from .encoders.base import ImageEncoder
from .decoders.base import BaseDecoder
from .smote.constrained_smote import ConstrainedSMOTE
from .quality.assessor import QualityAssessor


class SynthesisPipeline:
    """Main pipeline for SMOTE-based synthetic image generation."""
    
    def __init__(self,
                 encoder: ImageEncoder,
                 decoder: BaseDecoder,
                 smote: ConstrainedSMOTE,
                 quality_assessor: Optional[QualityAssessor] = None):
        """
        Initialize the synthesis pipeline.
        
        Args:
            encoder: Image encoder for generating embeddings
            decoder: Image decoder for reconstructing images
            smote: SMOTE implementation for embedding oversampling
            quality_assessor: Optional quality assessment module
        """
        self.encoder = encoder
        self.decoder = decoder
        self.smote = smote
        self.quality_assessor = quality_assessor or QualityAssessor()
        
        # Validate compatibility
        if encoder.get_embedding_dim() != decoder.get_embedding_dim():
            raise ValueError("Encoder and decoder embedding dimensions must match")
            
    def fit(self, images: torch.Tensor, labels: np.ndarray,
            train_decoder: bool = True, decoder_epochs: int = 50) -> None:
        """
        Fit the pipeline on training data.

        Args:
            images: Training images [B, C, H, W]
            labels: Corresponding labels [B]
            train_decoder: Whether to train the decoder
            decoder_epochs: Number of epochs for decoder training
        """
        decoder_type = type(self.decoder).__name__

        # For AutoencoderDecoder: train encoder+decoder jointly end-to-end first,
        # then re-extract reconstruction-friendly embeddings for SMOTE.
        if train_decoder and decoder_type in ('AutoencoderDecoder', 'DCGANDecoder'):
            self._train_end_to_end(images, num_epochs=decoder_epochs)

        # Generate embeddings (uses the now jointly-trained encoder if applicable)
        embeddings = self.encoder.encode(images)
        embeddings_np = embeddings.detach().cpu().numpy()

        # Fit SMOTE on embeddings
        self.smote.fit(embeddings_np, labels)

        # Train non-AE decoders the original way
        if train_decoder and decoder_type not in ('AutoencoderDecoder', 'DCGANDecoder'):
            if decoder_type == 'VAEDecoder':
                from .decoders.vae_trainer import VAETrainer
                trainer = VAETrainer(self.decoder, learning_rate=0.001)
            elif decoder_type == 'GANDecoder':
                from .decoders.gan_trainer import GANTrainer
                trainer = GANTrainer(self.decoder, learning_rate=0.001)
            elif decoder_type == 'DiffusionDecoder':
                from .decoders.diffusion_trainer import DiffusionTrainer
                trainer = DiffusionTrainer(self.decoder, learning_rate=0.001)
            else:
                from .decoders.autoencoder_trainer import AutoencoderTrainer
                trainer = AutoencoderTrainer(self.decoder, learning_rate=0.001)
            trainer.train(embeddings, images, num_epochs=decoder_epochs, batch_size=16)
            self.decoder._is_trained = True

    def _train_end_to_end(self, images: torch.Tensor, num_epochs: int,
                          learning_rate: float = 2e-4, batch_size: int = 32) -> None:
        """
        Train encoder and decoder jointly end-to-end.
        For DCGANDecoder: uses hybrid reconstruction + adversarial training.
        Phase 1 (first 30% of epochs): reconstruction only (MSE+L1+perceptual)
        Phase 2 (remaining 70%): adds GAN discriminator for sharpness.
        """
        import logging
        from torch.utils.data import DataLoader, TensorDataset
        import torch.optim as optim
        import torch.nn as nn
        _logger = logging.getLogger(__name__)

        device = self.encoder.device
        use_gan = type(self.decoder).__name__ == 'DCGANDecoder'

        # Unfreeze ALL encoder params for joint training
        for param in self.encoder.model.parameters():
            param.requires_grad = True

        gen_params = (list(self.encoder.model.parameters()) +
                      list(self.decoder.model.parameters()))

        opt_gen = optim.Adam(gen_params, lr=learning_rate, betas=(0.5, 0.999))
        sched_gen = optim.lr_scheduler.CosineAnnealingLR(
            opt_gen, T_max=num_epochs, eta_min=1e-5)
        criterion_mse = nn.MSELoss()
        criterion_l1 = nn.L1Loss()

        # Perceptual loss (VGG-based)
        try:
            from .decoders.autoencoder_trainer import PerceptualLoss
            perceptual_loss_fn = PerceptualLoss(device=device)
            use_perceptual = True
            _logger.info("  Perceptual loss: enabled")
        except Exception as e:
            use_perceptual = False
            _logger.info(f"  Perceptual loss: unavailable ({e}), using MSE+L1 only")

        # Build discriminator if using GAN mode
        disc = None
        opt_disc = None
        if use_gan:
            disc = self._build_discriminator(self.decoder.image_shape, base_channels=64).to(device)
            opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))
            _logger.info("  GAN discriminator: enabled (adversarial training)")

        recon_epochs = max(1, int(num_epochs * 0.3))  # warmup reconstruction only
        _logger.info(
            f"E2E training: encoder+decoder jointly for {num_epochs} epochs "
            f"({'GAN after epoch ' + str(recon_epochs) if use_gan else 'recon only'})"
        )

        dataset = TensorDataset(images)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.encoder.model.train()
        self.decoder.model.train()
        if disc is not None:
            disc.train()

        n_critic = 2  # D steps per G step in GAN phase

        for epoch in range(num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            adv_active = use_gan and epoch >= recon_epochs

            for (batch_imgs,) in loader:
                batch_imgs = batch_imgs.to(device)
                bs = batch_imgs.size(0)

                # ── Discriminator steps (GAN phase only) ──────────────────────
                if adv_active:
                    for _ in range(n_critic):
                        opt_disc.zero_grad()
                        with torch.no_grad():
                            emb_d = self.encoder.model(batch_imgs)
                            fake = self.decoder.model(emb_d)
                        # WGAN-GP: W-distance = E[D(real)] - E[D(fake)]
                        d_real = disc(batch_imgs).mean()
                        d_fake = disc(fake).mean()
                        # Gradient penalty
                        alpha = torch.rand(bs, 1, 1, 1, device=device)
                        interp = (alpha * batch_imgs + (1 - alpha) * fake).requires_grad_(True)
                        d_interp = disc(interp)
                        grads = torch.autograd.grad(
                            outputs=d_interp.sum(), inputs=interp,
                            create_graph=True)[0]
                        gp = ((grads.norm(2, dim=(1, 2, 3)) - 1) ** 2).mean()
                        d_loss = -d_real + d_fake + 10.0 * gp
                        d_loss.backward()
                        opt_disc.step()
                    epoch_d_loss += d_loss.item()

                # ── Generator / encoder+decoder step ──────────────────────────
                opt_gen.zero_grad()
                emb = self.encoder.model(batch_imgs)
                recon = self.decoder.model(emb)

                mse = criterion_mse(recon, batch_imgs)
                l1  = criterion_l1(recon, batch_imgs)
                g_loss = mse + 0.5 * l1
                if use_perceptual:
                    g_loss = g_loss + 0.05 * perceptual_loss_fn(recon, batch_imgs)

                if adv_active:
                    # WGAN generator loss: maximise D(fake) = minimise -D(fake)
                    g_adv = -disc(recon).mean()
                    frac = (epoch - recon_epochs) / max(1, num_epochs - recon_epochs)
                    lam_adv = 0.05 + 0.15 * frac  # 0.05 → 0.20
                    g_loss = g_loss + lam_adv * g_adv

                torch.nn.utils.clip_grad_norm_(gen_params, max_norm=1.0)
                g_loss.backward()
                opt_gen.step()
                epoch_g_loss += g_loss.item()

            sched_gen.step()
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                d_str = f"  D={epoch_d_loss/len(loader):.4f}" if adv_active else ""
                _logger.info(
                    f"  E2E Epoch {epoch:>3}: G={epoch_g_loss/len(loader):.4f}{d_str}"
                )

        self.encoder.model.eval()
        self.decoder.model.eval()
        if disc is not None:
            disc.eval()
        self.decoder._is_trained = True
        _logger.info("E2E training complete")

    @staticmethod
    def _build_discriminator(image_shape: tuple, base_channels: int = 64):
        """Build a WGAN-GP discriminator (no BN, no spectral norm — GP does the job)."""
        import torch.nn as nn

        c, h, _ = image_shape
        layers = []
        in_ch = c
        out_ch = base_channels
        cur = h
        while cur > 4:
            layers += [
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)
            cur //= 2
        # Final conv → scalar critic score per image
        layers.append(nn.Conv2d(in_ch, 1, 4, 1, 0, bias=True))

        class Discriminator(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.main = nn.Sequential(*layers)
            def forward(self, x):
                return self.main(x).view(x.size(0))

        disc = Discriminator(layers)
        for m in disc.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        return disc
        
    def generate_synthetic_images(self, 
                                n_samples: Optional[int] = None) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Generate synthetic images.
        
        Args:
            n_samples: Number of synthetic samples to generate
            
        Returns:
            Tuple of (synthetic_images, synthetic_labels)
        """
        # Generate synthetic embeddings
        synthetic_embeddings, synthetic_labels = self.smote.generate_synthetic(n_samples)
        
        if len(synthetic_embeddings) == 0:
            return torch.empty(0), np.array([])
            
        # Convert to tensor
        synthetic_embeddings_tensor = torch.from_numpy(synthetic_embeddings).float()
        
        # Decode to images
        synthetic_images = self.decoder.decode(synthetic_embeddings_tensor)
        
        return synthetic_images, synthetic_labels
        
    def evaluate_quality(self,
                        synthetic_images: torch.Tensor,
                        real_images: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate quality of synthetic images.

        Args:
            synthetic_images: Generated images
            real_images: Real reference images

        Returns:
            Flat quality metrics dictionary
        """
        nested = self.quality_assessor.evaluate_quality(synthetic_images, real_images)
        # Flatten nested structure so callers can do results['mse'] directly
        flat: Dict[str, Any] = {}
        for key, value in nested.items():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value
        return flat
        
    def save_pipeline(self, base_path: str) -> None:
        """Save the entire pipeline."""
        self.encoder.save_model(f"{base_path}_encoder.pth")
        self.decoder.save_model(f"{base_path}_decoder.pth")
        
    def load_pipeline(self, base_path: str) -> None:
        """Load the entire pipeline."""
        self.encoder.load_model(f"{base_path}_encoder.pth")
        self.decoder.load_model(f"{base_path}_decoder.pth")