"""
Main pipeline orchestrator for SMOTE-based image synthesis.
"""

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .encoders.base import ImageEncoder
from .decoders.base import BaseDecoder
from .smote.constrained_smote import ConstrainedSMOTE
from .quality.assessor import QualityAssessor


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class _EMA:
    """Exponential Moving Average of model parameters for smoother inference.

    Maintains a shadow copy of every learnable parameter.  After training,
    call ``apply()`` to swap the model to its smoothed version — this
    typically lowers FID and improves visual quality without any extra cost.

    Decay 0.9999 is appropriate for runs with ≥ 1 000 generator steps.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            n: p.data.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.shadow:
                    self.shadow[n].mul_(self.decay).add_(
                        p.data, alpha=1.0 - self.decay
                    )

    def apply(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Swap model parameters to EMA shadow; returns original weights."""
        backup: Dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])
        return backup

    @staticmethod
    def restore(model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SynthesisPipeline:
    """Main pipeline for SMOTE-based synthetic image generation."""

    def __init__(
        self,
        encoder: ImageEncoder,
        decoder: BaseDecoder,
        smote: ConstrainedSMOTE,
        quality_assessor: Optional[QualityAssessor] = None,
    ):
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

        if encoder.get_embedding_dim() != decoder.get_embedding_dim():
            raise ValueError("Encoder and decoder embedding dimensions must match")

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        images: torch.Tensor,
        labels: np.ndarray,
        train_decoder: bool = True,
        decoder_epochs: int = 50,
        start_epoch: int = 0,
        total_epochs: int = 0,
    ) -> None:
        """
        Fit the pipeline on training data.

        Args:
            images: Training images [B, C, H, W]
            labels: Corresponding labels [B]
            train_decoder: Whether to train the decoder
            decoder_epochs: Number of epochs for THIS call
            start_epoch: Global epoch offset (for segmented training / resume)
            total_epochs: Total epochs across all segments (0 = same as decoder_epochs)
        """
        decoder_type = type(self.decoder).__name__
        _total = total_epochs if total_epochs > 0 else (start_epoch + decoder_epochs)

        if train_decoder and decoder_type in ('AutoencoderDecoder', 'DCGANDecoder'):
            labels_tensor = torch.from_numpy(np.asarray(labels)).long()
            self._train_end_to_end(
                images,
                labels=labels_tensor,
                num_epochs=decoder_epochs,
                global_start_epoch=start_epoch,
                global_total_epochs=_total,
            )

        # Generate embeddings (uses the jointly-trained encoder)
        embeddings = self.encoder.encode(images)
        embeddings_np = embeddings.detach().cpu().numpy()

        # Fit SMOTE on embeddings
        self.smote.fit(embeddings_np, labels)

        # Train non-E2E decoders the original way
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

    # ------------------------------------------------------------------
    # End-to-end training (AutoencoderDecoder / DCGANDecoder)
    # ------------------------------------------------------------------

    def _train_end_to_end(
        self,
        images: torch.Tensor,
        num_epochs: int,
        learning_rate: float = 2e-4,
        batch_size: int = 32,
        global_start_epoch: int = 0,
        global_total_epochs: int = 0,
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Train encoder and decoder jointly end-to-end.

        Phase 1 (first 30% of global epochs): reconstruction only (MSE+L1+perceptual)
        Phase 2 (remaining 70%): adds WGAN-GP discriminator for sharpness.

        Improvements applied automatically:
        - Class-conditional generation when decoder.num_classes > 0
        - EMA of decoder weights applied at end for smoother inference
        - Feature matching loss against discriminator intermediate features
        """
        import logging
        from torch.utils.data import DataLoader, TensorDataset
        import torch.optim as optim
        _logger = logging.getLogger(__name__)

        device = self.encoder.device
        use_gan = type(self.decoder).__name__ == 'DCGANDecoder'
        use_cond = getattr(self.decoder, 'num_classes', 0) > 0
        g_total = global_total_epochs if global_total_epochs > 0 else num_epochs

        # Unfreeze ALL encoder params for joint training
        for param in self.encoder.model.parameters():
            param.requires_grad = True

        gen_params = (
            list(self.encoder.model.parameters())
            + list(self.decoder.model.parameters())
        )

        opt_gen = optim.Adam(gen_params, lr=learning_rate, betas=(0.5, 0.999))
        for pg in opt_gen.param_groups:
            pg.setdefault('initial_lr', pg['lr'])
        sched_gen = optim.lr_scheduler.CosineAnnealingLR(
            opt_gen, T_max=g_total, eta_min=1e-5,
            last_epoch=global_start_epoch - 1 if global_start_epoch > 0 else -1,
        )
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

        # Build discriminator for GAN mode
        disc = None
        opt_disc = None
        if use_gan:
            disc = self._build_discriminator(self.decoder.image_shape, base_channels=64).to(device)
            opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))
            _logger.info("  GAN discriminator: enabled (WGAN-GP + feature matching)")

        # EMA of encoder+decoder parameters for smoother inference
        ema_dec = _EMA(self.decoder.model, decay=0.9999)
        ema_enc = _EMA(self.encoder.model, decay=0.9999)
        _logger.info("  EMA: enabled for encoder+decoder (decay=0.9999)")

        # Warmup threshold is global so segments don't restart the recon-only phase
        recon_epochs_global = max(1, int(g_total * 0.3))
        cond_str = f", class-conditional ({self.decoder.num_classes} classes)" if use_cond else ""
        _logger.info(
            f"E2E training: {num_epochs} epochs "
            f"(global {global_start_epoch}–{global_start_epoch + num_epochs - 1} / {g_total})"
            + (f", GAN active after global epoch {recon_epochs_global}" if use_gan else "")
            + cond_str
        )

        # DataLoader — include labels when available
        if labels is not None:
            dataset = TensorDataset(images, labels.to(device))
        else:
            dataset = TensorDataset(images)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.encoder.model.train()
        self.decoder.model.train()
        if disc is not None:
            disc.train()

        n_critic = 5  # D steps per G step (standard WGAN-GP per Gulrajani et al.)

        for epoch in range(num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            global_epoch = global_start_epoch + epoch
            adv_active = use_gan and global_epoch >= recon_epochs_global

            for batch in loader:
                if labels is not None:
                    batch_imgs, batch_labels = batch[0], batch[1]
                else:
                    batch_imgs = batch[0]
                    batch_labels = None

                batch_imgs = batch_imgs.to(device)
                if batch_labels is not None:
                    batch_labels = batch_labels.to(device)
                bs = batch_imgs.size(0)

                # ── Discriminator steps (GAN phase only) ──────────────────────
                if adv_active:
                    for _ in range(n_critic):
                        opt_disc.zero_grad()
                        with torch.no_grad():
                            emb_d = self.encoder.model(batch_imgs)
                            if use_cond and batch_labels is not None:
                                fake = self.decoder.model(emb_d, batch_labels)
                            else:
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
                            create_graph=True,
                        )[0]
                        gp = ((grads.norm(2, dim=(1, 2, 3)) - 1) ** 2).mean()
                        d_loss = -d_real + d_fake + 10.0 * gp
                        d_loss.backward()
                        opt_disc.step()
                        epoch_d_loss += d_loss.item()

                # ── Generator / encoder+decoder step ──────────────────────────
                opt_gen.zero_grad()
                emb = self.encoder.model(batch_imgs)
                if use_cond and batch_labels is not None:
                    recon = self.decoder.model(emb, batch_labels)
                else:
                    recon = self.decoder.model(emb)

                mse = criterion_mse(recon, batch_imgs)
                l1  = criterion_l1(recon, batch_imgs)
                g_loss = mse + 0.5 * l1
                if use_perceptual:
                    g_loss = g_loss + 0.05 * perceptual_loss_fn(recon, batch_imgs)

                if adv_active:
                    # WGAN generator loss: maximise D(fake) = minimise -D(fake)
                    g_adv = -disc(recon).mean()
                    frac = (global_epoch - recon_epochs_global) / max(
                        1, g_total - recon_epochs_global
                    )
                    lam_adv = 0.05 + 0.15 * frac  # 0.05 → 0.20 ramp

                    # Feature matching loss — discriminator intermediate features
                    real_feats = disc.get_features(batch_imgs.detach())
                    fake_feats = disc.get_features(recon)
                    fm_loss = sum(
                        F.l1_loss(f, r.detach())
                        for f, r in zip(fake_feats, real_feats)
                    )
                    g_loss = g_loss + lam_adv * g_adv + 0.1 * fm_loss

                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(gen_params, max_norm=1.0)
                opt_gen.step()
                epoch_g_loss += g_loss.item()

                # Update EMA shadows after every generator step
                ema_dec.update(self.decoder.model)
                ema_enc.update(self.encoder.model)

            sched_gen.step()
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                d_str = f"  D={epoch_d_loss/(len(loader)*n_critic):.4f}" if adv_active else ""
                _logger.info(
                    f"  E2E Epoch {global_epoch:>3}/{g_total}: "
                    f"G={epoch_g_loss/len(loader):.4f}{d_str}"
                    + (" [GAN+FM]" if adv_active else " [recon]")
                )

        self.encoder.model.eval()
        self.decoder.model.eval()
        if disc is not None:
            disc.eval()

        # Apply EMA weights to encoder+decoder — yields smoother, higher-quality inference
        ema_dec.apply(self.decoder.model)
        ema_enc.apply(self.encoder.model)
        _logger.info("E2E training complete — EMA weights applied to encoder+decoder")
        self.decoder._is_trained = True

    # ------------------------------------------------------------------
    # Discriminator builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_discriminator(image_shape: tuple, base_channels: int = 64):
        """Build a WGAN-GP discriminator with feature map extraction.

        No BatchNorm (GP provides Lipschitz regularisation).
        ``get_features()`` exposes intermediate activations for feature matching.
        """
        c, h, _ = image_shape
        feature_layers: list = []
        in_ch = c
        out_ch = base_channels
        cur = h
        while cur > 4:
            feature_layers += [
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)
            cur //= 2
        final_layer = nn.Conv2d(in_ch, 1, 4, 1, 0, bias=True)

        class Discriminator(nn.Module):
            def __init__(self, feat_layers, final):
                super().__init__()
                self.feat_layers = nn.ModuleList(feat_layers)
                self.final = final

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for layer in self.feat_layers:
                    x = layer(x)
                return self.final(x).view(x.size(0))

            def get_features(self, x: torch.Tensor):
                """Return list of post-activation feature maps for feature matching."""
                features = []
                for layer in self.feat_layers:
                    x = layer(x)
                    if isinstance(layer, nn.LeakyReLU):
                        features.append(x)
                return features

        disc = Discriminator(feature_layers, final_layer)
        for m in disc.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        return disc

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate_synthetic_images(
        self, n_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Generate synthetic images.

        Args:
            n_samples: Number of synthetic samples to generate

        Returns:
            Tuple of (synthetic_images, synthetic_labels)
        """
        synthetic_embeddings, synthetic_labels = self.smote.generate_synthetic(n_samples)

        if len(synthetic_embeddings) == 0:
            return torch.empty(0), np.array([])

        synthetic_embeddings_tensor = torch.from_numpy(
            np.array(synthetic_embeddings)
        ).float()

        # Pass class labels to decoder when class conditioning is active
        if getattr(self.decoder, 'num_classes', 0) > 0:
            labels_tensor = torch.from_numpy(
                np.array(synthetic_labels, dtype=np.int64)
            ).long()
            synthetic_images = self.decoder.decode(
                synthetic_embeddings_tensor, labels_tensor
            )
        else:
            synthetic_images = self.decoder.decode(synthetic_embeddings_tensor)

        return synthetic_images, synthetic_labels

    def evaluate_quality(
        self,
        synthetic_images: torch.Tensor,
        real_images: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate quality of synthetic images.

        Returns:
            Flat quality metrics dictionary
        """
        nested = self.quality_assessor.evaluate_quality(synthetic_images, real_images)
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
