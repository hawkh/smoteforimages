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
        lambda_repulse: float = 0.01,
        repulse_margin: float = 0.3,
    ) -> None:
        """
        Train encoder and decoder jointly end-to-end.

        Phase 1 (first 30% of global epochs): reconstruction only (MSE+L1+perceptual)
        Phase 2 (remaining 70%): WGAN-GP discriminator + feature matching.

        Improvements:
        - Class-conditional generation when decoder.num_classes > 0
        - Projection discriminator (Miyato & Koyama 2018) for class-conditional scoring
        - Spectral normalisation on all discriminator conv layers (SN + GP hybrid)
        - Multi-scale feature matching at 3 discriminator depths [0.1, 0.3, 0.6]
        - Intra-class diversity repulsion loss to prevent per-class mode collapse
        - Adaptive λ_adv via EMA-smoothed Wasserstein distance monitoring
        - EMA of encoder+decoder applied at end for smoother inference
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

        # Build discriminator with SN + optional projection head
        disc = None
        opt_disc = None
        if use_gan:
            n_classes_disc = getattr(self.decoder, 'num_classes', 0) if use_cond else 0
            disc = self._build_discriminator(
                self.decoder.image_shape,
                base_channels=64,
                num_classes=n_classes_disc,
            ).to(device)
            opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))
            cond_disc_str = f" + projection discriminator ({n_classes_disc} classes)" if n_classes_disc > 0 else ""
            _logger.info(
                f"  GAN discriminator: enabled (SN + WGAN-GP + multi-scale FM{cond_disc_str})"
            )

        # EMA of encoder+decoder parameters for smoother inference
        ema_dec = _EMA(self.decoder.model, decay=0.9999)
        ema_enc = _EMA(self.encoder.model, decay=0.9999)
        _logger.info("  EMA: enabled for encoder+decoder (decay=0.9999)")

        recon_epochs_global = max(1, int(g_total * 0.3))
        cond_str = f", class-conditional ({self.decoder.num_classes} classes)" if use_cond else ""
        _logger.info(
            f"E2E training: {num_epochs} epochs "
            f"(global {global_start_epoch}–{global_start_epoch + num_epochs - 1} / {g_total})"
            + (f", GAN active after global epoch {recon_epochs_global}" if use_gan else "")
            + cond_str
        )

        dataset = TensorDataset(images, labels.to(device)) if labels is not None else TensorDataset(images)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.encoder.model.train()
        self.decoder.model.train()
        if disc is not None:
            disc.train()

        n_critic = 5  # D steps per G step (standard WGAN-GP)

        # Adaptive λ_adv state
        W_ema = 0.0
        W_history: list = []
        lam_adv_current = 0.05
        W_window = 10
        FM_WEIGHTS = [0.1, 0.3, 0.6]

        for epoch in range(num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_W_sum = 0.0
            n_W_steps = 0
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

                        # Projection discriminator: conditional score
                        d_real = disc(batch_imgs, batch_labels).mean()
                        d_fake = disc(fake, batch_labels).mean()

                        # WGAN-GP gradient penalty
                        alpha = torch.rand(bs, 1, 1, 1, device=device)
                        interp = (alpha * batch_imgs + (1 - alpha) * fake).requires_grad_(True)
                        d_interp = disc(interp, batch_labels)
                        grads = torch.autograd.grad(
                            outputs=d_interp.sum(), inputs=interp,
                            create_graph=True,
                        )[0]
                        gp = ((grads.norm(2, dim=(1, 2, 3)) - 1) ** 2).mean()
                        d_loss = -d_real + d_fake + 10.0 * gp
                        d_loss.backward()
                        opt_disc.step()
                        epoch_d_loss += d_loss.item()

                        # Accumulate Wasserstein estimate for adaptive λ_adv
                        epoch_W_sum += float(d_real.item() - d_fake.item())
                        n_W_steps += 1

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
                    # WGAN generator loss
                    g_adv = -disc(recon, batch_labels).mean()

                    # Multi-scale feature matching: 3 discriminator depths
                    real_feats = disc.get_features(batch_imgs.detach())
                    fake_feats = disc.get_features(recon)
                    n_scales = len(real_feats)
                    weights = FM_WEIGHTS[:n_scales] if n_scales <= 3 else [1.0 / n_scales] * n_scales
                    # Normalise weights to sum to 1
                    w_sum = sum(weights)
                    fm_loss = sum(
                        (w / w_sum) * F.l1_loss(f, r.detach())
                        for w, f, r in zip(weights, fake_feats, real_feats)
                    )

                    g_loss = g_loss + lam_adv_current * g_adv + 0.1 * fm_loss

                    # Intra-class repulsion: prevent per-class mode collapse
                    if lambda_repulse > 0 and batch_labels is not None:
                        repulse = self._compute_repulsion_loss(
                            emb, batch_labels, margin=repulse_margin
                        )
                        g_loss = g_loss + lambda_repulse * repulse

                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(gen_params, max_norm=1.0)
                opt_gen.step()
                epoch_g_loss += g_loss.item()

                ema_dec.update(self.decoder.model)
                ema_enc.update(self.encoder.model)

            sched_gen.step()

            # ── Adaptive λ_adv update (end of epoch) ──────────────────────────
            if adv_active and n_W_steps > 0:
                W_current = epoch_W_sum / n_W_steps
                W_ema = 0.99 * W_ema + 0.01 * W_current
                W_history.append(W_ema)
                if len(W_history) >= W_window:
                    dW = W_history[-1] - W_history[-W_window]
                    if dW < -0.01:  # W-distance dropping — GAN improving
                        lam_adv_current = min(lam_adv_current + 0.005, 0.50)
                    elif dW > 0.01:  # W-distance rising — GAN struggling
                        lam_adv_current = max(lam_adv_current - 0.005, 0.01)
                else:
                    # Linear ramp during initial GAN warmup
                    frac = (global_epoch - recon_epochs_global) / max(
                        1, g_total - recon_epochs_global
                    )
                    lam_adv_current = 0.05 + 0.15 * frac

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                n_batches = max(len(loader), 1)
                d_str = (
                    f"  D={epoch_d_loss/(n_batches*n_critic):.4f}"
                    f"  λ_adv={lam_adv_current:.3f}"
                ) if adv_active else ""
                _logger.info(
                    f"  E2E Epoch {global_epoch:>3}/{g_total}: "
                    f"G={epoch_g_loss/n_batches:.4f}{d_str}"
                    + (" [GAN+FM+repulse]" if adv_active else " [recon]")
                )

        self.encoder.model.eval()
        self.decoder.model.eval()
        if disc is not None:
            disc.eval()

        ema_dec.apply(self.decoder.model)
        ema_enc.apply(self.encoder.model)
        _logger.info("E2E training complete — EMA weights applied to encoder+decoder")
        self.decoder._is_trained = True

    # ------------------------------------------------------------------
    # Discriminator builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_discriminator(
        image_shape: tuple,
        base_channels: int = 64,
        num_classes: int = 0,
    ) -> nn.Module:
        """Build a WGAN-GP discriminator with spectral normalisation and optional
        class-conditional projection head.

        Design:
        - Spectral normalisation (SN) on every Conv2d — per-layer Lipschitz-1
          constraint via power iteration, complementing the global WGAN-GP penalty
        - Projection discriminator (Miyato & Koyama 2018) when num_classes > 0:
          score = phi(x) + <V·y, GAP(phi(x))>  where V = class embedding table
        - Feature extraction at 3 discriminator depths for multi-scale feature matching
        - No BatchNorm (GP provides Lipschitz regularisation; SN adds per-layer stability)
        """
        c, h, _ = image_shape
        feature_layers: list = []
        in_ch = c
        out_ch = base_channels
        cur = h
        # Track which LeakyReLU indices mark the 3 feature-extraction checkpoints
        lrelu_indices: list = []
        lrelu_count = 0

        while cur > 4:
            feature_layers += [
                torch.nn.utils.spectral_norm(
                    nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            lrelu_indices.append(lrelu_count)
            lrelu_count += 1
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)
            cur //= 2

        final_layer = torch.nn.utils.spectral_norm(
            nn.Conv2d(in_ch, 1, 4, 1, 0, bias=True)
        )
        penultimate_channels = in_ch  # channels entering final_layer

        # Pick 3 evenly-spaced LReLU checkpoints for multi-scale feature matching
        n_lrelu = len(lrelu_indices)
        if n_lrelu >= 3:
            extract_at = {lrelu_indices[0], lrelu_indices[n_lrelu // 2], lrelu_indices[-1]}
        elif n_lrelu == 2:
            extract_at = {lrelu_indices[0], lrelu_indices[1]}
        else:
            extract_at = set(lrelu_indices)

        _num_classes = num_classes  # captured for closure

        class Discriminator(nn.Module):
            def __init__(self, feat_layers, final, n_classes, penult_ch, feat_idx):
                super().__init__()
                self.feat_layers = nn.ModuleList(feat_layers)
                self.final = final
                self.extract_at = feat_idx

                # Projection discriminator class embedding
                if n_classes > 0:
                    self.class_embed = nn.Embedding(n_classes, penult_ch)
                    nn.init.normal_(self.class_embed.weight, 0.0, 0.02)
                else:
                    self.class_embed = None

            def _penultimate(self, x: torch.Tensor) -> torch.Tensor:
                lrelu_i = 0
                for layer in self.feat_layers:
                    x = layer(x)
                    if isinstance(layer, nn.LeakyReLU):
                        lrelu_i += 1
                return x

            def forward(
                self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
                feat = self._penultimate(x)
                score = self.final(feat).view(feat.size(0))

                # Projection: score += <class_embed(y), GAP(feat)>
                if self.class_embed is not None and labels is not None:
                    gap = F.adaptive_avg_pool2d(feat, 1).view(feat.size(0), -1)
                    class_bias = (self.class_embed(labels) * gap).sum(dim=1)
                    score = score + class_bias

                return score

            def get_features(self, x: torch.Tensor):
                """Return feature maps at 3 evenly-spaced discriminator depths.

                Used for multi-scale feature matching (early/mid/late activations).
                Weights in the calling code: [0.1, 0.3, 0.6] (texture → semantics).
                """
                features = []
                lrelu_i = 0
                for layer in self.feat_layers:
                    x = layer(x)
                    if isinstance(layer, nn.LeakyReLU):
                        if lrelu_i in self.extract_at:
                            features.append(x)
                        lrelu_i += 1
                return features

        disc = Discriminator(
            feature_layers, final_layer, _num_classes, penultimate_channels, extract_at
        )

        # Initialise weights — SN wraps weight as weight_orig, initialize that
        for m in disc.modules():
            if hasattr(m, 'weight_orig'):  # spectral_norm-wrapped
                nn.init.normal_(m.weight_orig, 0.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        return disc

    # ------------------------------------------------------------------
    # Repulsion loss
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_repulsion_loss(
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 0.3,
    ) -> torch.Tensor:
        """Intra-class embedding repulsion to prevent per-class mode collapse.

        For each class in the batch, computes pairwise L2 distances and penalises
        embedding pairs that are closer than ``margin``:
            L_repulse = mean(max(0, margin - d_ij)²)  for same-class pairs (i,j)

        Only upper-triangle pairs are used to avoid double-counting.
        Applied during Phase 2 (GAN phase) only, weighted by lambda_repulse.
        """
        unique_labels = torch.unique(labels)
        total_repulsion = embeddings.new_zeros(1)
        n_pairs = 0

        for lbl in unique_labels:
            class_embs = embeddings[labels == lbl]  # [K, D]
            if class_embs.size(0) < 2:
                continue
            # Pairwise distances
            diff = class_embs.unsqueeze(0) - class_embs.unsqueeze(1)  # [K, K, D]
            dists = diff.norm(dim=-1)  # [K, K]
            # Upper triangle (exclude self-pairs)
            mask_upper = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
            pair_dists = dists[mask_upper]
            violations = F.relu(margin - pair_dists)
            total_repulsion = total_repulsion + (violations ** 2).sum()
            n_pairs += len(pair_dists)

        if n_pairs == 0:
            return total_repulsion.squeeze()
        return (total_repulsion / n_pairs).squeeze()

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
