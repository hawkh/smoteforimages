"""Training runner with epoch-level progress callbacks.

Replicates the training loop from pipeline.py but injects a callback
at each epoch boundary so progress can be streamed via WebSocket.
"""

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class _EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        model.load_state_dict(self.shadow)


class TrainingRunner:
    """Runs the end-to-end training loop with progress callbacks."""

    def __init__(self, pipeline, on_epoch: Callable[[dict], None]):
        """
        Args:
            pipeline: SynthesisPipeline instance with encoder, decoder, smote.
            on_epoch: Called after each epoch with a dict of metrics.
        """
        self.pipeline = pipeline
        self.on_epoch = on_epoch
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 2e-4,
        start_epoch: int = 0,
    ):
        """Blocking training loop — run in a thread.

        Adapted from pipeline.py:_train_end_to_end with callback injection.
        """
        import torch.optim as optim

        device = self.pipeline.encoder.device
        encoder = self.pipeline.encoder
        decoder = self.pipeline.decoder
        use_gan = type(decoder).__name__ == "DCGANDecoder"
        use_cond = getattr(decoder, "num_classes", 0) > 0
        total_epochs = epochs

        # Unfreeze encoder
        for param in encoder.model.parameters():
            param.requires_grad = True

        gen_params = list(encoder.model.parameters()) + list(decoder.model.parameters())
        opt_gen = optim.Adam(gen_params, lr=learning_rate, betas=(0.5, 0.999))
        for pg in opt_gen.param_groups:
            pg.setdefault("initial_lr", pg["lr"])
        sched_gen = optim.lr_scheduler.CosineAnnealingLR(
            opt_gen, T_max=total_epochs, eta_min=1e-5,
            last_epoch=start_epoch - 1 if start_epoch > 0 else -1,
        )

        criterion_mse = nn.MSELoss()
        criterion_l1 = nn.L1Loss()

        # Perceptual loss
        use_perceptual = False
        perceptual_loss_fn = None
        try:
            from smote_image_synthesis.decoders.autoencoder_trainer import PerceptualLoss
            perceptual_loss_fn = PerceptualLoss(device=device)
            use_perceptual = True
        except Exception:
            pass

        # Build discriminator
        disc = None
        opt_disc = None
        if use_gan:
            n_classes_disc = getattr(decoder, "num_classes", 0) if use_cond else 0
            disc = self.pipeline._build_discriminator(
                decoder.image_shape,
                base_channels=64,
                num_classes=n_classes_disc,
            ).to(device)
            opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        # EMA
        ema_dec = _EMA(decoder.model, decay=0.9999)
        ema_enc = _EMA(encoder.model, decay=0.9999)

        recon_epochs = max(1, int(total_epochs * 0.3))
        n_critic = 5

        # Adaptive lambda_adv
        W_ema = 0.0
        W_history = []
        lam_adv = 0.05
        FM_WEIGHTS = [0.1, 0.3, 0.6]

        dataset = TensorDataset(images, labels.to(device))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        encoder.model.train()
        decoder.model.train()
        if disc is not None:
            disc.train()

        start_time = time.time()

        try:
            for epoch in range(epochs):
                if self._stop_requested:
                    self.on_epoch({
                        "type": "error",
                        "data": {"message": "Training stopped by user"},
                    })
                    break

                global_epoch = start_epoch + epoch
                adv_active = use_gan and global_epoch >= recon_epochs
                epoch_g_loss = 0.0
                epoch_d_loss = 0.0
                epoch_W_sum = 0.0
                n_W_steps = 0

                for batch in loader:
                    batch_imgs, batch_labels = batch[0].to(device), batch[1].to(device)
                    bs = batch_imgs.size(0)

                    # Discriminator steps
                    if adv_active and disc is not None:
                        for _ in range(n_critic):
                            opt_disc.zero_grad()
                            with torch.no_grad():
                                emb_d = encoder.model(batch_imgs)
                                if use_cond:
                                    fake = decoder.model(emb_d, batch_labels)
                                else:
                                    fake = decoder.model(emb_d)

                            d_real = disc(batch_imgs, batch_labels).mean()
                            d_fake = disc(fake, batch_labels).mean()

                            alpha = torch.rand(bs, 1, 1, 1, device=device)
                            interp = (alpha * batch_imgs + (1 - alpha) * fake).requires_grad_(True)
                            d_interp = disc(interp, batch_labels)
                            grads = torch.autograd.grad(
                                outputs=d_interp.sum(), inputs=interp, create_graph=True,
                            )[0]
                            gp = ((grads.norm(2, dim=(1, 2, 3)) - 1) ** 2).mean()
                            d_loss = -d_real + d_fake + 10.0 * gp
                            d_loss.backward()
                            opt_disc.step()
                            epoch_d_loss += d_loss.item()

                            epoch_W_sum += float(d_real.item() - d_fake.item())
                            n_W_steps += 1

                    # Generator step
                    opt_gen.zero_grad()
                    emb = encoder.model(batch_imgs)
                    if use_cond:
                        recon = decoder.model(emb, batch_labels)
                    else:
                        recon = decoder.model(emb)

                    g_loss = criterion_mse(recon, batch_imgs) + 0.5 * criterion_l1(recon, batch_imgs)
                    if use_perceptual:
                        g_loss = g_loss + 0.05 * perceptual_loss_fn(recon, batch_imgs)

                    if adv_active and disc is not None:
                        g_adv = -disc(recon, batch_labels).mean()
                        real_feats = disc.get_features(batch_imgs.detach())
                        fake_feats = disc.get_features(recon)
                        n_scales = len(real_feats)
                        weights = FM_WEIGHTS[:n_scales]
                        w_sum = sum(weights)
                        fm_loss = sum(
                            (w / w_sum) * F.l1_loss(f, r.detach())
                            for w, f, r in zip(weights, fake_feats, real_feats)
                        )
                        g_loss = g_loss + lam_adv * g_adv + 0.1 * fm_loss

                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(gen_params, max_norm=1.0)
                    opt_gen.step()
                    epoch_g_loss += g_loss.item()

                    ema_dec.update(decoder.model)
                    ema_enc.update(encoder.model)

                sched_gen.step()

                # Adaptive lambda_adv
                if adv_active and n_W_steps > 0:
                    W_current = epoch_W_sum / n_W_steps
                    W_ema = 0.99 * W_ema + 0.01 * W_current
                    W_history.append(W_ema)
                    if len(W_history) >= 10:
                        dW = W_history[-1] - W_history[-10]
                        if dW < -0.01:
                            lam_adv = min(lam_adv + 0.005, 0.50)
                        elif dW > 0.01:
                            lam_adv = max(lam_adv - 0.005, 0.01)
                    else:
                        frac = (global_epoch - recon_epochs) / max(1, total_epochs - recon_epochs)
                        lam_adv = 0.05 + 0.15 * frac

                # Emit epoch progress
                n_batches = max(len(loader), 1)
                phase_changed = (adv_active and global_epoch == recon_epochs)

                if phase_changed:
                    self.on_epoch({
                        "type": "phase_change",
                        "data": {
                            "from_phase": "recon",
                            "to_phase": "gan",
                            "at_epoch": global_epoch,
                        },
                    })

                self.on_epoch({
                    "type": "epoch",
                    "data": {
                        "epoch": global_epoch,
                        "total_epochs": total_epochs,
                        "phase": "gan" if adv_active else "recon",
                        "g_loss": round(epoch_g_loss / n_batches, 6),
                        "d_loss": round(epoch_d_loss / (n_batches * n_critic), 6) if adv_active else None,
                        "lambda_adv": round(lam_adv, 4),
                        "w_distance": round(W_ema, 6) if adv_active else None,
                        "lr": float(sched_gen.get_last_lr()[0]),
                        "ema_decay": 0.9999,
                        "elapsed_seconds": round(time.time() - start_time, 1),
                    },
                })

            # Apply EMA and finalize
            encoder.model.eval()
            decoder.model.eval()
            ema_dec.apply(decoder.model)
            ema_enc.apply(encoder.model)

            if not self._stop_requested:
                self.on_epoch({
                    "type": "complete",
                    "data": {
                        "total_epochs": total_epochs,
                        "final_g_loss": round(epoch_g_loss / max(len(loader), 1), 6),
                        "ema_applied": True,
                        "elapsed_total_seconds": round(time.time() - start_time, 1),
                    },
                })

        except Exception as e:
            logger.exception("Training error")
            self.on_epoch({
                "type": "error",
                "data": {"message": str(e)},
            })
