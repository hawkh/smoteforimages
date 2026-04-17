#!/usr/bin/env python3
"""
Frontier improvements test on CIFAR-10 cats & dogs.

Trains with all new improvements enabled, generates real vs synthetic
comparison grids, and loops until quality plateaus or a PSNR threshold is met.

Rounds:
  Round 1 — 80 epochs (cold start)
  Round 2 — +40 epochs if PSNR < 19 dB  (GAN warmup extension)
  Round 3 — +40 epochs if PSNR still < 20 dB (final refinement)

Outputs in ./frontier_output/:
  real_samples.png              — 25 real cats/dogs
  comparison_round_N.png        — real | SLERP-synthetic | vMF-synthetic
  metrics.json                  — per-round quality metrics
  frontier_pipeline_*           — saved weights
"""

import json
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder
from smote_image_synthesis.decoders.dcgan_decoder import DCGANDecoder
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE
from smote_image_synthesis.quality.assessor import QualityAssessor
from smote_image_synthesis.pipeline import SynthesisPipeline

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('frontier_test')

OUTPUT_DIR   = Path('frontier_output')
DATA_DIR     = Path('data')
IMAGE_SIZE   = 32
N_PER_CLASS  = 300
EMB_DIM      = 512
BASE_CHANNELS = 256      # 256 converges faster than 512 on small datasets
CLASS_NAMES  = {0: 'cat', 1: 'dog'}
CIFAR_CAT, CIFAR_DOG = 3, 5

# Quality thresholds for loop control (use G_loss, not PSNR)
GLOSS_TARGET    = 0.10   # G_loss below this = good convergence
MIN_IMPROVEMENT = 0.03   # G_loss improvement per round to keep going


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cats_dogs(n_per_class: int) -> tuple:
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    logger.info(f"Loading CIFAR-10 (train+test, {n_per_class} per class)...")
    imgs_l, lbl_l = [], []
    counts = {0: 0, 1: 0}
    for split in (True, False):
        ds = CIFAR10(str(DATA_DIR), train=split, download=True, transform=transform)
        for img, lbl in ds:
            if lbl == CIFAR_CAT and counts[0] < n_per_class:
                imgs_l.append(img); lbl_l.append(0); counts[0] += 1
            elif lbl == CIFAR_DOG and counts[1] < n_per_class:
                imgs_l.append(img); lbl_l.append(1); counts[1] += 1
            if counts[0] >= n_per_class and counts[1] >= n_per_class:
                break
    images = torch.stack(imgs_l)
    labels = np.array(lbl_l)
    logger.info(f"  {counts[0]} cats + {counts[1]} dogs loaded")
    return images, labels


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def denorm(t: torch.Tensor) -> np.ndarray:
    """[-1,1] tensor → [0,1] HWC numpy."""
    return ((t.clamp(-1, 1) + 1) / 2).cpu().permute(1, 2, 0).numpy().clip(0, 1)


def save_real_grid(images, labels, path):
    n = min(25, len(images))
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2 + 0.4))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(denorm(images[i]))
            ax.set_title(CLASS_NAMES[int(labels[i])], fontsize=7)
        ax.axis('off')
    fig.suptitle(f'Real CIFAR-10 Cats & Dogs (n={n})', fontsize=11, fontweight='bold')
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved → {path}")


def save_comparison(real, real_lbl, slerp_syn, slerp_lbl,
                    vmf_syn, vmf_lbl, round_num, metrics, path):
    """3-row comparison: real | SLERP synthetic | vMF synthetic."""
    n = min(10, len(real), len(slerp_syn), len(vmf_syn))
    fig = plt.figure(figsize=(n * 2.0, 7.5))
    gs  = gridspec.GridSpec(3, n, hspace=0.05, wspace=0.04,
                            top=0.88, bottom=0.02, left=0.01, right=0.99)

    row_titles = ['REAL', 'SLERP-SMOTE synthetic', 'vMF-SMOTE synthetic']
    row_colors = ['#1a6b3c', '#1a4b8c', '#7c1a6b']

    for row_idx, (imgs, labs, row_title, rc) in enumerate([
        (real,      real_lbl,  row_titles[0], row_colors[0]),
        (slerp_syn, slerp_lbl, row_titles[1], row_colors[1]),
        (vmf_syn,   vmf_lbl,   row_titles[2], row_colors[2]),
    ]):
        for col in range(n):
            ax = fig.add_subplot(gs[row_idx, col])
            if col < len(imgs):
                ax.imshow(denorm(imgs[col]))
                ax.set_title(CLASS_NAMES.get(int(labs[col]), '?'), fontsize=6.5)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(row_title, fontsize=8, color=rc, fontweight='bold',
                              rotation=90, labelpad=4)
                ax.yaxis.set_label_coords(-0.35, 0.5)
                ax.axis('on')
                ax.set_yticks([])
                ax.set_xticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

    psnr = metrics.get('psnr', float('nan'))
    mse  = metrics.get('mse',  float('nan'))
    fig.suptitle(
        f'Round {round_num} — Real vs Synthetic  |  '
        f'Recon PSNR: {psnr:.2f} dB  |  MSE: {mse:.4f}',
        fontsize=11, fontweight='bold', y=0.95,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Pipeline builder (all frontier improvements ON)
# ---------------------------------------------------------------------------

def build_pipeline(device):
    encoder = ResNetEncoder(
        architecture='resnet18',
        embedding_dim=EMB_DIM,
        pretrained=True,
        device=device,
        freeze_backbone=False,
        normalize_output=True,  # L2-norm → unit hypersphere
    )
    decoder = DCGANDecoder(
        embedding_dim=EMB_DIM,
        image_shape=(3, IMAGE_SIZE, IMAGE_SIZE),
        base_channels=BASE_CHANNELS,  # 256 converges 2x faster on small sets
        num_classes=2,                # cats (0) / dogs (1)
        class_embed_dim=64,
        use_self_attention=True,      # SAGAN attention at 16×16
        device=device,
    )
    # ── SLERP-SMOTE with all frontier improvements ──────────────────────────
    smote_slerp = ConstrainedSMOTE(
        k_neighbors=5,
        use_clustering=True,
        normalize_embeddings=False,   # Encoder already normalises
        use_slerp=True,
        density_weighted_t=True,      # Bias t to fill sparse geodesic gaps
        use_cluster_constraints=True, # SLERP within-cluster only
        use_outlier_detection=True,   # Filter outlier synthetics
        boundary_detection_method='isolation',
        track_ancestry=True,          # Record parent lineage
        random_state=42,
    )
    quality_assessor = QualityAssessor(metrics=['mse', 'psnr'], compute_diversity=True)
    pipeline = SynthesisPipeline(
        encoder=encoder, decoder=decoder,
        smote=smote_slerp, quality_assessor=quality_assessor,
    )
    return pipeline


# ---------------------------------------------------------------------------
# Patched pipeline training that logs G_loss per epoch for loop control
# ---------------------------------------------------------------------------

class InstrumentedPipeline(SynthesisPipeline):
    """Thin wrapper that captures per-epoch G_loss for convergence monitoring."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.g_loss_history: list = []

    def _train_end_to_end(self, images, num_epochs, learning_rate=2e-4,
                          batch_size=32, global_start_epoch=0,
                          global_total_epochs=0, labels=None,
                          lambda_repulse=0.01, repulse_margin=0.3):
        import logging
        from torch.utils.data import DataLoader, TensorDataset
        import torch.optim as optim
        _logger = logging.getLogger(__name__)

        device = self.encoder.device
        use_gan = type(self.decoder).__name__ == 'DCGANDecoder'
        use_cond = getattr(self.decoder, 'num_classes', 0) > 0
        g_total = global_total_epochs if global_total_epochs > 0 else num_epochs

        for param in self.encoder.model.parameters():
            param.requires_grad = True
        gen_params = (list(self.encoder.model.parameters())
                      + list(self.decoder.model.parameters()))
        opt_gen = optim.Adam(gen_params, lr=learning_rate, betas=(0.5, 0.999))
        for pg in opt_gen.param_groups:
            pg.setdefault('initial_lr', pg['lr'])
        sched_gen = optim.lr_scheduler.CosineAnnealingLR(
            opt_gen, T_max=g_total, eta_min=1e-5,
            last_epoch=global_start_epoch - 1 if global_start_epoch > 0 else -1,
        )
        criterion_mse = torch.nn.MSELoss()
        criterion_l1  = torch.nn.L1Loss()

        try:
            from smote_image_synthesis.decoders.autoencoder_trainer import PerceptualLoss
            perceptual_loss_fn = PerceptualLoss(device=device)
            use_perceptual = True
        except Exception:
            use_perceptual = False

        disc = None
        opt_disc = None
        if use_gan:
            n_cls = getattr(self.decoder, 'num_classes', 0) if use_cond else 0
            disc = self._build_discriminator(
                self.decoder.image_shape, base_channels=64, num_classes=n_cls
            ).to(device)
            opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        from smote_image_synthesis.pipeline import _EMA
        ema_dec = _EMA(self.decoder.model, decay=0.9999)
        ema_enc = _EMA(self.encoder.model, decay=0.9999)

        recon_epochs_global = max(1, int(g_total * 0.35))  # longer recon phase
        _logger.info(
            f"Training: {num_epochs} epochs (global {global_start_epoch}–"
            f"{global_start_epoch+num_epochs-1}/{g_total}), "
            f"GAN after epoch {recon_epochs_global}, "
            f"base_ch={BASE_CHANNELS}"
        )

        dataset = (TensorDataset(images, labels.to(device)) if labels is not None
                   else TensorDataset(images))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.encoder.model.train()
        self.decoder.model.train()
        if disc is not None:
            disc.train()

        n_critic = 5
        W_ema, W_history, lam_adv_current = 0.0, [], 0.05
        W_window = 10
        FM_WEIGHTS = [0.1, 0.3, 0.6]

        for epoch in range(num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_W_sum  = 0.0
            n_W_steps    = 0
            global_epoch = global_start_epoch + epoch
            adv_active   = use_gan and global_epoch >= recon_epochs_global

            for batch in loader:
                if labels is not None:
                    batch_imgs, batch_labels = batch[0], batch[1]
                else:
                    batch_imgs = batch[0]; batch_labels = None
                batch_imgs = batch_imgs.to(device)
                if batch_labels is not None:
                    batch_labels = batch_labels.to(device)
                bs = batch_imgs.size(0)

                if adv_active:
                    for _ in range(n_critic):
                        opt_disc.zero_grad()
                        with torch.no_grad():
                            emb_d = self.encoder.model(batch_imgs)
                            fake  = (self.decoder.model(emb_d, batch_labels)
                                     if use_cond and batch_labels is not None
                                     else self.decoder.model(emb_d))
                        d_real = disc(batch_imgs, batch_labels).mean()
                        d_fake = disc(fake,       batch_labels).mean()
                        alpha  = torch.rand(bs, 1, 1, 1, device=device)
                        interp = (alpha*batch_imgs + (1-alpha)*fake).requires_grad_(True)
                        d_interp = disc(interp, batch_labels)
                        grads = torch.autograd.grad(
                            d_interp.sum(), interp, create_graph=True)[0]
                        gp = ((grads.norm(2, dim=(1,2,3)) - 1)**2).mean()
                        d_loss = -d_real + d_fake + 10.0*gp
                        d_loss.backward(); opt_disc.step()
                        epoch_d_loss += d_loss.item()
                        epoch_W_sum  += float(d_real.item() - d_fake.item())
                        n_W_steps    += 1

                opt_gen.zero_grad()
                emb   = self.encoder.model(batch_imgs)
                recon = (self.decoder.model(emb, batch_labels)
                         if use_cond and batch_labels is not None
                         else self.decoder.model(emb))
                mse = criterion_mse(recon, batch_imgs)
                l1  = criterion_l1(recon,  batch_imgs)
                g_loss = mse + 0.5*l1
                if use_perceptual:
                    g_loss = g_loss + 0.05*perceptual_loss_fn(recon, batch_imgs)
                if adv_active:
                    g_adv = -disc(recon, batch_labels).mean()
                    real_feats = disc.get_features(batch_imgs.detach())
                    fake_feats = disc.get_features(recon)
                    n_sc = len(real_feats)
                    ws   = FM_WEIGHTS[:n_sc] if n_sc <= 3 else [1/n_sc]*n_sc
                    ws_s = sum(ws)
                    fm_loss = sum((w/ws_s)*torch.nn.functional.l1_loss(f, r.detach())
                                  for w,f,r in zip(ws, fake_feats, real_feats))
                    g_loss = g_loss + lam_adv_current*g_adv + 0.1*fm_loss
                    if lambda_repulse > 0 and batch_labels is not None:
                        g_loss = g_loss + lambda_repulse*self._compute_repulsion_loss(
                            emb, batch_labels, margin=repulse_margin)
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
                opt_gen.step()
                epoch_g_loss += g_loss.item()
                ema_dec.update(self.decoder.model)
                ema_enc.update(self.encoder.model)

            sched_gen.step()
            avg_g = epoch_g_loss / max(len(loader), 1)
            self.g_loss_history.append(avg_g)

            if adv_active and n_W_steps > 0:
                W_current = epoch_W_sum / n_W_steps
                W_ema = 0.99*W_ema + 0.01*W_current
                W_history.append(W_ema)
                if len(W_history) >= W_window:
                    dW = W_history[-1] - W_history[-W_window]
                    lam_adv_current = (min(lam_adv_current+0.005, 0.40) if dW < -0.01
                                       else max(lam_adv_current-0.005, 0.01) if dW > 0.01
                                       else lam_adv_current)
                else:
                    frac = (global_epoch - recon_epochs_global) / max(1, g_total - recon_epochs_global)
                    lam_adv_current = 0.05 + 0.15*frac

            if epoch % 10 == 0 or epoch == num_epochs-1:
                nb = max(len(loader), 1)
                d_str = (f"  D={epoch_d_loss/(nb*n_critic):.4f}  λ={lam_adv_current:.3f}"
                         if adv_active else "")
                _logger.info(
                    f"  Epoch {global_epoch:>3}/{g_total}: G={avg_g:.4f}{d_str}"
                    + (" [GAN+repulse]" if adv_active else " [recon]")
                )

        self.encoder.model.eval()
        self.decoder.model.eval()
        if disc is not None:
            disc.eval()
        ema_dec.apply(self.decoder.model)
        ema_enc.apply(self.encoder.model)
        _logger.info("Training segment complete — EMA applied")
        self.decoder._is_trained = True


def build_pipeline(device):
    encoder = ResNetEncoder(
        architecture='resnet18', embedding_dim=EMB_DIM, pretrained=True,
        device=device, freeze_backbone=False, normalize_output=True,
    )
    decoder = DCGANDecoder(
        embedding_dim=EMB_DIM,
        image_shape=(3, IMAGE_SIZE, IMAGE_SIZE),
        base_channels=BASE_CHANNELS,
        num_classes=2, class_embed_dim=64, use_self_attention=True,
        device=device,
    )
    smote_slerp = ConstrainedSMOTE(
        k_neighbors=5, use_clustering=True, normalize_embeddings=False,
        use_slerp=True, density_weighted_t=True,
        use_cluster_constraints=True, use_outlier_detection=True,
        boundary_detection_method='isolation', track_ancestry=True,
        random_state=42,
    )
    return InstrumentedPipeline(
        encoder=encoder, decoder=decoder, smote=smote_slerp,
        quality_assessor=QualityAssessor(metrics=['mse', 'psnr'], compute_diversity=True),
    )


# ---------------------------------------------------------------------------
# Synthetic generation helper
# ---------------------------------------------------------------------------

def gen_synthetics(pipeline, images, labels, device, n=50):
    """Generate SLERP and vMF synthetics from current encoder state."""
    slerp_syn, slerp_lbl = pipeline.generate_synthetic_images(n_samples=n)

    vmf_smote = ConstrainedSMOTE(
        k_neighbors=5, use_clustering=True, normalize_embeddings=False,
        use_vmf=True, vmf_concentration_scale=1.0,
        track_ancestry=True, random_state=42,
    )
    pipeline.encoder.model.eval()
    with torch.no_grad():
        embs = pipeline.encoder.model(images.to(device))
    vmf_smote.fit(embs.cpu().numpy(), labels)
    vmf_embs, vmf_lbl_np = vmf_smote.generate_synthetic(n_samples=n)

    if len(vmf_embs) > 0:
        vmf_t   = torch.from_numpy(vmf_embs).float().to(device)
        vmf_lbl = torch.from_numpy(vmf_lbl_np).long().to(device)
        pipeline.decoder.model.eval()
        with torch.no_grad():
            vmf_syn = pipeline.decoder.model(vmf_t, vmf_lbl).cpu()
    else:
        vmf_syn    = slerp_syn
        vmf_lbl_np = slerp_lbl

    return slerp_syn, slerp_lbl, vmf_syn, vmf_lbl_np


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    images, labels = load_cats_dogs(N_PER_CLASS)
    save_real_grid(images, labels, OUTPUT_DIR / 'real_samples.png')

    cat_mask = labels == 0
    dog_mask = labels == 1
    real_sample = torch.cat([images[cat_mask][:5], images[dog_mask][:5]])
    real_sample_lbl = np.array([0]*5 + [1]*5)

    pipeline = build_pipeline(device)

    # ── Training schedule: single continuous run, snapshot at mid + end ──────
    TOTAL_EPOCHS   = 200
    SNAPSHOT_EVERY = 100   # save comparison at epoch 100 and 200
    all_metrics    = {}
    snapshot_num   = 0

    logger.info(f"\n{'='*60}")
    logger.info(f"Single-run training: {TOTAL_EPOCHS} epochs total")
    logger.info(f"  base_channels={BASE_CHANNELS}, all frontier improvements ON")
    logger.info(f"{'='*60}")

    epoch_cursor = 0
    while epoch_cursor < TOTAL_EPOCHS:
        segment = min(SNAPSHOT_EVERY, TOTAL_EPOCHS - epoch_cursor)
        logger.info(f"\nTraining epochs {epoch_cursor} → {epoch_cursor+segment-1}...")

        pipeline.fit(
            images=images.to(device), labels=labels,
            train_decoder=True, decoder_epochs=segment,
            start_epoch=epoch_cursor, total_epochs=TOTAL_EPOCHS,
        )
        epoch_cursor += segment
        snapshot_num += 1

        # ── Generate synthetics ──────────────────────────────────────────────
        logger.info("Generating synthetics for comparison...")
        slerp_syn, slerp_lbl, vmf_syn, vmf_lbl = gen_synthetics(
            pipeline, images, labels, device, n=50
        )

        # ── G_loss convergence ───────────────────────────────────────────────
        recent_g  = np.mean(pipeline.g_loss_history[-10:]) if len(pipeline.g_loss_history) >= 10 else np.mean(pipeline.g_loss_history) if pipeline.g_loss_history else 1.0
        logger.info(f"  Snapshot {snapshot_num} — avg G_loss (last 10 ep): {recent_g:.4f}  ({epoch_cursor} epochs done)")

        metrics = {'epochs': epoch_cursor, 'avg_g_loss_last10': float(recent_g)}
        all_metrics[f'snapshot_{snapshot_num}'] = metrics

        # ── Comparison grid ──────────────────────────────────────────────────
        save_comparison(
            real=real_sample, real_lbl=real_sample_lbl,
            slerp_syn=slerp_syn[:10].cpu(), slerp_lbl=slerp_lbl[:10],
            vmf_syn=vmf_syn[:10].cpu(),     vmf_lbl=vmf_lbl[:10],
            round_num=snapshot_num,
            metrics={'psnr': 0, 'mse': 0, 'g_loss': recent_g},
            path=OUTPUT_DIR / f'comparison_ep{epoch_cursor}.png',
        )

        pipeline.save_pipeline(str(OUTPUT_DIR / 'frontier_pipeline'))
        logger.info(f"  Checkpoint + comparison saved")

        # ── Loop: continue if still improving and below target ───────────────
        if epoch_cursor >= TOTAL_EPOCHS:
            logger.info("Reached epoch budget. Done.")
            break
        if snapshot_num > 1:
            prev_g = list(all_metrics.values())[-2].get('avg_g_loss_last10', 1.0)
            improvement = prev_g - recent_g
            logger.info(f"  G_loss improvement: {improvement:+.4f}")
            if recent_g <= GLOSS_TARGET:
                logger.info(f"  G_loss target reached ({recent_g:.4f} ≤ {GLOSS_TARGET}). Done.")
                break
            if improvement < MIN_IMPROVEMENT:
                logger.info(f"  G_loss improvement < {MIN_IMPROVEMENT}. Running final segment.")
                # Run one last segment then exit
        # Continue training

    # ── Save metrics ─────────────────────────────────────────────────────────
    with open(OUTPUT_DIR / 'metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("DONE")
    logger.info(f"  {OUTPUT_DIR.resolve()}")
    for name, m in all_metrics.items():
        logger.info(f"  {name}: {m['epochs']} epochs  G_loss={m['avg_g_loss_last10']:.4f}")
    logger.info("  comparison_ep100.png / comparison_ep200.png — real | SLERP | vMF")


if __name__ == '__main__':
    main()
