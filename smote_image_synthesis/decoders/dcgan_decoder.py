"""
DCGAN-style convolutional decoder for high-quality image synthesis.

Architecture:
    emb [+ class_emb (optional)]
    → Linear → BN → ReLU → reshape(ch[0], 4, 4)
    → DeConv blocks with BN + ReLU
    → SelfAttention2d at 16×16 spatial resolution
    → DeConv final → Tanh
"""

from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
from pathlib import Path
import logging

from .base import BaseDecoder

logger = logging.getLogger(__name__)


class _Reshape(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        super().__init__()
        self.c, self.h, self.w = c, h, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), self.c, self.h, self.w)


class SelfAttention2d(nn.Module):
    """Non-local self-attention block (Zhang et al. 2019, SAGAN).

    Inserts a residual attention term scaled by a learnable gamma initialised
    to 0 so the block starts as an identity and grows gradually during training.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        mid = max(1, in_channels // 8)
        self.q = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.k = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.v = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self._mid = mid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.q(x).view(B, self._mid, -1).permute(0, 2, 1)   # (B, HW, mid)
        k = self.k(x).view(B, self._mid, -1)                      # (B, mid, HW)
        scale = self._mid ** -0.5
        attn = torch.softmax(torch.bmm(q, k) * scale, dim=-1)    # (B, HW, HW)
        v = self.v(x).view(B, C, -1)                               # (B, C, HW)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out


class _Generator(nn.Module):
    """Generator backbone with optional class conditioning.

    If *class_embed* is provided, the class embedding is concatenated to the
    noise/image embedding *before* the first linear projection.  The caller is
    responsible for ensuring the first Linear layer was built with the matching
    wider input dimension.
    """

    def __init__(
        self,
        main: nn.Sequential,
        class_embed: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.main = main
        self.class_embed = class_embed

    def forward(
        self,
        z: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.class_embed is not None:
            if labels is not None:
                c = self.class_embed(labels.to(z.device).long())
            else:
                # No class provided — use zero embedding (null class signal)
                c = torch.zeros(
                    z.size(0), self.class_embed.embedding_dim, device=z.device
                )
            z = torch.cat([z, c], dim=1)
        return self.main(z)


class DCGANDecoder(BaseDecoder):
    """
    DCGAN-style decoder with self-attention and optional class conditioning.

    For a 64×64 target (base_channels=512, num_classes=2):
        emb(512) + class_emb(64) = 576
        → Linear(576, 512*4*4) → BN → ReLU → reshape(512, 4, 4)
        → DeConv 512→256  (8×8)   BN ReLU
        → DeConv 256→128  (16×16) BN ReLU  ← SelfAttention2d inserted here
        → DeConv 128→64   (32×32) BN ReLU
        → DeConv 64→3     (64×64) Tanh
    """

    def __init__(
        self,
        embedding_dim: int,
        image_shape: Tuple[int, int, int],
        base_channels: int = 256,
        num_classes: int = 0,
        class_embed_dim: int = 64,
        use_self_attention: bool = True,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        config = config or {}
        config.update({
            "base_channels": base_channels,
            "num_classes": num_classes,
            "class_embed_dim": class_embed_dim,
            "use_self_attention": use_self_attention,
        })
        super().__init__(embedding_dim, image_shape, device, config)

        self.base_channels = base_channels
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim
        self.use_self_attention = use_self_attention

        self.model = self._build_model().to(self.device)
        self._initialize_weights()

        cond_str = f", num_classes={num_classes}" if num_classes > 0 else ""
        attn_str = ", self_attention" if use_self_attention else ""
        logger.info(
            f"DCGANDecoder: emb={embedding_dim} → {image_shape}, "
            f"base_channels={base_channels}{cond_str}{attn_str}"
        )

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        c, h, w = self.image_shape

        # Validate power-of-2 spatial dimensions (required for transposed conv upsampling)
        if h < 8 or (h & (h - 1)) != 0:
            raise ValueError(
                f"image_shape height must be a power of 2 >= 8, got {h}"
            )
        if w < 8 or (w & (w - 1)) != 0:
            raise ValueError(
                f"image_shape width must be a power of 2 >= 8, got {w}"
            )

        # Effective input: image embedding + optional class embedding
        effective_dim = self.embedding_dim
        class_embed: Optional[nn.Embedding] = None
        if self.num_classes > 0:
            class_embed = nn.Embedding(self.num_classes, self.class_embed_dim)
            nn.init.normal_(class_embed.weight, 0.0, 0.02)
            effective_dim = self.embedding_dim + self.class_embed_dim

        # Number of 2× upsampling steps needed: 4 → h
        n_up = 0
        size = 4
        while size < h:
            size *= 2
            n_up += 1

        # Channel schedule: [base, base/2, ..., 32, ...]
        ch = [max(32, self.base_channels // (2 ** i)) for i in range(n_up + 1)]

        layers: list = [
            nn.Linear(effective_dim, ch[0] * 4 * 4, bias=False),
            nn.BatchNorm1d(ch[0] * 4 * 4),
            nn.ReLU(inplace=True),
            _Reshape(ch[0], 4, 4),
        ]

        current_size = 4
        for i in range(n_up - 1):
            layers += [
                nn.ConvTranspose2d(ch[i], ch[i + 1], 4, 2, 1, bias=False),
                nn.BatchNorm2d(ch[i + 1]),
                nn.ReLU(inplace=True),
            ]
            current_size *= 2
            # Insert self-attention once the spatial map reaches 16×16
            if self.use_self_attention and current_size == 16:
                layers.append(SelfAttention2d(ch[i + 1]))

        # Final upsampling layer → Tanh output
        layers += [
            nn.ConvTranspose2d(ch[n_up - 1], c, 4, 2, 1, bias=True),
            nn.Tanh(),
        ]

        return _Generator(nn.Sequential(*layers), class_embed)

    def _initialize_weights(self) -> None:
        for m in self.model.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            # nn.Embedding is initialised in _build_model; leave it alone

    # ------------------------------------------------------------------
    # BaseDecoder interface
    # ------------------------------------------------------------------

    def decode(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode embeddings to images, optionally conditioned on class labels."""
        is_valid, err = self.validate_embeddings(embeddings)
        if not is_valid:
            raise ValueError(f"Invalid embeddings: {err}")
        embeddings = embeddings.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            images = self.model(embeddings, labels)
        self.model.train(was_training)
        return images

    def train_decoder(self, embeddings: torch.Tensor, images: torch.Tensor) -> None:
        """Placeholder — training done via SynthesisPipeline._train_end_to_end."""
        self._is_trained = True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load_from_config(cls, config_path: Path) -> "DCGANDecoder":
        import json
        config_path = Path(config_path)
        with open(config_path) as f:
            data = json.load(f)
        cfg = dict(data.get("config", {}))
        decoder = cls(
            embedding_dim=data["embedding_dim"],
            image_shape=tuple(data["image_shape"]),
            base_channels=cfg.pop("base_channels", 256),
            num_classes=cfg.pop("num_classes", 0),
            class_embed_dim=cfg.pop("class_embed_dim", 64),
            use_self_attention=cfg.pop("use_self_attention", True),
        )
        model_path = Path(data["model_path"])
        if model_path.exists():
            decoder.load_model(model_path.with_suffix(""))
        return decoder
