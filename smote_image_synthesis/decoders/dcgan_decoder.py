"""
DCGAN-style convolutional decoder for high-quality image synthesis.

Architecture: embedding → FC(512*4*4) → reshape → deconv stack → image
Channels:     base_ch → base_ch/2 → ... → 32 → C
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


class DCGANDecoder(BaseDecoder):
    """
    DCGAN-style decoder: projects embeddings into a 4×4 feature map then
    progressively doubles spatial resolution with transposed convolutions.

    For a 64×64 target:
        emb (256/512)
        → Linear → BN → ReLU → (256, 4, 4)
        → DeConv 256→128  (8×8)   BN ReLU
        → DeConv 128→64   (16×16) BN ReLU
        → DeConv 64→32    (32×32) BN ReLU
        → DeConv 32→C     (64×64) Tanh
    """

    def __init__(
        self,
        embedding_dim: int,
        image_shape: Tuple[int, int, int],
        base_channels: int = 256,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        config = config or {}
        config["base_channels"] = base_channels
        super().__init__(embedding_dim, image_shape, device, config)

        self.base_channels = base_channels
        self.model = self._build_model().to(self.device)
        self._initialize_weights()

        logger.info(
            f"DCGANDecoder: emb={embedding_dim} → {image_shape}, "
            f"base_channels={base_channels}"
        )

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        c, h, w = self.image_shape

        # Number of 2× upsampling steps needed: 4 → h
        n_up = 0
        size = 4
        while size < h:
            size *= 2
            n_up += 1

        # Channel schedule: [base, base/2, ..., 32, 32, ...]
        ch = [max(32, self.base_channels // (2 ** i)) for i in range(n_up + 1)]

        layers: list = [
            # FC projection: embedding → (ch[0], 4, 4)
            nn.Linear(self.embedding_dim, ch[0] * 4 * 4, bias=False),
            nn.BatchNorm1d(ch[0] * 4 * 4),
            nn.ReLU(inplace=True),
            _Reshape(ch[0], 4, 4),
        ]

        # Intermediate upsampling blocks
        for i in range(n_up - 1):
            layers += [
                nn.ConvTranspose2d(
                    ch[i], ch[i + 1], kernel_size=4, stride=2, padding=1, bias=False
                ),
                nn.BatchNorm2d(ch[i + 1]),
                nn.ReLU(inplace=True),
            ]

        # Final layer: last intermediate channel → output channels, Tanh
        layers += [
            nn.ConvTranspose2d(
                ch[n_up - 1], c, kernel_size=4, stride=2, padding=1, bias=True
            ),
            nn.Tanh(),
        ]

        return nn.Sequential(*layers)

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

    # ------------------------------------------------------------------
    # BaseDecoder interface
    # ------------------------------------------------------------------

    def decode(self, embeddings: torch.Tensor) -> torch.Tensor:
        is_valid, err = self.validate_embeddings(embeddings)
        if not is_valid:
            raise ValueError(f"Invalid embeddings: {err}")
        embeddings = embeddings.to(self.device)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            images = self.model(embeddings)
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
        decoder = cls(
            embedding_dim=data["embedding_dim"],
            image_shape=tuple(data["image_shape"]),
            **data.get("config", {}),
        )
        model_path = Path(data["model_path"])
        if model_path.exists():
            decoder.load_model(model_path.with_suffix(""))
        return decoder
