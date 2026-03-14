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
                          learning_rate: float = 3e-4, batch_size: int = 32) -> None:
        """
        Train encoder and decoder jointly end-to-end for reconstruction.
        This produces embeddings that carry pixel-level information the decoder
        can actually use, avoiding the grey-blob failure mode.
        """
        import logging
        from torch.utils.data import DataLoader, TensorDataset
        import torch.optim as optim
        import torch.nn as nn
        _logger = logging.getLogger(__name__)

        device = self.encoder.device

        # Unfreeze ALL encoder params for joint training
        for param in self.encoder.model.parameters():
            param.requires_grad = True

        all_params = (list(self.encoder.model.parameters()) +
                      list(self.decoder.model.parameters()))

        optimizer = optim.Adam(all_params, lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-5)
        criterion_mse = nn.MSELoss()
        criterion_l1 = nn.L1Loss()

        # Try to load perceptual loss (VGG-based) for better color/texture
        try:
            from .decoders.autoencoder_trainer import PerceptualLoss
            perceptual_loss_fn = PerceptualLoss(device=device)
            use_perceptual = True
            _logger.info("  Perceptual loss: enabled")
        except Exception as e:
            use_perceptual = False
            _logger.info(f"  Perceptual loss: unavailable ({e}), using MSE+L1 only")

        dataset = TensorDataset(images)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0)

        self.encoder.model.train()
        self.decoder.model.train()
        _logger.info(f"E2E training: encoder+decoder jointly for {num_epochs} epochs")

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for (batch_imgs,) in loader:
                batch_imgs = batch_imgs.to(device)
                optimizer.zero_grad()
                embeddings = self.encoder.model(batch_imgs)
                reconstructed = self.decoder.model(embeddings)
                mse = criterion_mse(reconstructed, batch_imgs)
                l1 = criterion_l1(reconstructed, batch_imgs)
                loss = mse + 0.5 * l1
                if use_perceptual:
                    loss = loss + 0.05 * perceptual_loss_fn(reconstructed, batch_imgs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                _logger.info(f"  E2E Epoch {epoch:>3}: loss={epoch_loss/len(loader):.4f}")

        self.encoder.model.eval()
        self.decoder.model.eval()
        self.decoder._is_trained = True
        _logger.info("E2E training complete")
        
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