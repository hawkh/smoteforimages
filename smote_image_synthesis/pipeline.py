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
            
    def fit(self, images: torch.Tensor, labels: np.ndarray) -> None:
        """
        Fit the pipeline on training data.
        
        Args:
            images: Training images [B, C, H, W]
            labels: Corresponding labels [B]
        """
        # Generate embeddings
        embeddings = self.encoder.encode(images)
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # Fit SMOTE on embeddings
        self.smote.fit(embeddings_np, labels)
        
        # Train decoder (if needed)
        self.decoder.train_decoder(embeddings, images)
        
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
            Quality metrics dictionary
        """
        return self.quality_assessor.evaluate_quality(synthetic_images, real_images)
        
    def save_pipeline(self, base_path: str) -> None:
        """Save the entire pipeline."""
        self.encoder.save_model(f"{base_path}_encoder.pth")
        self.decoder.save_model(f"{base_path}_decoder.pth")
        
    def load_pipeline(self, base_path: str) -> None:
        """Load the entire pipeline."""
        self.encoder.load_model(f"{base_path}_encoder.pth")
        self.decoder.load_model(f"{base_path}_decoder.pth")