"""
Quality assessment for synthetic images.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from scipy import linalg
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class QualityAssessor:
    """Comprehensive quality assessment for synthetic images.
    
    Features:
    - Multiple quality metrics (FID, LPIPS, SSIM, PSNR)
    - Diversity metrics for evaluating sample variety
    - Distribution analysis for synthetic vs real comparison
    - Visual quality reporting and analysis
    """
    
    def __init__(self, 
                 metrics: List[str] = ['fid', 'lpips', 'ssim', 'psnr'],
                 fid_batch_size: int = 50,
                 compute_diversity: bool = True,
                 diversity_sample_size: int = 1000,
                 device: Optional[torch.device] = None):
        """
        Initialize quality assessor.
        
        Args:
            metrics: List of metrics to compute
            fid_batch_size: Batch size for FID computation
            compute_diversity: Whether to compute diversity metrics
            diversity_sample_size: Sample size for diversity computation
            device: Device for computation
        """
        self.metrics = metrics
        self.fid_batch_size = fid_batch_size
        self.compute_diversity = compute_diversity
        self.diversity_sample_size = diversity_sample_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Supported metrics
        self.supported_metrics = [
            'fid', 'lpips', 'ssim', 'psnr', 'ms_ssim', 'mse', 'mae'
        ]
        
        # Validate metrics
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}. Supported: {self.supported_metrics}")
        
        # Initialize metric computers
        self._initialize_metric_computers()
        
        logger.info(f"Initialized QualityAssessor with metrics: {metrics}")
                
    def evaluate_quality(self, 
                        synthetic_images: torch.Tensor, 
                        real_images: torch.Tensor,
                        return_detailed: bool = False) -> Dict[str, Any]:
        """
        Evaluate quality of synthetic images against real images.
        
        Args:
            synthetic_images: Generated images [B, C, H, W]
            real_images: Real reference images [B, C, H, W]
            return_detailed: Whether to return detailed analysis
            
        Returns:
            Dictionary of metric scores and optional detailed analysis
        """
        # Validate inputs
        self._validate_inputs(synthetic_images, real_images)
        
        results = {'metrics': {}}
        
        # Compute each requested metric
        for metric in self.metrics:
            try:
                if metric == 'fid':
                    score = self._compute_fid(synthetic_images, real_images)
                elif metric == 'lpips':
                    score = self._compute_lpips(synthetic_images, real_images)
                elif metric == 'ssim':
                    score = self._compute_ssim(synthetic_images, real_images)
                elif metric == 'ms_ssim':
                    score = self._compute_ms_ssim(synthetic_images, real_images)
                elif metric == 'psnr':
                    score = self._compute_psnr(synthetic_images, real_images)
                elif metric == 'mse':
                    score = self._compute_mse(synthetic_images, real_images)
                elif metric == 'mae':
                    score = self._compute_mae(synthetic_images, real_images)
                else:
                    logger.warning(f"Unknown metric: {metric}, skipping")
                    continue
                
                results['metrics'][metric] = float(score)
                logger.debug(f"Computed {metric}: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Error computing {metric}: {e}")
                results['metrics'][metric] = float('nan')
        
        # Compute diversity metrics if requested
        if self.compute_diversity:
            diversity_metrics = self.compute_diversity_metrics(synthetic_images)
            results['diversity'] = diversity_metrics
        
        # Add detailed analysis if requested
        if return_detailed:
            results['detailed_analysis'] = self._compute_detailed_analysis(
                synthetic_images, real_images
            )
        
        return results
        
    def compute_diversity_metrics(self, synthetic_images: torch.Tensor) -> Dict[str, float]:
        """
        Compute diversity metrics for synthetic images.
        
        Args:
            synthetic_images: Generated images [B, C, H, W]
            
        Returns:
            Dictionary of diversity scores
        """
        results = {}
        
        # Limit sample size for efficiency
        if len(synthetic_images) > self.diversity_sample_size:
            indices = torch.randperm(len(synthetic_images))[:self.diversity_sample_size]
            images = synthetic_images[indices]
        else:
            images = synthetic_images
        
        batch_size = images.shape[0]
        if batch_size < 2:
            logger.warning("Need at least 2 images for diversity computation")
            return {'mean_pairwise_distance': 0.0, 'std_pairwise_distance': 0.0}
        
        # Flatten images for distance computation
        flattened = images.view(batch_size, -1).cpu().numpy()
        
        # Compute pairwise distances efficiently
        distances = pairwise_distances(flattened, metric='euclidean')
        
        # Extract upper triangle (excluding diagonal)
        upper_triangle = distances[np.triu_indices_from(distances, k=1)]
        
        if len(upper_triangle) > 0:
            results['mean_pairwise_distance'] = float(np.mean(upper_triangle))
            results['std_pairwise_distance'] = float(np.std(upper_triangle))
            results['min_pairwise_distance'] = float(np.min(upper_triangle))
            results['max_pairwise_distance'] = float(np.max(upper_triangle))
            
            # Compute diversity index (normalized)
            mean_dist = results['mean_pairwise_distance']
            max_possible_dist = np.sqrt(np.prod(images.shape[1:]))  # Theoretical max
            results['diversity_index'] = mean_dist / max_possible_dist if max_possible_dist > 0 else 0.0
        else:
            results = {
                'mean_pairwise_distance': 0.0,
                'std_pairwise_distance': 0.0,
                'min_pairwise_distance': 0.0,
                'max_pairwise_distance': 0.0,
                'diversity_index': 0.0
            }
        
        logger.debug(f"Computed diversity metrics for {batch_size} images")
        return results
        
    def generate_report(self, metrics: Dict[str, float]) -> str:
        """
        Generate a quality assessment report.
        
        Args:
            metrics: Dictionary of computed metrics
            
        Returns:
            Formatted report string
        """
        report = "Quality Assessment Report\n"
        report += "=" * 30 + "\n\n"
        
        for metric, value in metrics.items():
            report += f"{metric.upper()}: {value:.4f}\n"
            
        return report
        
    def _compute_mse(self, synthetic: torch.Tensor, real: torch.Tensor) -> float:
        """Compute Mean Squared Error."""
        mse = torch.mean((synthetic - real) ** 2)
        return mse.item()
        
    def _compute_ssim(self, synthetic: torch.Tensor, real: torch.Tensor) -> float:
        """Compute Structural Similarity Index (placeholder)."""
        # Placeholder implementation - would use proper SSIM calculation
        # For now, return inverse of MSE as approximation
        mse = self._compute_mse(synthetic, real)
        return 1.0 / (1.0 + mse)
        
    def _compute_lpips(self, synthetic: torch.Tensor, real: torch.Tensor) -> float:
        """Compute LPIPS (placeholder)."""
        # Placeholder - would use actual LPIPS implementation
        return 0.5
        
    def _compute_fid(self, synthetic: torch.Tensor, real: torch.Tensor) -> float:
        """Compute FID (placeholder)."""
        # Placeholder - would use actual FID implementation
        return 50.0