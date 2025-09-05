"""
Quality assessment reporting and visualization utilities.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class QualityReporter:
    """
    Comprehensive quality assessment reporting and visualization.
    
    Features:
    - Detailed quality metric reports
    - Visual comparisons between synthetic and real images
    - Statistical analysis and distribution plots
    - Export capabilities for reports and visualizations
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        report_format: str = 'html',
        save_plots: bool = True,
        plot_dpi: int = 300
    ):
        """
        Initialize quality reporter.
        
        Args:
            output_dir: Directory for saving reports and plots
            report_format: Format for reports ('html', 'json', 'txt')
            save_plots: Whether to save generated plots
            plot_dpi: DPI for saved plots
        """
        self.output_dir = Path(output_dir) if output_dir else Path('./quality_reports')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.report_format = report_format
        self.save_plots = save_plots
        self.plot_dpi = plot_dpi
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info(f"Initialized QualityReporter with output_dir: {self.output_dir}")
    
    def generate_comprehensive_report(
        self,
        quality_results: Dict[str, Any],
        synthetic_images: torch.Tensor,
        real_images: torch.Tensor,
        report_name: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive quality assessment report.
        
        Args:
            quality_results: Results from QualityAssessor
            synthetic_images: Generated images [B, C, H, W]
            real_images: Real reference images [B, C, H, W]
            report_name: Name for the report file
            
        Returns:
            Path to generated report
        """
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"quality_report_{timestamp}"
        
        # Generate visualizations
        plot_paths = self._generate_all_plots(
            quality_results, synthetic_images, real_images, report_name
        )
        
        # Generate report based on format
        if self.report_format == 'html':
            report_path = self._generate_html_report(
                quality_results, plot_paths, report_name
            )
        elif self.report_format == 'json':
            report_path = self._generate_json_report(
                quality_results, report_name
            )
        else:  # txt
            report_path = self._generate_text_report(
                quality_results, report_name
            )
        
        logger.info(f"Quality report generated: {report_path}")
        return str(report_path)
    
    def _generate_all_plots(
        self,
        quality_results: Dict[str, Any],
        synthetic_images: torch.Tensor,
        real_images: torch.Tensor,
        report_name: str
    ) -> Dict[str, str]:
        """Generate all visualization plots."""
        plot_paths = {}
        
        # Metrics overview plot
        if 'metrics' in quality_results:
            plot_path = self._plot_metrics_overview(
                quality_results['metrics'], f"{report_name}_metrics"
            )
            plot_paths['metrics_overview'] = plot_path
        
        # Diversity analysis plot
        if 'diversity' in quality_results:
            plot_path = self._plot_diversity_analysis(
                quality_results['diversity'], f"{report_name}_diversity"
            )
            plot_paths['diversity_analysis'] = plot_path
        
        # Image comparison grid
        plot_path = self._plot_image_comparison(
            synthetic_images, real_images, f"{report_name}_comparison"
        )
        plot_paths['image_comparison'] = plot_path
        
        # Distribution analysis
        plot_path = self._plot_distribution_analysis(
            synthetic_images, real_images, f"{report_name}_distribution"
        )
        plot_paths['distribution_analysis'] = plot_path
        
        # Detailed analysis if available
        if 'detailed_analysis' in quality_results:
            plot_path = self._plot_detailed_analysis(
                quality_results['detailed_analysis'], f"{report_name}_detailed"
            )
            plot_paths['detailed_analysis'] = plot_path
        
        return plot_paths
    
    def _plot_metrics_overview(self, metrics: Dict[str, float], name: str) -> str:
        """Plot overview of quality metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of metrics
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # Filter out NaN values
        valid_metrics = [(name, value) for name, value in zip(metric_names, metric_values) 
                        if not np.isnan(value)]
        
        if valid_metrics:
            names, values = zip(*valid_metrics)
            
            bars = ax1.bar(names, values, alpha=0.7)
            ax1.set_title('Quality Metrics Overview')
            ax1.set_ylabel('Score')
            ax1.tick_params(axis='x', rotation=45)
            
            # Color bars based on values (assuming lower is better for most metrics)
            colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(values)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.4f}', ha='center', va='bottom')
        
        # Radar plot of normalized metrics
        if len(valid_metrics) >= 3:
            self._create_radar_plot(ax2, dict(valid_metrics))
        else:
            ax2.text(0.5, 0.5, 'Need at least 3 metrics\\nfor radar plot', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Metrics Radar Plot')
        
        plt.tight_layout()
        
        if self.save_plots:
            plot_path = self.output_dir / f"{name}.png"
            plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return ""
    
    def _create_radar_plot(self, ax, metrics: Dict[str, float]):
        """Create radar plot for metrics."""
        # Normalize metrics to 0-1 range for radar plot
        values = list(metrics.values())
        # Simple normalization - in practice, you'd use domain-specific ranges
        normalized_values = [(v - min(values)) / (max(values) - min(values) + 1e-8) 
                            for v in values]
        
        labels = list(metrics.keys())
        
        # Compute angle for each metric
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        
        # Close the plot
        normalized_values += normalized_values[:1]
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, normalized_values, 'o-', linewidth=2)
        ax.fill(angles, normalized_values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title('Normalized Metrics Radar')
        ax.grid(True)
    
    def _plot_diversity_analysis(self, diversity: Dict[str, float], name: str) -> str:
        """Plot diversity metrics analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Diversity metrics bar plot
        ax = axes[0, 0]
        div_metrics = ['mean_pairwise_distance', 'std_pairwise_distance', 
                      'diversity_index']
        div_values = [diversity.get(metric, 0) for metric in div_metrics]
        
        bars = ax.bar(div_metrics, div_values, alpha=0.7)
        ax.set_title('Diversity Metrics')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
        
        # Distance distribution (if available)
        ax = axes[0, 1]
        if 'min_pairwise_distance' in diversity and 'max_pairwise_distance' in diversity:
            # Create synthetic distribution for visualization
            mean_dist = diversity.get('mean_pairwise_distance', 0)
            std_dist = diversity.get('std_pairwise_distance', 0)
            
            if std_dist > 0:
                # Generate sample distribution
                samples = np.random.normal(mean_dist, std_dist, 1000)
                samples = samples[samples >= 0]  # Remove negative values
                
                ax.hist(samples, bins=30, alpha=0.7, density=True)
                ax.axvline(mean_dist, color='red', linestyle='--', 
                          label=f'Mean: {mean_dist:.3f}')
                ax.set_title('Pairwise Distance Distribution')
                ax.set_xlabel('Distance')
                ax.set_ylabel('Density')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'Insufficient diversity data', 
                       ha='center', va='center', transform=ax.transAxes)
        
        # Diversity score interpretation
        ax = axes[1, 0]
        diversity_score = diversity.get('diversity_index', 0)
        
        # Create a gauge-like plot
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        ax.plot(theta, r, 'k-', linewidth=3)
        
        # Color segments
        n_segments = 5
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        for i in range(n_segments):
            start = i * np.pi / n_segments
            end = (i + 1) * np.pi / n_segments
            theta_seg = np.linspace(start, end, 20)
            r_seg = np.ones_like(theta_seg)
            ax.fill_between(theta_seg, 0, r_seg, color=colors[i], alpha=0.3)
        
        # Add diversity score indicator
        score_angle = diversity_score * np.pi
        ax.plot([score_angle, score_angle], [0, 1], 'r-', linewidth=4)
        ax.set_ylim(0, 1.2)
        ax.set_xlim(0, np.pi)
        ax.set_title(f'Diversity Score: {diversity_score:.3f}')
        ax.set_aspect('equal')
        
        # Summary text
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""Diversity Analysis Summary:

Mean Pairwise Distance: {diversity.get('mean_pairwise_distance', 0):.4f}
Std Pairwise Distance: {diversity.get('std_pairwise_distance', 0):.4f}
Min Distance: {diversity.get('min_pairwise_distance', 0):.4f}
Max Distance: {diversity.get('max_pairwise_distance', 0):.4f}
Diversity Index: {diversity.get('diversity_index', 0):.4f}

Interpretation:
{'High diversity' if diversity.get('diversity_index', 0) > 0.5 else 'Low diversity'}"""
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        if self.save_plots:
            plot_path = self.output_dir / f"{name}.png"
            plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return ""
    
    def _plot_image_comparison(self, synthetic_images: torch.Tensor, 
                              real_images: torch.Tensor, name: str) -> str:
        """Plot side-by-side comparison of synthetic and real images."""
        n_samples = min(8, len(synthetic_images), len(real_images))
        
        fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
        
        for i in range(n_samples):
            # Real images (top row)
            real_img = real_images[i].cpu().detach()
            if real_img.shape[0] == 1:  # Grayscale
                axes[0, i].imshow(real_img.squeeze(), cmap='gray')
            else:  # RGB
                # Normalize to [0, 1] if needed
                if real_img.min() < 0:
                    real_img = (real_img + 1) / 2
                axes[0, i].imshow(real_img.permute(1, 2, 0))
            
            axes[0, i].set_title('Real' if i == 0 else '')
            axes[0, i].axis('off')
            
            # Synthetic images (bottom row)
            synth_img = synthetic_images[i].cpu().detach()
            if synth_img.shape[0] == 1:  # Grayscale
                axes[1, i].imshow(synth_img.squeeze(), cmap='gray')
            else:  # RGB
                # Normalize to [0, 1] if needed
                if synth_img.min() < 0:
                    synth_img = (synth_img + 1) / 2
                axes[1, i].imshow(synth_img.permute(1, 2, 0))
            
            axes[1, i].set_title('Synthetic' if i == 0 else '')
            axes[1, i].axis('off')
        
        plt.suptitle('Real vs Synthetic Images Comparison')
        plt.tight_layout()
        
        if self.save_plots:
            plot_path = self.output_dir / f"{name}.png"
            plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return ""
    
    def _plot_distribution_analysis(self, synthetic_images: torch.Tensor, 
                                   real_images: torch.Tensor, name: str) -> str:
        """Plot statistical distribution analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Flatten images for analysis
        synth_flat = synthetic_images.flatten().cpu().numpy()
        real_flat = real_images.flatten().cpu().numpy()
        
        # Pixel value distributions
        ax = axes[0, 0]
        ax.hist(real_flat, bins=50, alpha=0.7, label='Real', density=True)
        ax.hist(synth_flat, bins=50, alpha=0.7, label='Synthetic', density=True)
        ax.set_title('Pixel Value Distribution')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Density')
        ax.legend()
        
        # Mean pixel values per image
        ax = axes[0, 1]
        synth_means = synthetic_images.mean(dim=(1, 2, 3)).cpu().numpy()
        real_means = real_images.mean(dim=(1, 2, 3)).cpu().numpy()
        
        ax.hist(real_means, bins=20, alpha=0.7, label='Real')
        ax.hist(synth_means, bins=20, alpha=0.7, label='Synthetic')
        ax.set_title('Mean Pixel Value per Image')
        ax.set_xlabel('Mean Pixel Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Standard deviation per image
        ax = axes[1, 0]
        synth_stds = synthetic_images.std(dim=(1, 2, 3)).cpu().numpy()
        real_stds = real_images.std(dim=(1, 2, 3)).cpu().numpy()
        
        ax.hist(real_stds, bins=20, alpha=0.7, label='Real')
        ax.hist(synth_stds, bins=20, alpha=0.7, label='Synthetic')
        ax.set_title('Pixel Std Dev per Image')
        ax.set_xlabel('Standard Deviation')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Statistical summary
        ax = axes[1, 1]
        ax.axis('off')
        
        # Compute statistics
        stats_text = f"""Statistical Summary:

Real Images:
  Mean: {real_flat.mean():.4f}
  Std: {real_flat.std():.4f}
  Min: {real_flat.min():.4f}
  Max: {real_flat.max():.4f}

Synthetic Images:
  Mean: {synth_flat.mean():.4f}
  Std: {synth_flat.std():.4f}
  Min: {synth_flat.min():.4f}
  Max: {synth_flat.max():.4f}

Distribution Similarity:
  Mean diff: {abs(real_flat.mean() - synth_flat.mean()):.4f}
  Std diff: {abs(real_flat.std() - synth_flat.std()):.4f}"""
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        
        if self.save_plots:
            plot_path = self.output_dir / f"{name}.png"
            plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return ""
    
    def _plot_detailed_analysis(self, detailed_analysis: Dict[str, Any], name: str) -> str:
        """Plot detailed analysis results."""
        # This is a placeholder for detailed analysis plotting
        # Implementation would depend on the specific detailed analysis results
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'Detailed Analysis\n(Implementation depends on analysis type)',
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Detailed Analysis Results')
        
        if self.save_plots:
            plot_path = self.output_dir / f"{name}.png"
            plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return ""
    
    def _generate_html_report(self, quality_results: Dict[str, Any], 
                             plot_paths: Dict[str, str], report_name: str) -> Path:
        """Generate HTML quality report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quality Assessment Report - {report_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #333; }}
                .section {{ margin: 30px 0; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .plot img {{ max-width: 100%; height: auto; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Quality Assessment Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Add metrics section
        if 'metrics' in quality_results:
            html_content += """
            <div class="section">
                <h2>Quality Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
            """
            
            for metric, value in quality_results['metrics'].items():
                interpretation = self._interpret_metric(metric, value)
                html_content += f"""
                    <tr>
                        <td>{metric.upper()}</td>
                        <td>{value:.6f}</td>
                        <td>{interpretation}</td>
                    </tr>
                """
            
            html_content += "</table></div>"
        
        # Add diversity section
        if 'diversity' in quality_results:
            html_content += """
            <div class="section">
                <h2>Diversity Analysis</h2>
            """
            
            for metric, value in quality_results['diversity'].items():
                html_content += f'<div class="metric"><strong>{metric}:</strong> {value:.6f}</div>'
            
            html_content += "</div>"
        
        # Add plots
        for plot_name, plot_path in plot_paths.items():
            if plot_path:
                plot_filename = Path(plot_path).name
                html_content += f"""
                <div class="section">
                    <h3>{plot_name.replace('_', ' ').title()}</h3>
                    <div class="plot">
                        <img src="{plot_filename}" alt="{plot_name}">
                    </div>
                </div>
                """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save HTML report
        report_path = self.output_dir / f"{report_name}.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
    
    def _generate_json_report(self, quality_results: Dict[str, Any], report_name: str) -> Path:
        """Generate JSON quality report."""
        report_data = {
            'report_name': report_name,
            'timestamp': datetime.now().isoformat(),
            'quality_results': quality_results
        }
        
        report_path = self.output_dir / f"{report_name}.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return report_path
    
    def _generate_text_report(self, quality_results: Dict[str, Any], report_name: str) -> Path:
        """Generate text quality report."""
        lines = [
            f"Quality Assessment Report - {report_name}",
            "=" * 50,
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Add metrics
        if 'metrics' in quality_results:
            lines.extend([
                "Quality Metrics:",
                "-" * 20
            ])
            
            for metric, value in quality_results['metrics'].items():
                interpretation = self._interpret_metric(metric, value)
                lines.append(f"{metric.upper()}: {value:.6f} ({interpretation})")
            
            lines.append("")
        
        # Add diversity
        if 'diversity' in quality_results:
            lines.extend([
                "Diversity Analysis:",
                "-" * 20
            ])
            
            for metric, value in quality_results['diversity'].items():
                lines.append(f"{metric}: {value:.6f}")
        
        report_path = self.output_dir / f"{report_name}.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        
        return report_path
    
    def _interpret_metric(self, metric: str, value: float) -> str:
        """Provide interpretation for metric values."""
        if np.isnan(value):
            return "Unable to compute"
        
        # Basic interpretations - would be more sophisticated in practice
        if metric.lower() == 'fid':
            if value < 10:
                return "Excellent quality"
            elif value < 30:
                return "Good quality"
            elif value < 50:
                return "Moderate quality"
            else:
                return "Poor quality"
        
        elif metric.lower() == 'ssim':
            if value > 0.9:
                return "Excellent similarity"
            elif value > 0.7:
                return "Good similarity"
            elif value > 0.5:
                return "Moderate similarity"
            else:
                return "Poor similarity"
        
        elif metric.lower() in ['mse', 'mae']:
            if value < 0.01:
                return "Very low error"
            elif value < 0.1:
                return "Low error"
            elif value < 0.5:
                return "Moderate error"
            else:
                return "High error"
        
        else:
            return "See documentation for interpretation"
    
    def export_metrics_csv(self, quality_results: Dict[str, Any], filename: str) -> str:
        """Export metrics to CSV file."""
        csv_path = self.output_dir / f"{filename}.csv"
        
        # Flatten all metrics into a single dictionary
        all_metrics = {}
        
        if 'metrics' in quality_results:
            all_metrics.update(quality_results['metrics'])
        
        if 'diversity' in quality_results:
            for key, value in quality_results['diversity'].items():
                all_metrics[f'diversity_{key}'] = value
        
        # Convert to DataFrame and save
        df = pd.DataFrame([all_metrics])
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Metrics exported to CSV: {csv_path}")
        return str(csv_path)