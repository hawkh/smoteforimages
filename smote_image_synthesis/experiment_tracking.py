"""
Comprehensive experiment tracking and logging system for SMOTE image synthesis.
"""

import os
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import uuid
import hashlib

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetadata:
    """Metadata for tracking experiments."""
    experiment_id: str
    experiment_name: str
    description: str
    creation_timestamp: str
    status: str
    config_hash: str
    tags: List[str]
    author: Optional[str] = None


@dataclass
class ExperimentRun:
    """Individual experiment run information."""
    run_id: str
    experiment_id: str
    start_time: str
    end_time: Optional[str]
    status: str
    config: Dict[str, Any]
    metrics: Dict[str, List[float]]
    artifacts: List[str]


class ExperimentTracker:
    """
    Comprehensive experiment tracking system.
    
    Features:
    - Experiment lifecycle management
    - Metrics and artifact tracking
    - Configuration versioning
    - Experiment comparison
    - Visualization and reporting
    """
    
    def __init__(
        self,
        base_dir: Union[str, Path] = "./experiments",
        tensorboard_dir: Optional[Union[str, Path]] = None,
        auto_save_interval: int = 10
    ):
        """Initialize experiment tracker."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.tensorboard_dir = Path(tensorboard_dir) if tensorboard_dir else self.base_dir / "tensorboard"
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_save_interval = auto_save_interval
        
        # Storage
        self.experiments_db = self.base_dir / "experiments.json"
        self.runs_db = self.base_dir / "runs.json"
        
        # Current state
        self.current_experiment: Optional[ExperimentMetadata] = None
        self.current_run: Optional[ExperimentRun] = None
        self.tensorboard_writer: Optional[SummaryWriter] = None
        
        # Metrics buffer
        self.metrics_buffer: Dict[str, List[float]] = {}
        self.step_counter = 0
        
        # Load existing data
        self._load_databases()
        
        logger.info(f"ExperimentTracker initialized with base_dir: {self.base_dir}")
    
    def _load_databases(self) -> None:
        """Load experiment databases."""
        if self.experiments_db.exists():
            with open(self.experiments_db, 'r') as f:
                self.experiments = json.load(f)
        else:
            self.experiments = {}
        
        if self.runs_db.exists():
            with open(self.runs_db, 'r') as f:
                self.runs = json.load(f)
        else:
            self.runs = {}
    
    def _save_databases(self) -> None:
        """Save experiment databases."""
        with open(self.experiments_db, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)
        
        with open(self.runs_db, 'w') as f:
            json.dump(self.runs, f, indent=2, default=str)
    
    def create_experiment(
        self,
        name: str,
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        author: Optional[str] = None
    ) -> str:
        """Create a new experiment."""
        experiment_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Calculate config hash
        config = config or {}
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        # Create experiment metadata
        experiment = ExperimentMetadata(
            experiment_id=experiment_id,
            experiment_name=name,
            description=description,
            creation_timestamp=timestamp,
            status="created",
            config_hash=config_hash,
            tags=tags or [],
            author=author
        )
        
        # Store experiment
        self.experiments[experiment_id] = asdict(experiment)
        
        # Create experiment directory
        exp_dir = self.base_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # Save config if provided
        if config:
            with open(exp_dir / "config.json", 'w') as f:
                json.dump(config, f, indent=2, default=str)
        
        self._save_databases()
        self.current_experiment = experiment
        
        logger.info(f"Created experiment '{name}' with ID: {experiment_id}")
        return experiment_id
    
    def start_run(
        self,
        experiment_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None
    ) -> str:
        """Start a new experiment run."""
        if experiment_id is None:
            if self.current_experiment is None:
                raise ValueError("No current experiment set. Create or set an experiment first.")
            experiment_id = self.current_experiment.experiment_id
        
        run_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Create run
        run = ExperimentRun(
            run_id=run_id,
            experiment_id=experiment_id,
            start_time=timestamp,
            end_time=None,
            status="running",
            config=config or {},
            metrics={},
            artifacts=[]
        )
        
        # Store run
        self.runs[run_id] = asdict(run)
        self.current_run = run
        
        # Initialize TensorBoard writer
        if run_name:
            tb_dir = self.tensorboard_dir / f"{experiment_id}_{run_name}_{run_id[:8]}"
        else:
            tb_dir = self.tensorboard_dir / f"{experiment_id}_{run_id[:8]}"
        
        self.tensorboard_writer = SummaryWriter(str(tb_dir))
        
        # Reset metrics buffer
        self.metrics_buffer = {}
        self.step_counter = 0
        
        # Update experiment status
        self.experiments[experiment_id]["status"] = "running"
        
        self._save_databases()
        
        logger.info(f"Started run {run_id} for experiment {experiment_id}")
        return run_id
    
    def log_metric(
        self,
        metric_name: str,
        value: float,
        step: Optional[int] = None,
        commit: bool = True
    ) -> None:
        """Log a metric value."""
        if self.current_run is None:
            raise ValueError("No active run. Start a run first.")
        
        step = step or self.step_counter
        
        # Add to buffer
        if metric_name not in self.metrics_buffer:
            self.metrics_buffer[metric_name] = []
        
        self.metrics_buffer[metric_name].append(value)
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(metric_name, value, step)
        
        # Auto-save
        if commit and step % self.auto_save_interval == 0:
            self._save_metrics()
        
        self.step_counter = max(self.step_counter, step + 1)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        commit: bool = True
    ) -> None:
        """Log multiple metrics at once."""
        step = step or self.step_counter
        
        for metric_name, value in metrics.items():
            self.log_metric(metric_name, value, step, commit=False)
        
        if commit:
            self._save_metrics()
        
        self.step_counter = step + 1
    
    def _save_metrics(self) -> None:
        """Save metrics to database."""
        if self.current_run is None:
            return
        
        # Update run metrics
        run_id = self.current_run.run_id
        self.runs[run_id]["metrics"] = self.metrics_buffer.copy()
        
        # Save to file
        run_dir = self.base_dir / self.current_run.experiment_id / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        with open(run_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics_buffer, f, indent=2, default=str)
        
        # Flush TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.flush()
    
    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        artifact_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log an artifact (model, data, etc.)."""
        if self.current_run is None:
            raise ValueError("No active run. Start a run first.")
        
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {artifact_path}")
        
        # Copy artifact to experiment directory
        run_dir = self.base_dir / self.current_run.experiment_id / "runs" / self.current_run.run_id
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        artifact_id = str(uuid.uuid4())
        artifact_filename = f"{artifact_id}_{artifact_path.name}"
        dest_path = artifacts_dir / artifact_filename
        
        import shutil
        shutil.copy2(artifact_path, dest_path)
        
        # Store artifact reference
        self.current_run.artifacts.append(str(dest_path))
        
        # Update run
        run_id = self.current_run.run_id
        self.runs[run_id]["artifacts"] = self.current_run.artifacts
        
        self._save_databases()
        
        logger.info(f"Logged artifact {artifact_type}: {artifact_id}")
        return artifact_id
    
    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard."""
        if self.tensorboard_writer and self.current_run:
            # Convert numpy types to Python types
            converted_hparams = {}
            for key, value in hparams.items():
                if isinstance(value, np.ndarray):
                    converted_hparams[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    converted_hparams[key] = value.item()
                else:
                    converted_hparams[key] = value
            
            self.tensorboard_writer.add_hparams(converted_hparams, {})
    
    def log_image(
        self,
        tag: str,
        image: Union[np.ndarray, torch.Tensor],
        step: Optional[int] = None
    ) -> None:
        """Log an image to TensorBoard."""
        if self.tensorboard_writer:
            step = step or self.step_counter
            
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu()
            
            self.tensorboard_writer.add_image(tag, image, step)
    
    def log_figure(
        self,
        tag: str,
        figure: plt.Figure,
        step: Optional[int] = None,
        close: bool = True
    ) -> None:
        """Log a matplotlib figure to TensorBoard."""
        if self.tensorboard_writer:
            step = step or self.step_counter
            self.tensorboard_writer.add_figure(tag, figure, step)
            
            if close:
                plt.close(figure)
    
    def end_run(
        self,
        status: str = "completed",
        error_message: Optional[str] = None
    ) -> None:
        """End the current run."""
        if self.current_run is None:
            logger.warning("No active run to end")
            return
        
        # Update run
        end_time = datetime.now().isoformat()
        run_id = self.current_run.run_id
        
        self.runs[run_id]["end_time"] = end_time
        self.runs[run_id]["status"] = status
        
        if error_message:
            self.runs[run_id]["error_message"] = error_message
        
        # Save final metrics
        self._save_metrics()
        
        # Close TensorBoard writer
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
            self.tensorboard_writer = None
        
        # Update experiment status
        experiment_id = self.current_run.experiment_id
        experiment_runs = [r for r in self.runs.values() if r["experiment_id"] == experiment_id]
        if all(r["status"] in ["completed", "failed"] for r in experiment_runs):
            self.experiments[experiment_id]["status"] = "completed"
        
        self._save_databases()
        
        logger.info(f"Ended run {run_id} with status: {status}")
        self.current_run = None
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by ID."""
        return self.experiments.get(experiment_id)
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run by ID."""
        return self.runs.get(run_id)
    
    def list_experiments(
        self,
        status_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List experiments with optional filtering."""
        experiments = list(self.experiments.values())
        
        if status_filter:
            experiments = [e for e in experiments if e["status"] == status_filter]
        
        if tag_filter:
            experiments = [
                e for e in experiments 
                if any(tag in e["tags"] for tag in tag_filter)
            ]
        
        return sorted(experiments, key=lambda x: x["creation_timestamp"], reverse=True)
    
    def list_runs(self, experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List runs, optionally filtered by experiment."""
        runs = list(self.runs.values())
        
        if experiment_id:
            runs = [r for r in runs if r["experiment_id"] == experiment_id]
        
        return sorted(runs, key=lambda x: x["start_time"], reverse=True)
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compare multiple runs."""
        comparison_data = []
        
        for run_id in run_ids:
            run = self.get_run(run_id)
            if not run:
                logger.warning(f"Run {run_id} not found")
                continue
            
            row = {
                "run_id": run_id,
                "experiment_id": run["experiment_id"],
                "start_time": run["start_time"],
                "status": run["status"]
            }
            
            # Add metrics
            run_metrics = run.get("metrics", {})
            if metrics:
                for metric in metrics:
                    if metric in run_metrics:
                        values = run_metrics[metric]
                        row[f"{metric}_final"] = values[-1] if values else None
                        row[f"{metric}_best"] = max(values) if values else None
                        row[f"{metric}_avg"] = np.mean(values) if values else None
            else:
                # Add all metrics
                for metric, values in run_metrics.items():
                    if values:
                        row[f"{metric}_final"] = values[-1]
                        row[f"{metric}_best"] = max(values)
                        row[f"{metric}_avg"] = np.mean(values)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def plot_metric_comparison(
        self,
        run_ids: List[str],
        metric_name: str,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot metric comparison across runs."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for run_id in run_ids:
            run = self.get_run(run_id)
            if not run or metric_name not in run.get("metrics", {}):
                continue
            
            values = run["metrics"][metric_name]
            steps = list(range(len(values)))
            
            ax.plot(steps, values, label=f"Run {run_id[:8]}", linewidth=2)
        
        ax.set_xlabel("Step")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Metric Comparison: {metric_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_experiment_report(
        self,
        experiment_id: str,
        output_path: Optional[str] = None
    ) -> str:
        """Generate a comprehensive experiment report."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        runs = self.list_runs(experiment_id)
        
        # Generate report
        report_lines = [
            f"# Experiment Report: {experiment['experiment_name']}",
            f"**ID:** {experiment_id}",
            f"**Description:** {experiment['description']}",
            f"**Created:** {experiment['creation_timestamp']}",
            f"**Status:** {experiment['status']}",
            f"**Tags:** {', '.join(experiment['tags'])}",
            "",
            "## Runs Summary",
            f"Total runs: {len(runs)}",
            ""
        ]
        
        # Run details
        for i, run in enumerate(runs):
            report_lines.extend([
                f"### Run {i+1}: {run['run_id'][:8]}",
                f"- Start: {run['start_time']}",
                f"- Status: {run['status']}",
                ""
            ])
            
            # Metrics summary
            if run.get("metrics"):
                report_lines.append("**Final Metrics:**")
                for metric, values in run["metrics"].items():
                    if values:
                        report_lines.append(f"- {metric}: {values[-1]:.4f}")
                report_lines.append("")
        
        # Save report
        if not output_path:
            output_path = self.base_dir / experiment_id / "experiment_report.md"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"Experiment report saved to: {output_path}")
        return str(output_path)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.current_run:
            if exc_type:
                self.end_run("failed", str(exc_val))
            else:
                self.end_run("completed")
        
        if self.tensorboard_writer:
            self.tensorboard_writer.close()


# Convenience functions for easy integration
def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """Set up comprehensive logging configuration."""
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


class ExperimentContext:
    """Context manager for experiment tracking."""
    
    def __init__(
        self,
        tracker: ExperimentTracker,
        experiment_name: str,
        config: Dict[str, Any],
        description: str = "",
        tags: Optional[List[str]] = None
    ):
        self.tracker = tracker
        self.experiment_name = experiment_name
        self.config = config
        self.description = description
        self.tags = tags
        self.experiment_id = None
        self.run_id = None
    
    def __enter__(self):
        self.experiment_id = self.tracker.create_experiment(
            name=self.experiment_name,
            description=self.description,
            config=self.config,
            tags=self.tags
        )
        self.run_id = self.tracker.start_run()
        self.tracker.log_hyperparameters(self.config)
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.tracker.end_run("failed", str(exc_val))
        else:
            self.tracker.end_run("completed")