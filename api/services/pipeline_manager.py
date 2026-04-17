"""Pipeline lifecycle manager — singleton holding pipeline instances per run."""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from api.config import DATASETS_DIR, OUTPUTS_DIR, DEVICE


@dataclass
class RunState:
    """State for a single pipeline run."""
    run_id: str
    dataset_id: str
    config: dict[str, Any]
    status: str = "idle"  # idle | training | trained | generating | complete | error
    pipeline: Any = None  # SynthesisPipeline instance
    progress_queue: Optional[asyncio.Queue] = None
    history: list[dict] = field(default_factory=list)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    generation_result: Optional[dict] = None
    quality_result: Optional[dict] = None


class PipelineManager:
    """Singleton managing all pipeline runs."""

    _instance: Optional["PipelineManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._runs = {}
        return cls._instance

    def create_run(self, dataset_id: str, config: dict) -> RunState:
        """Create a new pipeline run and instantiate components."""
        run_id = uuid.uuid4().hex[:12]

        # Verify dataset exists
        dataset_path = DATASETS_DIR / dataset_id
        if not dataset_path.exists():
            raise ValueError(f"Dataset {dataset_id} not found")

        # Count classes
        classes = [d.name for d in sorted(dataset_path.iterdir()) if d.is_dir()]
        num_classes = len(classes)
        if num_classes < 2:
            raise ValueError(f"Need at least 2 classes, found {num_classes}")

        config["num_classes"] = num_classes
        config["class_names"] = classes

        # Lazy-import ML modules (heavy)
        from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder
        from smote_image_synthesis.decoders.dcgan_decoder import DCGANDecoder
        from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE
        from smote_image_synthesis.pipeline import SynthesisPipeline

        image_size = config.get("image_size", 64)
        embedding_dim = config.get("embedding_dim", 512)

        encoder = ResNetEncoder(
            architecture=config.get("architecture", "resnet18"),
            embedding_dim=embedding_dim,
            pretrained=config.get("pretrained", True),
            device=DEVICE,
        )

        decoder = DCGANDecoder(
            embedding_dim=embedding_dim,
            image_shape=(3, image_size, image_size),
            base_channels=config.get("base_channels", 256),
            num_classes=num_classes,
            class_embed_dim=config.get("class_embed_dim", 64),
            use_self_attention=config.get("use_self_attention", True),
            device=DEVICE,
        )

        smote = ConstrainedSMOTE(
            k_neighbors=config.get("k_neighbors", 5),
            use_slerp=config.get("use_slerp", True),
            use_vmf=config.get("use_vmf", False),
            vmf_concentration_scale=config.get("vmf_concentration_scale", 1.0),
            density_weighted_t=config.get("density_weighted_t", False),
            use_cluster_constraints=config.get("use_cluster_constraints", False),
            use_outlier_detection=config.get("use_outlier_detection", False),
            track_ancestry=config.get("track_ancestry", False),
        )

        pipeline = SynthesisPipeline(
            encoder=encoder,
            decoder=decoder,
            smote=smote,
        )

        state = RunState(
            run_id=run_id,
            dataset_id=dataset_id,
            config=config,
            pipeline=pipeline,
            progress_queue=asyncio.Queue(maxsize=500),
        )
        self._runs[run_id] = state

        # Save config to disk
        run_dir = OUTPUTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))

        return state

    def get_run(self, run_id: str) -> Optional[RunState]:
        return self._runs.get(run_id)

    def list_runs(self) -> list[dict]:
        return [
            {
                "run_id": s.run_id,
                "dataset_id": s.dataset_id,
                "status": s.status,
                "config": s.config,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(s.created_at)),
            }
            for s in self._runs.values()
        ]

    def delete_run(self, run_id: str) -> bool:
        if run_id in self._runs:
            del self._runs[run_id]
            return True
        return False


def get_manager() -> PipelineManager:
    """Get the global PipelineManager singleton."""
    return PipelineManager()
