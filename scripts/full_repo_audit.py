#!/usr/bin/env python3
"""Run a deep repository audit and end-to-end pipeline smoke tests."""

from __future__ import annotations

import argparse
import ast
import compileall
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_COMMAND = [
    sys.executable,
    "-m",
    "pytest",
    "-q",
    "tests",
    "test_basic_structure.py",
]
CORE_IMPORTS = [
    "numpy",
    "torch",
    "torchvision",
    "sklearn",
    "imblearn",
    "PIL",
    "matplotlib",
    "scipy",
    "pandas",
    "seaborn",
    "smote_image_synthesis",
]
OPTIONAL_GAP_PREFIXES = {"mcp"}


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def run_command(cmd: list[str], timeout: int = 900) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def parse_requirements(path: Path) -> set[str]:
    requirements = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        for splitter in ("==", ">=", "<=", "~=", ">", "<"):
            if splitter in cleaned:
                cleaned = cleaned.split(splitter, 1)[0].strip()
                break
        requirements.add(cleaned.lower().replace("_", "-"))
    return requirements


def normalize_import(name: str) -> str:
    mapping = {
        "pil": "pillow",
        "sklearn": "scikit-learn",
        "cv2": "opencv-python",
    }
    lowered = name.lower().replace("_", "-")
    return mapping.get(lowered, lowered)


def discover_imports(paths: Iterable[Path]) -> set[str]:
    discovered: set[str] = set()
    for file_path in paths:
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    discovered.add(alias.name.split(".", 1)[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                discovered.add(node.module.split(".", 1)[0])
    return discovered


def run_pipeline_smoke_test() -> tuple[bool, str]:
    smoke_code = r'''
import numpy as np
import torch
from smote_image_synthesis.encoders.resnet_encoder import ResNetEncoder
from smote_image_synthesis.decoders.autoencoder_decoder import AutoencoderDecoder
from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE
from smote_image_synthesis.quality.assessor import QualityAssessor
from smote_image_synthesis.pipeline import SynthesisPipeline

torch.manual_seed(7)
np.random.seed(7)

images = torch.rand(12, 3, 64, 64)
labels = np.array([0] * 8 + [1] * 4)

encoder = ResNetEncoder(architecture='resnet18', embedding_dim=128, pretrained=False, device=torch.device('cpu'))
decoder = AutoencoderDecoder(embedding_dim=128, image_shape=(3, 64, 64), device=torch.device('cpu'))
smote = ConstrainedSMOTE(k_neighbors=3, use_clustering=False, normalize_embeddings=False)
assessor = QualityAssessor(metrics=['mse', 'mae'], compute_diversity=True, device=torch.device('cpu'))

pipeline = SynthesisPipeline(encoder=encoder, decoder=decoder, smote=smote, quality_assessor=assessor)
pipeline.fit(images, labels, train_decoder=False)
synthetic_images, synthetic_labels = pipeline.generate_synthetic_images(4)

assert synthetic_images.shape[0] == len(synthetic_labels)
assert synthetic_images.ndim == 4
results = pipeline.evaluate_quality(synthetic_images[:2], images[:2])
assert 'metrics' in results and 'mse' in results['metrics']
print('smoke-test-ok')
'''
    code, stdout, stderr = run_command([sys.executable, "-c", smoke_code], timeout=1200)
    if code == 0 and "smoke-test-ok" in stdout:
        return True, "Pipeline fit/generate/evaluate smoke test passed."
    combined = "\n".join(part for part in [stdout, stderr] if part)
    return False, combined or "Unknown failure running smoke test."


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit this repository and run full pipeline checks.")
    parser.add_argument(
        "--output",
        default="audit_reports/latest_audit.md",
        help="Path for the markdown audit report (relative to repo root).",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest execution (useful in restricted or long-running environments).",
    )
    args = parser.parse_args()

    checks: list[CheckResult] = []

    py_files = [p for p in REPO_ROOT.rglob("*.py") if ".git" not in p.parts]
    has_compilation_errors = not compileall.compile_dir(REPO_ROOT, quiet=1, force=False)
    if has_compilation_errors:
        checks.append(CheckResult("Python syntax compilation", "FAIL", "compileall reported errors."))
    else:
        checks.append(CheckResult("Python syntax compilation", "PASS", f"Compiled {len(py_files)} Python files."))

    missing_imports = []
    for module_name in CORE_IMPORTS:
        code, _, stderr = run_command([sys.executable, "-c", f"import {module_name}"])
        if code != 0:
            missing_imports.append(f"{module_name}: {stderr.splitlines()[-1] if stderr else 'Import failed'}")
    if missing_imports:
        checks.append(CheckResult("Core import validation", "FAIL", "; ".join(missing_imports)))
    else:
        checks.append(CheckResult("Core import validation", "PASS", "All core modules import successfully."))

    requirements = parse_requirements(REPO_ROOT / "requirements.txt")
    imports = {normalize_import(name) for name in discover_imports(py_files)}
    stdlib_and_internal = {
        "os", "sys", "json", "math", "pathlib", "typing", "dataclasses", "tempfile", "unittest",
        "subprocess", "logging", "time", "datetime", "argparse", "collections", "itertools", "random",
        "warnings", "functools", "abc", "io", "base64", "re", "shutil", "traceback", "hashlib",
        "inspect", "copy", "threading", "queue", "asyncio",
    }
    stdlib_and_internal.update({"smote_image_synthesis", "tests"})
    missing_from_requirements = sorted(
        name for name in imports
        if name not in requirements
        and name not in stdlib_and_internal
        and not any(name.startswith(prefix) for prefix in OPTIONAL_GAP_PREFIXES)
    )
    if missing_from_requirements:
        checks.append(
            CheckResult(
                "Dependency declaration audit",
                "WARN",
                "Potential undeclared dependencies: " + ", ".join(missing_from_requirements),
            )
        )
    else:
        checks.append(CheckResult("Dependency declaration audit", "PASS", "All discovered imports map to requirements."))

    if args.skip_tests:
        checks.append(CheckResult("Automated tests", "WARN", "Skipped via --skip-tests."))
    else:
        code, stdout, stderr = run_command(DEFAULT_TEST_COMMAND, timeout=1800)
        if code == 0:
            checks.append(CheckResult("Automated tests", "PASS", stdout.splitlines()[-1] if stdout else "Pytest completed."))
        else:
            detail = "\n".join(part for part in [stdout, stderr] if part)
            checks.append(CheckResult("Automated tests", "FAIL", detail[:3000]))

    smoke_ok, smoke_detail = run_pipeline_smoke_test()
    checks.append(CheckResult("Pipeline smoke test", "PASS" if smoke_ok else "FAIL", smoke_detail))

    git_code, git_stdout, _ = run_command(["git", "ls-files"])
    tracked = git_stdout.splitlines() if git_code == 0 else []
    pycache_tracked = [f for f in tracked if "__pycache__/" in f or f.endswith(".pyc")]
    if pycache_tracked:
        checks.append(CheckResult("Repository hygiene", "WARN", f"Tracked cache artifacts: {len(pycache_tracked)} files."))
    else:
        checks.append(CheckResult("Repository hygiene", "PASS", "No tracked __pycache__/.pyc artifacts detected."))

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pass_count = sum(1 for c in checks if c.status == "PASS")
    warn_count = sum(1 for c in checks if c.status == "WARN")
    fail_count = sum(1 for c in checks if c.status == "FAIL")

    lines = [
        "# Repository Audit Report",
        "",
        f"Generated: {timestamp}",
        f"Repository: `{REPO_ROOT}`",
        "",
        "## Summary",
        f"- PASS: **{pass_count}**",
        f"- WARN: **{warn_count}**",
        f"- FAIL: **{fail_count}**",
        "",
        "## Check Results",
    ]

    for check in checks:
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[check.status]
        lines.append(f"- {icon} **{check.name}** ({check.status})")
        lines.append(f"  - {check.detail}")

    lines.extend([
        "",
        "## Next-Step Plan",
        "1. Resolve all FAIL checks first (environment/dependency/test breakages).",
        "2. Resolve WARN checks to improve reproducibility and deployment safety.",
        "3. Run this audit in CI on every PR: `python scripts/full_repo_audit.py`.",
        "4. Treat this report as release gating: no FAILs before shipping.",
        "",
        "## Machine-Readable Results",
        "```json",
        json.dumps([check.__dict__ for check in checks], indent=2),
        "```",
    ])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Audit report written to: {output_path}")
    print(f"PASS={pass_count} WARN={warn_count} FAIL={fail_count}")

    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
