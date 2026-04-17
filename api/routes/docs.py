"""Patent documentation routes — sections, equations, architecture."""

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.config import PATENT_MD, PATENT_TEX
from api.models.responses import PatentSection, EquationInfo

router = APIRouter(prefix="/api/docs", tags=["docs"])


@router.get("/patent", response_model=list[PatentSection])
async def get_patent_sections():
    """Parse patent disclosure markdown into sections."""
    if not PATENT_MD.exists():
        raise HTTPException(404, "Patent document not found")

    content = PATENT_MD.read_text(encoding="utf-8")
    sections = []
    current_id = ""
    current_title = ""
    current_lines = []

    for line in content.split("\n"):
        if line.startswith("## "):
            # Save previous section
            if current_title:
                sections.append(PatentSection(
                    id=current_id,
                    title=current_title,
                    content_md="\n".join(current_lines).strip(),
                ))
            current_title = line[3:].strip()
            current_id = re.sub(r"[^a-z0-9]+", "-", current_title.lower()).strip("-")
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_title:
        sections.append(PatentSection(
            id=current_id,
            title=current_title,
            content_md="\n".join(current_lines).strip(),
        ))

    return sections


@router.get("/equations", response_model=list[EquationInfo])
async def get_equations():
    """Return key equations for KaTeX rendering."""
    # Core equations from the patent — hand-curated for display
    equations = [
        EquationInfo(
            id="l2norm",
            name="L2 Normalization",
            latex=r"\mathbf{z} = \frac{\mathbf{p}}{\|\mathbf{p}\|_2}",
            description="Projects encoder output onto the unit hypersphere S^{D-1}",
        ),
        EquationInfo(
            id="slerp",
            name="Spherical Linear Interpolation (SLERP)",
            latex=r"\hat{\mathbf{z}}(t) = \frac{\sin\bigl((1-t)\omega\bigr)}{\sin\omega}\mathbf{z}_0 + \frac{\sin\bigl(t\omega\bigr)}{\sin\omega}\mathbf{z}_1",
            description="Interpolates along the geodesic arc on the unit hypersphere, preserving manifold structure",
        ),
        EquationInfo(
            id="omega",
            name="Inter-Embedding Angle",
            latex=r"\omega = \arccos(\mathbf{z}_0 \cdot \mathbf{z}_1)",
            description="Angle between two unit-norm embeddings, used as the arc length for SLERP",
        ),
        EquationInfo(
            id="vmf",
            name="von Mises-Fisher Distribution",
            latex=r"f_D(\mathbf{z};\, \boldsymbol{\mu},\, \kappa) = C_D(\kappa)\exp\!\bigl(\kappa\,\boldsymbol{\mu}^\top \mathbf{z}\bigr)",
            description="Natural distribution on the unit hypersphere; fitted per class for vMF-SMOTE sampling",
        ),
        EquationInfo(
            id="kappa_mle",
            name="vMF Concentration (MLE)",
            latex=r"\hat{\kappa} \approx \frac{\bar{R}(D - \bar{R}^2)}{1 - \bar{R}^2}",
            description="Banerjee et al. (2005) approximation for the vMF concentration parameter",
        ),
        EquationInfo(
            id="wgan_gp",
            name="Gradient Penalty",
            latex=r"\mathcal{L}_{\text{GP}} = \lambda_{\text{GP}}\,\mathbb{E}_{\hat{\mathbf{x}}}\!\left[\bigl(\|\nabla_{\hat{\mathbf{x}}} D_{\boldsymbol{\psi}}(\hat{\mathbf{x}}, y)\|_2 - 1\bigr)^2\right]",
            description="WGAN-GP gradient penalty enforcing the 1-Lipschitz constraint on the discriminator",
        ),
        EquationInfo(
            id="proj_disc",
            name="Projection Discriminator",
            latex=r"D(\mathbf{x}, y) = \mathbf{w}^\top \phi(\mathbf{x}) + \mathbf{V}[y]^\top \cdot \text{GAP}(\phi(\mathbf{x}))",
            description="Class-conditional scoring via inner product between class embedding and discriminator features",
        ),
        EquationInfo(
            id="spectral_norm",
            name="Spectral Normalisation",
            latex=r"\bar{\mathbf{W}} = \frac{\mathbf{W}}{\sigma(\mathbf{W})}",
            description="Per-layer Lipschitz-1 constraint via largest singular value normalization",
        ),
        EquationInfo(
            id="feature_matching",
            name="Multi-Scale Feature Matching",
            latex=r"\mathcal{L}_{\text{FM}} = \sum_{i=1}^{3} w_i \cdot \|\mathbf{f}_i(\mathbf{x}) - \mathbf{f}_i(\hat{\mathbf{x}})\|_1, \quad (w_1, w_2, w_3) = (0.1, 0.3, 0.6)",
            description="L1 feature discrepancy at three discriminator depths with progressive semantic weighting",
        ),
        EquationInfo(
            id="ema",
            name="Exponential Moving Average",
            latex=r"\tilde{\boldsymbol{\theta}}_t = \mu\,\tilde{\boldsymbol{\theta}}_{t-1} + (1 - \mu)\,\boldsymbol{\theta}_t, \quad \mu = 0.9999",
            description="Shadow parameter averaging for smoother inference; applied to both encoder and decoder",
        ),
        EquationInfo(
            id="fid",
            name="Frechet Inception Distance",
            latex=r"\text{FID} = \|\boldsymbol{\mu}_r - \boldsymbol{\mu}_s\|_2^2 + \text{tr}\!\left(\boldsymbol{\Sigma}_r + \boldsymbol{\Sigma}_s - 2(\boldsymbol{\Sigma}_r \boldsymbol{\Sigma}_s)^{1/2}\right)",
            description="Primary quality metric measuring distributional distance between real and synthetic images",
        ),
        EquationInfo(
            id="repulsion",
            name="Intra-Class Repulsion Loss",
            latex=r"\mathcal{L}_{\text{repulse}} = \frac{1}{K}\sum_{k=1}^{K}\frac{1}{|\mathcal{P}_k|}\sum_{(i,j) \in \mathcal{P}_k}\max(0,\; \mathbf{z}_i^\top\mathbf{z}_j - m)",
            description="Hinge-margin penalty preventing per-class mode collapse in the embedding space",
        ),
    ]
    return equations


@router.get("/architecture")
async def get_architecture():
    """Return structured architecture component descriptions."""
    return {
        "components": [
            {
                "name": "ResNet Encoder + L2 Head",
                "description": "Maps images to unit-norm embeddings on S^{D-1}",
                "type": "encoder",
                "connections": ["SLERP/vMF Oversampler"],
            },
            {
                "name": "SLERP/vMF Oversampler",
                "description": "Dual-mode synthesis: geodesic interpolation or distributional sampling",
                "type": "oversampler",
                "connections": ["Class-Conditional Decoder"],
            },
            {
                "name": "Class-Conditional Decoder",
                "description": "DCGAN with self-attention, spectral norm, and class embedding",
                "type": "decoder",
                "connections": ["Projection Discriminator", "EMA Manager"],
            },
            {
                "name": "Projection Discriminator",
                "description": "SN + GP hybrid with class-conditional projection scoring",
                "type": "discriminator",
                "connections": ["Adaptive Lambda Controller"],
            },
            {
                "name": "Adaptive Lambda Controller",
                "description": "EMA-smoothed Wasserstein monitor adjusting adversarial weight",
                "type": "controller",
                "connections": ["Class-Conditional Decoder"],
            },
            {
                "name": "EMA Manager",
                "description": "Shadow parameters for encoder + decoder (mu=0.9999)",
                "type": "ema",
                "connections": [],
            },
        ],
        "phases": [
            {"name": "Phase 1: Reconstruction", "epochs": "0 to 0.3T", "losses": ["MSE", "L1", "Perceptual"]},
            {"name": "Phase 2: Adversarial", "epochs": "0.3T to T", "losses": ["WGAN-GP", "Feature Matching", "Repulsion", "Reconstruction"]},
        ],
    }
