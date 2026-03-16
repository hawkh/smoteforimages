"""
Unit tests for SLERP-SMOTE core algorithm and EMA shadow parameter manager.
These test the patent's core mathematical claims.
"""
import math
import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from smote_image_synthesis.smote.constrained_smote import ConstrainedSMOTE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v):
    return v / np.linalg.norm(v)


def _slerp_ref(v0, v1, t):
    """Reference implementation of SLERP for test comparison."""
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    omega = math.acos(dot)
    if abs(omega) < 1e-6:
        lerp = (1 - t) * v0 + t * v1
        return lerp / np.linalg.norm(lerp)
    return (math.sin((1 - t) * omega) / math.sin(omega)) * v0 + \
           (math.sin(t * omega) / math.sin(omega)) * v1


# ---------------------------------------------------------------------------
# SLERP property tests
# ---------------------------------------------------------------------------

class TestSlerpProperties:
    """Verify the SLERP implementation satisfies the mathematical properties
    stated in the patent claims."""

    @pytest.fixture
    def smote(self):
        return ConstrainedSMOTE(k_neighbors=2, use_slerp=True, random_state=42)

    def test_unit_norm_output(self, smote):
        """SLERP of two unit-norm vectors must produce a unit-norm vector."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            D = rng.integers(8, 64)
            v0 = _unit(rng.standard_normal(D))
            v1 = _unit(rng.standard_normal(D))
            t = float(rng.uniform(0.0, 1.0))
            result = smote._slerp(v0, v1, t)
            assert abs(np.linalg.norm(result) - 1.0) < 1e-5, \
                f"SLERP output norm={np.linalg.norm(result):.6f}, expected 1.0"

    def test_boundary_t0_returns_v0(self, smote):
        """SLERP at t=0 must return v0."""
        rng = np.random.default_rng(1)
        v0 = _unit(rng.standard_normal(16))
        v1 = _unit(rng.standard_normal(16))
        result = smote._slerp(v0, v1, 0.0)
        np.testing.assert_allclose(result, v0, atol=1e-6,
            err_msg="SLERP(v0,v1,t=0) must equal v0")

    def test_boundary_t1_returns_v1(self, smote):
        """SLERP at t=1 must return v1."""
        rng = np.random.default_rng(2)
        v0 = _unit(rng.standard_normal(16))
        v1 = _unit(rng.standard_normal(16))
        result = smote._slerp(v0, v1, 1.0)
        np.testing.assert_allclose(result, v1, atol=1e-6,
            err_msg="SLERP(v0,v1,t=1) must equal v1")

    def test_symmetry(self, smote):
        """SLERP(v0,v1,t) == SLERP(v1,v0,1-t)."""
        rng = np.random.default_rng(3)
        for _ in range(10):
            v0 = _unit(rng.standard_normal(32))
            v1 = _unit(rng.standard_normal(32))
            t = float(rng.uniform(0.01, 0.99))
            r1 = smote._slerp(v0, v1, t)
            r2 = smote._slerp(v1, v0, 1.0 - t)
            np.testing.assert_allclose(r1, r2, atol=1e-5,
                err_msg=f"SLERP symmetry failed at t={t:.3f}")

    def test_matches_reference_formula(self, smote):
        """SLERP output must match the closed-form reference formula."""
        rng = np.random.default_rng(4)
        for _ in range(15):
            D = rng.integers(4, 64)
            v0 = _unit(rng.standard_normal(D))
            v1 = _unit(rng.standard_normal(D))
            t = float(rng.uniform(0.05, 0.95))
            result = smote._slerp(v0, v1, t)
            expected = _slerp_ref(v0, v1, t)
            np.testing.assert_allclose(result, expected, atol=1e-5,
                err_msg="SLERP does not match the patent formula")

    def test_near_collinear_fallback_is_unit_norm(self, smote):
        """When v0 ≈ v1 (angle < 1e-6), fallback must still produce unit norm."""
        rng = np.random.default_rng(5)
        v0 = _unit(rng.standard_normal(16))
        v1 = v0 + rng.standard_normal(16) * 1e-8  # nearly identical
        v1 = _unit(v1)
        t = 0.5
        result = smote._slerp(v0, v1, t)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5, \
            "Near-collinear fallback SLERP must produce unit-norm output"

    def test_midpoint_is_halfway_angle(self, smote):
        """SLERP at t=0.5 must lie exactly halfway (equal angles to v0 and v1)."""
        rng = np.random.default_rng(6)
        v0 = _unit(rng.standard_normal(8))
        v1 = _unit(rng.standard_normal(8))
        mid = smote._slerp(v0, v1, 0.5)
        angle_to_v0 = math.acos(np.clip(np.dot(mid, v0), -1, 1))
        angle_to_v1 = math.acos(np.clip(np.dot(mid, v1), -1, 1))
        assert abs(angle_to_v0 - angle_to_v1) < 1e-5, \
            f"Midpoint angles differ: {angle_to_v0:.6f} vs {angle_to_v1:.6f}"


# ---------------------------------------------------------------------------
# EMA tests
# ---------------------------------------------------------------------------

class TestEMA:
    """Verify the EMA shadow parameter manager (patent Section 4.6)."""

    @pytest.fixture
    def ema_cls(self):
        from smote_image_synthesis.pipeline import _EMA
        return _EMA

    def test_shadow_differs_after_update(self, ema_cls):
        """After update(), shadow parameters must differ from model if decay < 1."""
        model = torch.nn.Linear(4, 4)
        ema = ema_cls(model, decay=0.9999)
        # Modify model weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p) * 10.0)
        ema.update(model)
        # Shadow should be a weighted average, not equal to new weights
        for (name, shadow_p), p in zip(ema.shadow.items(), model.parameters()):
            assert not torch.allclose(shadow_p, p), \
                f"Shadow '{name}' must differ from model params after single update with high decay"

    def test_apply_substitutes_weights(self, ema_cls):
        """apply() must copy shadow parameters into the model."""
        model = torch.nn.Linear(4, 4)
        ema = ema_cls(model, decay=0.5)  # low decay so shadow differs quickly
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(5.0)
        ema.update(model)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)
        ema.apply(model)
        for name, p in model.named_parameters():
            np.testing.assert_allclose(
                p.detach().cpu().numpy(),
                ema.shadow[name].cpu().numpy(),
                atol=1e-6,
                err_msg=f"apply() did not copy shadow into model param '{name}'"
            )

    def test_restore_reverts_weights(self, ema_cls):
        """restore() must revert model to pre-apply weights."""
        model = torch.nn.Linear(4, 4)
        ema = ema_cls(model, decay=0.5)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(7.0)
        original_vals = {n: p.detach().clone() for n, p in model.named_parameters()}
        ema.update(model)
        backup = ema.apply(model)
        ema.restore(model, backup)
        for name, p in model.named_parameters():
            np.testing.assert_allclose(
                p.detach().cpu().numpy(),
                original_vals[name].cpu().numpy(),
                atol=1e-6,
                err_msg=f"restore() did not revert param '{name}'"
            )

    def test_decay_value_0_9999(self, ema_cls):
        """EMA with decay=0.9999 must use correct weighting."""
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.0)
        ema = ema_cls(model, decay=0.9999)
        # Shadow init should be 0.0
        assert torch.allclose(ema.shadow['weight'], torch.zeros(1, 1))
        # Update with weight=1.0
        with torch.no_grad():
            model.weight.fill_(1.0)
        ema.update(model)
        expected = 0.9999 * 0.0 + (1 - 0.9999) * 1.0  # = 0.0001
        actual = ema.shadow['weight'].item()
        assert abs(actual - expected) < 1e-7, \
            f"EMA shadow={actual:.7f}, expected {expected:.7f} for decay=0.9999"
