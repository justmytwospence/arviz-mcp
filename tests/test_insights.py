"""Tests for key_findings insight tool."""

import arviz as az

from arviz_mcp.models import get_group_names
from arviz_mcp.tools.insights import KeyFindingsInput, _evidence_strength, _run_trust_assessment
from arviz_mcp.utils import get_group, get_model


def _call_key_findings(ctx, **kwargs):
    """Call the key_findings logic directly."""
    params = KeyFindingsInput(**kwargs)
    dt = get_model(ctx, params.model)
    groups = get_group_names(dt)

    trust = _run_trust_assessment(dt, groups)
    assert "level" in trust
    assert "detail" in trust

    findings = []
    if "posterior" in groups:
        posterior = get_group(dt, "posterior", params.model)
        hdi_data = az.hdi(dt, prob=params.ci_prob)
        hdi_ds = hdi_data.ds

        for var_name in posterior.data_vars:
            samples = posterior[var_name].values.flatten()
            if len(samples) == 0:
                continue

            if var_name in hdi_ds.data_vars:
                hdi_vals = hdi_ds[var_name].values
                if hdi_vals.ndim == 1 and len(hdi_vals) == 2:
                    findings.append(var_name)

    return trust, findings


class TestKeyFindings:
    def test_basic(self, ctx_with_models):
        trust, findings = _call_key_findings(ctx_with_models, model="centered")
        assert trust["level"] in ("reliable", "caution", "unreliable")
        assert len(findings) > 0

    def test_trust_assessment_structure(self, ctx_with_models):
        dt = get_model(ctx_with_models, "centered")
        groups = get_group_names(dt)
        trust = _run_trust_assessment(dt, groups)
        assert trust["level"] in ("reliable", "caution", "unreliable")
        assert len(trust["detail"]) > 0

    def test_findings_have_scalar_params(self, ctx_with_models):
        """Verify that scalar parameters like mu and tau are found."""
        _, findings = _call_key_findings(ctx_with_models, model="centered")
        assert "mu" in findings
        assert "tau" in findings

    def test_custom_ci_prob(self, ctx_with_models):
        trust, findings = _call_key_findings(ctx_with_models, model="centered", ci_prob=0.89)
        assert trust["level"] in ("reliable", "caution", "unreliable")


class TestEvidenceStrength:
    def test_very_strong(self):
        assert _evidence_strength(0.995) == "very strong"

    def test_strong(self):
        assert _evidence_strength(0.96) == "strong"

    def test_moderate(self):
        assert _evidence_strength(0.92) == "moderate"

    def test_weak(self):
        assert _evidence_strength(0.80) == "weak"

    def test_inconclusive(self):
        assert _evidence_strength(0.60) == "inconclusive"
