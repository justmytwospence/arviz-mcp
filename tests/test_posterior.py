"""Tests for posterior query tools."""

import arviz as az
import numpy as np
import pytest

from arviz_mcp.utils import get_group, get_model


def test_hdi(ctx_with_models):
    dt = get_model(ctx_with_models, "centered")
    hdi = az.hdi(dt, var_names=["mu"], prob=0.94)
    hdi_ds = hdi.ds
    assert "mu" in hdi_ds.data_vars
    vals = hdi_ds["mu"].values
    assert vals[0] < vals[1]  # lower < upper


def test_posterior_stat_mean(ctx_with_models):
    dt = get_model(ctx_with_models, "centered")
    post_ds = get_group(dt, "posterior", "centered")
    samples = post_ds["mu"].values.flatten()
    mean = float(np.mean(samples))
    assert -10 < mean < 20  # reasonable range for centered_eight mu


def test_posterior_stat_prob_above(ctx_with_models):
    dt = get_model(ctx_with_models, "centered")
    post_ds = get_group(dt, "posterior", "centered")
    samples = post_ds["mu"].values.flatten()
    prob = float(np.mean(samples > 0))
    assert 0 < prob < 1


def test_posterior_stat_quantile(ctx_with_models):
    dt = get_model(ctx_with_models, "centered")
    post_ds = get_group(dt, "posterior", "centered")
    samples = post_ds["mu"].values.flatten()
    q50 = float(np.quantile(samples, 0.5))
    mean = float(np.mean(samples))
    # Median and mean should be in the same ballpark
    assert abs(q50 - mean) < 5


def test_evidence_strength():
    from arviz_mcp.tools.posterior import _evidence_strength

    assert _evidence_strength(0.995) == "very strong"
    assert _evidence_strength(0.96) == "strong"
    assert _evidence_strength(0.92) == "moderate"
    assert _evidence_strength(0.80) == "weak"
    assert _evidence_strength(0.60) == "inconclusive"


def test_posterior_stat_quantile_requires_value(ctx_with_models, get_tool):
    """posterior_stat must raise (not return a string) when 'value' is missing."""
    from arviz_mcp.tools.posterior import PosteriorStatInput

    fn = get_tool("posterior_stat")
    with pytest.raises(ValueError):
        fn(
            PosteriorStatInput(model="centered", var_name="mu", stat="quantile"),
            ctx_with_models,
        )


def test_posterior_stat_unknown_stat_raises(ctx_with_models, get_tool):
    from arviz_mcp.tools.posterior import PosteriorStatInput

    fn = get_tool("posterior_stat")
    with pytest.raises(ValueError):
        fn(
            PosteriorStatInput(model="centered", var_name="mu", stat="bogus"),
            ctx_with_models,
        )


def test_compare_parameters(ctx_with_models):
    """Test comparing two parameters from the joint posterior."""
    dt = get_model(ctx_with_models, "centered")
    post_ds = get_group(dt, "posterior", "centered")

    samples_mu = post_ds["mu"].values.flatten()
    samples_tau = post_ds["tau"].values.flatten()
    n = min(len(samples_mu), len(samples_tau))
    diff = samples_mu[:n] - samples_tau[:n]

    prob_mu_gt = float(np.mean(diff > 0))
    prob_tau_gt = float(np.mean(diff < 0))

    # Probabilities should sum to ~1 (allowing for exact equality)
    assert abs(prob_mu_gt + prob_tau_gt + float(np.mean(diff == 0)) - 1.0) < 1e-10
    # Both should be valid probabilities
    assert 0 <= prob_mu_gt <= 1
    assert 0 <= prob_tau_gt <= 1
