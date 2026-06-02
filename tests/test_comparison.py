"""Tests for model comparison tools."""

import arviz as az

from arviz_mcp.utils import get_model


def test_compare(ctx_with_models):
    dt1 = get_model(ctx_with_models, "centered")
    dt2 = get_model(ctx_with_models, "non_centered")

    df = az.compare({"centered": dt1, "non_centered": dt2})
    assert "rank" in df.columns
    assert "elpd" in df.columns
    assert len(df) == 2


def test_loo(ctx_with_models):
    dt = get_model(ctx_with_models, "centered")
    loo = az.loo(dt)
    assert hasattr(loo, "elpd")
    assert hasattr(loo, "se")
    assert hasattr(loo, "p")
    assert loo.elpd < 0  # Log likelihood should be negative


def test_loo_pointwise(ctx_with_models):
    dt = get_model(ctx_with_models, "centered")
    loo = az.loo(dt, pointwise=True)
    assert hasattr(loo, "pareto_k")
