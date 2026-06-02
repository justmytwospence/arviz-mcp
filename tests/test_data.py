"""Tests for raw data access tool."""

from arviz_mcp.utils import get_group, get_model


def test_get_values_basic(ctx_with_models):
    dt = get_model(ctx_with_models, "centered")
    ds = get_group(dt, "posterior", "centered")
    var = ds["mu"]
    assert "chain" in var.dims
    assert "draw" in var.dims


def test_get_values_with_sel(ctx_with_models):
    dt = get_model(ctx_with_models, "centered")
    ds = get_group(dt, "posterior", "centered")
    var = ds["theta"].sel(school="Choate")
    assert "school" not in var.dims
    assert var.shape == (4, 500)  # chain x draw


def test_get_values_with_stat(ctx_with_models):
    dt = get_model(ctx_with_models, "centered")
    ds = get_group(dt, "posterior", "centered")
    var = ds["theta"]
    mean = var.mean(dim=["chain", "draw"]).values
    assert mean.shape == (8,)  # 8 schools


def test_get_values_observed_data(ctx_with_models):
    dt = get_model(ctx_with_models, "centered")
    ds = get_group(dt, "observed_data", "centered")
    assert "obs" in ds.data_vars
