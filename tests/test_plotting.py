"""Tests for plotting tools."""

import matplotlib

matplotlib.use("Agg")

import arviz as az

from arviz_mcp.utils import get_model, plot_collection_to_image


def _assert_valid_image(pc):
    """Verify a PlotCollection converts to a valid PNG ImageContent."""
    img = plot_collection_to_image(pc)
    content = img.to_image_content()
    assert content.type == "image"
    assert content.mimeType == "image/png"
    assert len(content.data) > 0
    # Verify under 1MB (base64 is ~4/3 of raw bytes)
    raw_bytes = len(content.data) * 3 / 4
    assert raw_bytes < 1_000_000


def test_plot_trace(ctx_with_models):
    dt = get_model(ctx_with_models, "centered")
    pc = az.plot_trace(dt, var_names=["mu"])
    _assert_valid_image(pc)


def test_plot_forest(ctx_with_models):
    dt = get_model(ctx_with_models, "centered")
    pc = az.plot_forest(dt, var_names=["mu", "tau"])
    _assert_valid_image(pc)


def test_plot_energy(ctx_with_models):
    dt = get_model(ctx_with_models, "centered")
    pc = az.plot_energy(dt)
    _assert_valid_image(pc)


def test_plot_pair(ctx_with_models):
    dt = get_model(ctx_with_models, "centered")
    pc = az.plot_pair(dt, var_names=["mu", "tau"])
    _assert_valid_image(pc)


def test_plot_ppc_dist(ctx_with_models):
    dt = get_model(ctx_with_models, "centered")
    pc = az.plot_ppc_dist(dt)
    _assert_valid_image(pc)


def test_plot_loo_pit(ctx_with_models):
    dt = get_model(ctx_with_models, "centered")
    pc = az.plot_loo_pit(dt)
    _assert_valid_image(pc)


def test_plot_prior_posterior(ctx_with_models):
    dt = get_model(ctx_with_models, "centered")
    pc = az.plot_prior_posterior(dt, var_names=["mu"])
    _assert_valid_image(pc)
