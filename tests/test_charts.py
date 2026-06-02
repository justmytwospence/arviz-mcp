"""Tests for Chart.js widget builders."""

import numpy as np
import pandas as pd

from arviz_mcp.charts import (
    comparison_chart,
    density_chart,
    energy_chart,
    forest_chart,
    ppc_chart,
    prior_posterior_chart,
    trace_chart,
)

CHART_JS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"


def _assert_valid_widget(result: dict) -> None:
    """Check that a chart builder returns valid widget output."""
    assert "text" in result
    assert "widget_html" in result
    html = result["widget_html"]
    assert isinstance(html, str)
    assert len(html) > 0
    # Must include Chart.js CDN
    assert CHART_JS_CDN in html
    # Must have a canvas element
    assert "<canvas" in html
    # Must not have full HTML document tags
    assert "<!DOCTYPE" not in html
    assert "<html" not in html
    assert "<head" not in html
    assert "<body" not in html
    # Must have sendPrompt follow-up buttons
    assert "sendPrompt" in html
    # Must disable Chart.js default legend
    assert "legend:" in html and "display: false" in html


def _make_summary_df(n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    means = rng.normal(0, 3, n)
    sds = rng.uniform(0.5, 2, n)
    return pd.DataFrame(
        {
            "mean": means,
            "sd": sds,
            "eti_3%": means - 2 * sds,
            "eti_97%": means + 2 * sds,
        },
        index=[f"param_{i}" for i in range(n)],
    )


class TestForestChart:
    def test_basic(self):
        df = _make_summary_df()
        result = forest_chart(df, var_name="test_model")
        _assert_valid_widget(result)
        assert "test_model" in result["text"]

    def test_single_param(self):
        df = _make_summary_df(1)
        result = forest_chart(df)
        _assert_valid_widget(result)

    def test_many_params(self):
        df = _make_summary_df(20)
        result = forest_chart(df)
        _assert_valid_widget(result)
        # Chart height should scale with number of bars
        assert "height: 880px" in result["widget_html"]


class TestDensityChart:
    def test_combined(self):
        rng = np.random.default_rng(42)
        result = density_chart(rng.normal(0, 1, 1000), var_name="mu")
        _assert_valid_widget(result)
        assert "mu" in result["text"]

    def test_per_chain(self):
        rng = np.random.default_rng(42)
        chains = {
            "Chain 0": rng.normal(0, 1, 500),
            "Chain 1": rng.normal(0, 1, 500),
        }
        result = density_chart(chains, var_name="mu")
        _assert_valid_widget(result)
        assert "Chain 0" in result["widget_html"]


class TestTraceChart:
    def test_basic(self):
        rng = np.random.default_rng(42)
        chains = {
            "Chain 0": rng.normal(0, 1, 500),
            "Chain 1": rng.normal(0, 1, 500),
        }
        result = trace_chart(chains, var_name="mu")
        _assert_valid_widget(result)
        assert "2 chains" in result["text"]

    def test_long_chains_subsampled(self):
        rng = np.random.default_rng(42)
        chains = {"Chain 0": rng.normal(0, 1, 5000)}
        result = trace_chart(chains)
        _assert_valid_widget(result)
        # HTML should be reasonable size (subsampled)
        assert len(result["widget_html"]) < 500_000


class TestPPCChart:
    def test_basic(self):
        rng = np.random.default_rng(42)
        observed = rng.normal(0, 1, 100)
        predicted = rng.normal(0, 1, (50, 100))
        result = ppc_chart(observed, predicted, var_name="y")
        _assert_valid_widget(result)
        assert "y" in result["text"]

    def test_single_prediction(self):
        rng = np.random.default_rng(42)
        observed = rng.normal(0, 1, 100)
        predicted = rng.normal(0, 1, (1, 100))
        result = ppc_chart(observed, predicted)
        _assert_valid_widget(result)


class TestPriorPosteriorChart:
    def test_basic(self):
        rng = np.random.default_rng(42)
        prior = rng.normal(0, 5, 1000)
        posterior = rng.normal(2, 1, 1000)
        result = prior_posterior_chart(prior, posterior, var_name="mu")
        _assert_valid_widget(result)
        assert "Prior" in result["widget_html"]
        assert "Posterior" in result["widget_html"]
        assert "substantially" in result["text"]


class TestComparisonChart:
    def test_basic(self):
        df = pd.DataFrame(
            {
                "elpd_loo": [-200.5, -210.3],
                "se": [10.1, 11.2],
                "weight": [0.8, 0.2],
            },
            index=["model_a", "model_b"],
        )
        result = comparison_chart(df)
        _assert_valid_widget(result)
        assert "model_a" in result["text"]

    def test_no_weights(self):
        df = pd.DataFrame(
            {
                "elpd_loo": [-200.5, -210.3],
                "se": [10.1, 11.2],
            },
            index=["model_a", "model_b"],
        )
        result = comparison_chart(df)
        _assert_valid_widget(result)


class TestEnergyChart:
    def test_basic(self):
        rng = np.random.default_rng(42)
        energy = rng.normal(100, 10, 2000)
        result = energy_chart(energy)
        _assert_valid_widget(result)
        assert "BFMI" in result["text"]

    def test_marginal_and_transition_differ(self):
        # Regression: the transition curve must be the lag-1 energy differences,
        # not the marginal energy shifted by a constant (which would make the two
        # curves identical and the diagnostic meaningless).
        from arviz_mcp.charts.energy import _energy_series

        rng = np.random.default_rng(0)
        energy = rng.normal(100, 10, 5000)
        series = _energy_series(energy)
        assert series["marginal_x"] != series["transition_x"]
        marginal_range = max(series["marginal_x"]) - min(series["marginal_x"])
        transition_range = max(series["transition_x"]) - min(series["transition_x"])
        # Differences of (near-)iid draws have ~sqrt(2)x the spread of the marginal.
        assert transition_range > marginal_range
