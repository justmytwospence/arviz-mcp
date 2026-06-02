"""Prior vs posterior density comparison chart."""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde

from arviz_mcp.charts.templates import (
    CHAIN_COLORS,
    GRID_COLOR,
    animation_config,
    chart_container,
    crosshair_plugin_js,
    followup_buttons,
    hover_emphasis_config,
    interaction_config,
    js_json,
    legend_html,
    legend_toggle_js,
    metric_cards,
    script_tag,
    tooltip_config,
    unique_id,
)

PRIOR_COLOR = "#999999"


def _compute_kde(values: np.ndarray, n_points: int = 200) -> tuple[list, list]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return [], []
    kde = gaussian_kde(values)
    x = np.linspace(values.min(), values.max(), n_points)
    y = kde(x)
    return np.round(x, 4).tolist(), np.round(y, 4).tolist()


def prior_posterior_chart(
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    var_name: str = "parameter",
) -> dict:
    """Build a prior vs posterior density comparison.

    Parameters
    ----------
    prior_samples : 1D ndarray
        Prior draws.
    posterior_samples : 1D ndarray
        Posterior draws.
    var_name : str
        Variable name.
    """
    canvas_id = unique_id("pripost")
    chart_var = f"chart_{canvas_id}"
    prior_samples = np.asarray(prior_samples).ravel()
    posterior_samples = np.asarray(posterior_samples).ravel()

    prior_x, prior_y = _compute_kde(prior_samples)
    post_x, post_y = _compute_kde(posterior_samples)

    datasets_js = [
        {
            "label": "Prior",
            "data": [{"x": xi, "y": yi} for xi, yi in zip(prior_x, prior_y)],
            "borderColor": PRIOR_COLOR,
            "backgroundColor": PRIOR_COLOR + "26",
            "fill": True,
            "tension": 0.4,
            "pointRadius": 0,
            "pointHoverRadius": 3,
            "pointHoverBackgroundColor": PRIOR_COLOR,
            "borderWidth": 2,
            "borderDash": [6, 4],
        },
        {
            "label": "Posterior",
            "data": [{"x": xi, "y": yi} for xi, yi in zip(post_x, post_y)],
            "borderColor": CHAIN_COLORS[0],
            "backgroundColor": CHAIN_COLORS[0] + "26",
            "fill": True,
            "tension": 0.4,
            "pointRadius": 0,
            "pointHoverRadius": 3,
            "pointHoverBackgroundColor": CHAIN_COLORS[0],
            "borderWidth": 2,
        },
    ]

    prior_mean = round(float(np.mean(prior_samples)), 2)
    post_mean = round(float(np.mean(posterior_samples)), 2)
    post_std = round(float(np.std(posterior_samples)), 2)

    cards = metric_cards(
        [
            {"label": "Prior mean", "value": str(prior_mean)},
            {"label": "Posterior mean", "value": str(post_mean)},
            {"label": "Posterior SD", "value": str(post_std)},
        ]
    )

    legend_entries = [
        {"color": PRIOR_COLOR, "label": "Prior", "shape": "dashed", "datasetIndex": 0},
        {"color": CHAIN_COLORS[0], "label": "Posterior", "shape": "square", "datasetIndex": 1},
    ]

    js = f"""
var {chart_var} = new Chart(document.getElementById('{canvas_id}'), {{
  type: 'scatter',
  data: {{ datasets: {js_json(datasets_js)} }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    showLine: true,
    {animation_config()}
    {interaction_config()}
    {hover_emphasis_config()}
    scales: {{
      x: {{
        type: 'linear',
        title: {{ display: true, text: '{var_name}', font: {{ size: 12 }} }},
        grid: {{ color: '{GRID_COLOR}' }}
      }},
      y: {{
        title: {{ display: true, text: 'Density', font: {{ size: 12 }} }},
        grid: {{ color: '{GRID_COLOR}' }},
        beginAtZero: true
      }}
    }},
    plugins: {{
      legend: {{ display: false }},
      {tooltip_config()}
    }}
  }},
  plugins: [{crosshair_plugin_js()}]
}});
{legend_toggle_js(chart_var, canvas_id)}"""

    widget_html = (
        cards
        + chart_container(canvas_id, 360)
        + legend_html(legend_entries, chart_id=canvas_id)
        + followup_buttons(
            [
                {"label": "Run diagnostics", "prompt": "Run full diagnostics on this model"},
                {"label": "Forest plot", "prompt": "Show the forest plot for all parameters"},
            ]
        )
        + script_tag(js)
    )

    text = (
        f"Prior vs posterior for {var_name}: "
        f"prior mean = {prior_mean}, posterior mean = {post_mean} (SD = {post_std}). "
        f"The data {'substantially' if abs(post_mean - prior_mean) > post_std else 'modestly'} updated the prior."
    )

    return {"text": text, "widget_html": widget_html}
