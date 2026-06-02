"""PPC chart: observed vs posterior predictive densities."""

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
    interaction_config,
    js_json,
    legend_html,
    legend_toggle_js,
    metric_cards,
    script_tag,
    tooltip_config,
    unique_id,
)


def _kde(values: np.ndarray, x_grid: np.ndarray) -> list:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return [0.0] * len(x_grid)
    kde = gaussian_kde(values)
    return np.round(kde(x_grid), 4).tolist()


def ppc_chart(
    observed: np.ndarray,
    predicted_samples: np.ndarray,
    var_name: str = "y",
    num_overlay: int = 20,
) -> dict:
    """Build a PPC overlay chart.

    Parameters
    ----------
    observed : 1D ndarray
        Observed data.
    predicted_samples : 2D ndarray
        Shape (n_samples, n_obs). Posterior predictive draws.
    var_name : str
        Variable name.
    num_overlay : int
        Number of predictive draws to overlay.
    """
    canvas_id = unique_id("ppc")
    chart_var = f"chart_{canvas_id}"
    observed = np.asarray(observed).ravel()
    predicted_samples = np.asarray(predicted_samples)

    if predicted_samples.ndim == 1:
        predicted_samples = predicted_samples.reshape(1, -1)

    # Common x grid
    all_vals = np.concatenate([observed, predicted_samples.ravel()])
    all_finite = all_vals[np.isfinite(all_vals)]
    x_grid = np.linspace(all_finite.min(), all_finite.max(), 200)
    x_list = np.round(x_grid, 4).tolist()

    # Observed KDE
    obs_y = _kde(observed, x_grid)

    datasets_js = []

    # Predictive sample overlays
    n_to_draw = min(num_overlay, len(predicted_samples))
    indices = np.random.default_rng(42).choice(len(predicted_samples), n_to_draw, replace=False)
    for idx in indices:
        pred_y = _kde(predicted_samples[idx], x_grid)
        datasets_js.append(
            {
                "label": "Predicted",
                "data": [{"x": xi, "y": yi} for xi, yi in zip(x_list, pred_y)],
                "borderColor": CHAIN_COLORS[0] + "40",
                "backgroundColor": "transparent",
                "pointRadius": 0,
                "pointHoverRadius": 0,
                "borderWidth": 1,
                "tension": 0.4,
            }
        )

    # Observed on top
    obs_dataset_idx = len(datasets_js)
    datasets_js.append(
        {
            "label": "Observed",
            "data": [{"x": xi, "y": yi} for xi, yi in zip(x_list, obs_y)],
            "borderColor": "#111111",
            "backgroundColor": "transparent",
            "pointRadius": 0,
            "pointHoverRadius": 3,
            "pointHoverBackgroundColor": "#111111",
            "borderWidth": 2.5,
            "tension": 0.4,
        }
    )

    obs_mean = round(float(np.mean(observed)), 2)
    pred_mean = round(float(np.mean(predicted_samples)), 2)

    cards = metric_cards(
        [
            {"label": "Observed mean", "value": str(obs_mean)},
            {"label": "Predicted mean", "value": str(pred_mean)},
            {"label": "Samples shown", "value": str(n_to_draw)},
        ]
    )

    legend_entries = [
        {"color": "#111111", "label": "Observed", "shape": "line", "datasetIndex": obs_dataset_idx},
        {
            "color": CHAIN_COLORS[0],
            "label": "Posterior predictive",
            "shape": "line",
            "datasetIndex": 0,
        },
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
                {
                    "label": "Poor fit?",
                    "prompt": "Are there specific observations the model fits poorly?",
                },
                {"label": "Compare models", "prompt": "Compare with an alternative model"},
            ]
        )
        + script_tag(js)
    )

    text = (
        f"PPC for {var_name}: observed mean = {obs_mean}, predicted mean = {pred_mean}. "
        f"Showing {n_to_draw} posterior predictive draws overlaid on observed density."
    )

    return {"text": text, "widget_html": widget_html}
