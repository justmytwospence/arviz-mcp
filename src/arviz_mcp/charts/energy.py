"""Energy diagnostic chart: marginal energy vs energy transition."""

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


def _compute_kde(values: np.ndarray, n_points: int = 200) -> tuple[list, list]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return [], []
    kde = gaussian_kde(values)
    x = np.linspace(values.min(), values.max(), n_points)
    y = kde(x)
    return np.round(x, 4).tolist(), np.round(y, 4).tolist()


def _energy_series(energy: np.ndarray) -> dict:
    """Compute KDE curves for the marginal-energy and energy-transition distributions.

    Mirrors ArviZ's BFMI energy plot: the marginal energy is mean-centered and the
    transition distribution is the lag-1 differences of the energy. Centering both
    around zero is what makes the overlap between the two curves meaningful.
    """
    energy = np.asarray(energy).ravel()
    marginal_x, marginal_y = _compute_kde(energy - energy.mean())
    transition_x, transition_y = _compute_kde(np.diff(energy))
    return {
        "marginal_x": marginal_x,
        "marginal_y": marginal_y,
        "transition_x": transition_x,
        "transition_y": transition_y,
    }


def energy_chart(
    energy: np.ndarray,
) -> dict:
    """Build an energy diagnostic chart.

    Parameters
    ----------
    energy : 1D ndarray
        Energy values from sample_stats.
    """
    canvas_id = unique_id("energy")
    chart_var = f"chart_{canvas_id}"
    energy = np.asarray(energy).ravel()

    series = _energy_series(energy)
    marginal_x, marginal_y = series["marginal_x"], series["marginal_y"]
    transition_x, transition_y = series["transition_x"], series["transition_y"]

    # BFMI approximation: var of energy transitions over var of marginal energy
    bfmi = round(float(np.var(np.diff(energy)) / np.var(energy)), 3)

    datasets_js = [
        {
            "label": "Marginal energy",
            "data": [{"x": xi, "y": yi} for xi, yi in zip(marginal_x, marginal_y)],
            "borderColor": CHAIN_COLORS[0],
            "backgroundColor": CHAIN_COLORS[0] + "26",
            "fill": True,
            "tension": 0.4,
            "pointRadius": 0,
            "pointHoverRadius": 3,
            "pointHoverBackgroundColor": CHAIN_COLORS[0],
            "borderWidth": 2,
        },
        {
            "label": "Energy transition",
            "data": [{"x": xi, "y": yi} for xi, yi in zip(transition_x, transition_y)],
            "borderColor": CHAIN_COLORS[1],
            "backgroundColor": CHAIN_COLORS[1] + "26",
            "fill": True,
            "tension": 0.4,
            "pointRadius": 0,
            "pointHoverRadius": 3,
            "pointHoverBackgroundColor": CHAIN_COLORS[1],
            "borderWidth": 2,
        },
    ]

    bfmi_status = "good" if bfmi > 0.3 else "low (potential issues)"

    cards = metric_cards(
        [
            {"label": "BFMI", "value": str(bfmi), "sub": bfmi_status},
            {"label": "Energy mean", "value": str(round(float(energy.mean()), 1))},
            {"label": "Energy std", "value": str(round(float(energy.std()), 1))},
        ]
    )

    legend_entries = [
        {
            "color": CHAIN_COLORS[0],
            "label": "Marginal energy",
            "shape": "square",
            "datasetIndex": 0,
        },
        {
            "color": CHAIN_COLORS[1],
            "label": "Energy transition",
            "shape": "square",
            "datasetIndex": 1,
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
    {hover_emphasis_config()}
    scales: {{
      x: {{
        type: 'linear',
        title: {{ display: true, text: 'Energy (centered)', font: {{ size: 12 }} }},
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
                {"label": "Trace plot", "prompt": "Show the trace plot to check convergence"},
            ]
        )
        + script_tag(js)
    )

    text = (
        f"Energy diagnostic: BFMI = {bfmi} ({bfmi_status}). "
        f"When the two distributions overlap well, the sampler is exploring efficiently."
    )

    return {"text": text, "widget_html": widget_html}
