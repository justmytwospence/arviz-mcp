"""Posterior density chart: KDE line with area fill, optionally per chain."""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde

from arviz_mcp.charts.templates import (
    CHAIN_COLORS,
    GRID_COLOR,
    ZERO_LINE_COLOR,
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


def density_chart(
    samples: np.ndarray | dict[str, np.ndarray],
    var_name: str = "parameter",
    show_zero_line: bool = True,
) -> dict:
    """Build a posterior density (KDE) chart.

    Parameters
    ----------
    samples : ndarray or dict
        If ndarray, a single combined density is drawn.
        If dict mapping chain name to ndarray, per-chain densities are drawn.
    var_name : str
        Variable name for labels.
    show_zero_line : bool
        Whether to draw a vertical zero reference line.
    """
    canvas_id = unique_id("density")
    chart_var = f"chart_{canvas_id}"
    datasets_js = []
    legend_entries = []

    if isinstance(samples, dict):
        all_values = np.concatenate(list(samples.values()))
        for i, (chain_name, chain_vals) in enumerate(samples.items()):
            color = CHAIN_COLORS[i % len(CHAIN_COLORS)]
            x, y = _compute_kde(chain_vals)
            datasets_js.append(
                {
                    "label": chain_name,
                    "data": [{"x": xi, "y": yi} for xi, yi in zip(x, y)],
                    "borderColor": color,
                    "backgroundColor": color.replace(")", ", 0.15)").replace("rgb", "rgba")
                    if "rgb" in color
                    else color + "26",
                    "fill": True,
                    "tension": 0.4,
                    "pointRadius": 0,
                    "pointHoverRadius": 3,
                    "pointHoverBackgroundColor": color,
                    "borderWidth": 2,
                }
            )
            legend_entries.append(
                {"color": color, "label": chain_name, "shape": "square", "datasetIndex": i}
            )
    else:
        all_values = np.asarray(samples).ravel()
        color = CHAIN_COLORS[0]
        x, y = _compute_kde(all_values)
        datasets_js.append(
            {
                "label": var_name,
                "data": [{"x": xi, "y": yi} for xi, yi in zip(x, y)],
                "borderColor": color,
                "backgroundColor": color + "26",
                "fill": True,
                "tension": 0.4,
                "pointRadius": 0,
                "pointHoverRadius": 3,
                "pointHoverBackgroundColor": color,
                "borderWidth": 2,
            }
        )
        legend_entries.append(
            {"color": color, "label": var_name, "shape": "square", "datasetIndex": 0}
        )

    if show_zero_line:
        legend_entries.append({"color": ZERO_LINE_COLOR, "label": "Zero", "shape": "dashed"})

    mean_val = round(float(np.mean(all_values)), 2)
    std_val = round(float(np.std(all_values)), 2)
    median_val = round(float(np.median(all_values)), 2)

    cards = metric_cards(
        [
            {"label": "Mean", "value": str(mean_val)},
            {"label": "Std Dev", "value": str(std_val)},
            {"label": "Median", "value": str(median_val)},
        ]
    )

    zero_plugin = ""
    if show_zero_line:
        x_vals = all_values
        if float(x_vals.min()) <= 0 <= float(x_vals.max()):
            zero_plugin = f"""{{
    id: 'zeroLine',
    afterDraw(chart) {{
      const x = chart.scales.x.getPixelForValue(0);
      const ctx = chart.ctx;
      const area = chart.chartArea;
      ctx.save(); ctx.strokeStyle = '{ZERO_LINE_COLOR}'; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(x, area.top); ctx.lineTo(x, area.bottom); ctx.stroke(); ctx.restore();
    }}
  }}"""

    plugins_list = [crosshair_plugin_js()]
    if zero_plugin:
        plugins_list.append(zero_plugin)
    plugins_arr = "[" + ", ".join(plugins_list) + "]"

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
  plugins: {plugins_arr}
}});
{legend_toggle_js(chart_var, canvas_id)}"""

    widget_html = (
        cards
        + chart_container(canvas_id, 360)
        + legend_html(legend_entries, chart_id=canvas_id)
        + followup_buttons(
            [
                {
                    "label": "Different from zero?",
                    "prompt": f"Is {var_name} significantly different from zero?",
                },
                {
                    "label": "Prior vs posterior",
                    "prompt": f"Show the prior vs posterior for {var_name}",
                },
            ]
        )
        + script_tag(js)
    )

    text = f"Posterior density for {var_name}: mean = {mean_val}, std = {std_val}, median = {median_val}."

    return {"text": text, "widget_html": widget_html}
