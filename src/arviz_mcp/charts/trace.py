"""Trace chart: multi-chain line chart showing MCMC samples over iterations."""

from __future__ import annotations

import numpy as np

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


def trace_chart(
    chain_values: dict[str, np.ndarray],
    var_name: str = "parameter",
) -> dict:
    """Build a trace plot as a multi-series line chart.

    Parameters
    ----------
    chain_values : dict
        Mapping chain name -> 1D array of samples.
    var_name : str
        Variable name for labels.
    """
    canvas_id = unique_id("trace")
    chart_var = f"chart_{canvas_id}"
    datasets_js = []
    legend_entries = []
    all_vals = []

    for i, (chain_name, vals) in enumerate(chain_values.items()):
        color = CHAIN_COLORS[i % len(CHAIN_COLORS)]
        vals = np.asarray(vals).ravel()
        all_vals.append(vals)
        # Subsample if >2000 draws to keep HTML small
        if len(vals) > 2000:
            step = len(vals) // 2000
            vals = vals[::step]
        data_points = [{"x": int(j), "y": round(float(v), 4)} for j, v in enumerate(vals)]
        datasets_js.append(
            {
                "label": chain_name,
                "data": data_points,
                "borderColor": color,
                "backgroundColor": color,
                "pointRadius": 0,
                "pointHoverRadius": 4,
                "pointHoverBackgroundColor": color,
                "borderWidth": 1,
                "tension": 0,
            }
        )
        legend_entries.append(
            {"color": color, "label": chain_name, "shape": "line", "datasetIndex": i}
        )

    combined = np.concatenate(all_vals)
    mean_val = round(float(np.mean(combined)), 2)
    std_val = round(float(np.std(combined)), 2)
    n_chains = len(chain_values)

    cards = metric_cards(
        [
            {"label": "Chains", "value": str(n_chains)},
            {"label": "Mean", "value": str(mean_val)},
            {"label": "Std Dev", "value": str(std_val)},
        ]
    )

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
        title: {{ display: true, text: 'Draw', font: {{ size: 12 }} }},
        grid: {{ color: '{GRID_COLOR}' }}
      }},
      y: {{
        title: {{ display: true, text: '{var_name}', font: {{ size: 12 }} }},
        grid: {{ color: '{GRID_COLOR}' }}
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
        + chart_container(canvas_id, 300)
        + legend_html(legend_entries, chart_id=canvas_id)
        + followup_buttons(
            [
                {"label": "Run diagnostics", "prompt": "Run full diagnostics on this model"},
                {"label": "Pair plot", "prompt": "Show the pair plot to check correlations"},
            ]
        )
        + script_tag(js)
    )

    text = f"Trace plot for {var_name}: {n_chains} chains, mean = {mean_val}, std = {std_val}."

    return {"text": text, "widget_html": widget_html}
