"""Forest plot: horizontal bar chart with credible interval error bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

from arviz_mcp.charts.templates import (
    CHAIN_COLORS,
    GRID_COLOR,
    ZERO_LINE_COLOR,
    animation_config,
    chart_container,
    followup_buttons,
    js_json,
    legend_html,
    metric_cards,
    script_tag,
    unique_id,
)


def forest_chart(
    summary_df: pd.DataFrame,
    var_name: str = "parameters",
    ci_prob: float = 0.94,
) -> dict:
    """Build a forest plot as a Chart.js horizontal bar with error bars.

    Parameters
    ----------
    summary_df : DataFrame
        Must have columns: mean, sd, and CI bounds (eti_3% / eti_97% or similar).
    var_name : str
        Label for the x-axis.
    ci_prob : float
        Credible interval probability used.
    """
    labels = [str(idx) for idx in summary_df.index]
    means = np.round(summary_df["mean"].values, 2).tolist()
    sd = np.round(summary_df["sd"].values, 2).tolist()

    ci_cols = [c for c in summary_df.columns if c.startswith("eti_") or c.startswith("hdi_")]
    if len(ci_cols) >= 2:
        lo = np.round(summary_df[ci_cols[0]].values, 2).tolist()
        hi = np.round(summary_df[ci_cols[1]].values, 2).tolist()
    else:
        lo = [round(m - 2 * s, 2) for m, s in zip(means, sd)]
        hi = [round(m + 2 * s, 2) for m, s in zip(means, sd)]

    order = sorted(range(len(means)), key=lambda i: means[i], reverse=True)
    labels = [labels[i] for i in order]
    means = [means[i] for i in order]
    lo = [lo[i] for i in order]
    hi = [hi[i] for i in order]
    sd = [sd[i] for i in order]

    n_bars = len(labels)
    chart_height = max(n_bars * 40 + 80, 200)
    x_min = round(min(lo) - 2, 1)
    x_max = round(max(hi) + 2, 1)
    ci_pct = int(ci_prob * 100)
    canvas_id = unique_id("forest")

    cards = [
        {"label": "Parameters", "value": str(n_bars)},
        {"label": "CI Level", "value": f"{ci_pct}%"},
        {"label": "Largest effect", "value": str(means[0]), "sub": labels[0]},
    ]

    legend = legend_html(
        [
            {"color": CHAIN_COLORS[0], "label": f"Mean with {ci_pct}% CI", "shape": "square"},
            {"color": ZERO_LINE_COLOR, "label": "Zero reference", "shape": "dashed"},
        ]
    )

    buttons = followup_buttons(
        [
            {"label": "Check convergence", "prompt": "Show me the trace plot to check convergence"},
            {"label": "Run diagnostics", "prompt": "Run full diagnostics on this model"},
            {
                "label": "Widest interval?",
                "prompt": "Which parameter has the widest credible interval?",
            },
        ]
    )

    js = f"""
const labels = {js_json(labels)};
const means = {js_json(means)};
const lo = {js_json(lo)};
const hi = {js_json(hi)};
const sd = {js_json(sd)};
new Chart(document.getElementById('{canvas_id}'), {{
  type: 'bar',
  data: {{
    labels: labels,
    datasets: [{{
      data: means,
      backgroundColor: '{CHAIN_COLORS[0]}',
      hoverBackgroundColor: '{CHAIN_COLORS[0]}CC',
      borderRadius: 3,
      barThickness: 14
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    {animation_config()}
    scales: {{
      x: {{
        title: {{ display: true, text: '{var_name}', font: {{ size: 12 }} }},
        grid: {{ color: '{GRID_COLOR}' }},
        min: {x_min}, max: {x_max}
      }},
      y: {{ grid: {{ display: false }}, ticks: {{ font: {{ size: 13 }} }} }}
    }},
    plugins: {{ legend: {{ display: false }}, tooltip: {{
      backgroundColor: 'rgba(0,0,0,0.8)',
      titleFont: {{ size: 13 }},
      bodyFont: {{ size: 12 }},
      padding: 10,
      cornerRadius: 6,
      callbacks: {{
        title: (items) => labels[items[0].dataIndex],
        label: (ctx) => {{
          const i = ctx.dataIndex;
          return ['Mean: ' + means[i] + ' (SD: ' + sd[i] + ')', '{ci_pct}% CI: [' + lo[i] + ', ' + hi[i] + ']'];
        }}
      }}
    }} }}
  }},
  plugins: [{{
    id: 'zeroLine',
    afterDraw(chart) {{
      const xScale = chart.scales.x;
      if ({x_min} <= 0 && 0 <= {x_max}) {{
        const x = xScale.getPixelForValue(0);
        const ctx = chart.ctx;
        const area = chart.chartArea;
        ctx.save(); ctx.strokeStyle = '{ZERO_LINE_COLOR}'; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
        ctx.beginPath(); ctx.moveTo(x, area.top); ctx.lineTo(x, area.bottom); ctx.stroke(); ctx.restore();
      }}
    }}
  }}, {{
    id: 'errorBars',
    afterDraw(chart) {{
      const meta = chart.getDatasetMeta(0);
      const ctx = chart.ctx;
      const xScale = chart.scales.x;
      ctx.save(); ctx.strokeStyle = '{CHAIN_COLORS[0]}'; ctx.lineWidth = 1.5;
      meta.data.forEach((bar, i) => {{
        const y = bar.y;
        const xL = xScale.getPixelForValue(lo[i]);
        const xH = xScale.getPixelForValue(hi[i]);
        ctx.beginPath(); ctx.moveTo(xL, y); ctx.lineTo(xH, y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(xL, y - 5); ctx.lineTo(xL, y + 5); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(xH, y - 5); ctx.lineTo(xH, y + 5); ctx.stroke();
      }});
      ctx.restore();
    }}
  }}]
}});"""

    widget_html = (
        metric_cards(cards)
        + chart_container(canvas_id, chart_height)
        + legend
        + buttons
        + script_tag(js)
    )

    text = (
        f"Forest plot for {var_name}: {labels[0]} has the largest mean ({means[0]}). "
        f"All {ci_pct}% credible intervals shown with error bars."
    )

    return {"text": text, "widget_html": widget_html}
