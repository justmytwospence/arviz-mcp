"""Model comparison chart: horizontal bar chart of ELPD differences."""

from __future__ import annotations

import numpy as np
import pandas as pd

from arviz_mcp.charts.templates import (
    EXTENDED_COLORS,
    GRID_COLOR,
    animation_config,
    chart_container,
    followup_buttons,
    js_json,
    legend_html,
    metric_cards,
    script_tag,
    unique_id,
)


def comparison_chart(
    compare_df: pd.DataFrame,
) -> dict:
    """Build a model comparison chart from az.compare() output.

    Parameters
    ----------
    compare_df : DataFrame
        Output of az.compare(). Expected columns include: elpd_loo, se, weight, etc.
    """
    canvas_id = unique_id("compare")

    labels = [str(idx) for idx in compare_df.index]
    elpd = np.round(compare_df["elpd_loo"].values, 1).tolist()
    se = np.round(compare_df["se"].values, 1).tolist()

    lo = [round(e - 2 * s, 1) for e, s in zip(elpd, se)]
    hi = [round(e + 2 * s, 1) for e, s in zip(elpd, se)]

    weights = []
    if "weight" in compare_df.columns:
        weights = np.round(compare_df["weight"].values, 3).tolist()

    n_models = len(labels)
    chart_height = max(n_models * 50 + 80, 200)
    x_min = round(min(lo) - 5, 0)
    x_max = round(max(hi) + 5, 0)

    colors = [EXTENDED_COLORS[i % len(EXTENDED_COLORS)] for i in range(n_models)]

    cards_data = [
        {"label": "Best model", "value": labels[0], "sub": f"ELPD = {elpd[0]}"},
        {"label": "Models compared", "value": str(n_models)},
    ]
    if weights:
        cards_data.append({"label": "Best weight", "value": str(weights[0])})

    weight_line = ""
    if weights:
        weight_line = f"const weights = {js_json(weights)};"

    weight_tooltip = ""
    if weights:
        weight_tooltip = " + (weights[i] !== undefined ? '\\nWeight: ' + weights[i] : '')"

    js = f"""
const labels = {js_json(labels)};
const elpd = {js_json(elpd)};
const se = {js_json(se)};
const lo = {js_json(lo)};
const hi = {js_json(hi)};
const colors = {js_json(colors)};
{weight_line}
new Chart(document.getElementById('{canvas_id}'), {{
  type: 'bar',
  data: {{
    labels: labels,
    datasets: [{{
      data: elpd,
      backgroundColor: colors,
      hoverBackgroundColor: colors.map(c => c + 'CC'),
      borderRadius: 3,
      barThickness: 18
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    {animation_config()}
    scales: {{
      x: {{
        title: {{ display: true, text: 'ELPD (LOO)', font: {{ size: 12 }} }},
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
          return 'ELPD: ' + elpd[i] + ' (SE: ' + se[i] + ')\\n95% range: [' + lo[i] + ', ' + hi[i] + ']'{weight_tooltip};
        }}
      }}
    }} }}
  }},
  plugins: [{{
    id: 'errorBars',
    afterDraw(chart) {{
      const meta = chart.getDatasetMeta(0);
      const ctx = chart.ctx;
      const xScale = chart.scales.x;
      ctx.save(); ctx.lineWidth = 1.5;
      meta.data.forEach((bar, i) => {{
        ctx.strokeStyle = colors[i];
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
        metric_cards(cards_data)
        + chart_container(canvas_id, chart_height)
        + legend_html(
            [{"color": c, "label": lab, "shape": "square"} for c, lab in zip(colors, labels)]
        )
        + followup_buttons(
            [
                {
                    "label": "What drives the difference?",
                    "prompt": "What drives the difference between these models?",
                },
                {"label": "LOO-PIT plot", "prompt": "Show the LOO-PIT plot for the best model"},
            ]
        )
        + script_tag(js)
    )

    text = (
        f"Model comparison: {labels[0]} ranks best (ELPD = {elpd[0]}). "
        + (f"Stacking weight = {weights[0]}. " if weights else "")
        + f"Compared {n_models} models."
    )

    return {"text": text, "widget_html": widget_html}
