"""Shared HTML helpers for Chart.js widget builders."""

from __future__ import annotations

import json
import uuid
from typing import Any

import numpy as np

CHART_JS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"

CHAIN_COLORS = ["#3266ad", "#E8811A", "#54A24B", "#E24B4A"]
EXTENDED_COLORS = [
    "#3266ad",
    "#E8811A",
    "#54A24B",
    "#E24B4A",
    "#534AB7",
    "#0F6E56",
    "#993C1D",
    "#993556",
    "#185FA5",
    "#3B6D11",
    "#854F0B",
]
ZERO_LINE_COLOR = "#E24B4A"
GRID_COLOR = "rgba(128,128,128,0.1)"


class _Encoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return round(float(obj), 4)
        if isinstance(obj, np.ndarray):
            return [
                round(float(x), 4) if isinstance(x, (float, np.floating)) else x
                for x in obj.tolist()
            ]
        return super().default(obj)


def js_json(obj: Any) -> str:
    return json.dumps(obj, cls=_Encoder)


def unique_id(prefix: str = "chart") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def metric_cards(cards: list[dict]) -> str:
    if not cards:
        return ""
    n_cols = min(len(cards), 4)
    items = []
    for c in cards:
        sub = (
            f'<p style="font-size: 12px; color: var(--color-text-tertiary); margin: 4px 0 0;">{c.get("sub", "")}</p>'
            if c.get("sub")
            else ""
        )
        items.append(
            f'<div style="background: var(--color-background-secondary); border-radius: var(--border-radius-md); padding: 1rem;">'
            f'<p style="font-size: 13px; color: var(--color-text-secondary); margin: 0 0 4px;">{c["label"]}</p>'
            f'<p style="font-size: 24px; font-weight: 500; margin: 0;">{c["value"]}</p>'
            f"{sub}</div>"
        )
    return (
        f'<div style="display: grid; grid-template-columns: repeat({n_cols}, minmax(0, 1fr)); gap: 12px; margin-bottom: 1.5rem;">'
        + "".join(items)
        + "</div>"
    )


def legend_html(entries: list[dict], chart_id: str = "") -> str:
    """Build an interactive legend. If chart_id is provided, clicking toggles dataset visibility."""
    spans = []
    for i, e in enumerate(entries):
        shape = e.get("shape", "square")
        if shape == "line":
            swatch = f'<span class="swatch" style="width: 10px; height: 2px; background: {e["color"]};"></span>'
        elif shape == "dashed":
            swatch = f'<span class="swatch" style="width: 10px; height: 2px; background: {e["color"]}; border-top: 1.5px dashed {e["color"]};"></span>'
        else:
            swatch = f'<span class="swatch" style="width: 10px; height: 10px; border-radius: 2px; background: {e["color"]};"></span>'
        dataset_idx = e.get("datasetIndex", i)
        if chart_id:
            click = f' onclick="toggleDataset_{chart_id}({dataset_idx}, this)" style="display: flex; align-items: center; gap: 4px; cursor: pointer; user-select: none;"'
        else:
            click = ' style="display: flex; align-items: center; gap: 4px;"'
        spans.append(f"<span{click}>{swatch}{e['label']}</span>")
    return (
        '<div style="display: flex; flex-wrap: wrap; gap: 16px; margin-top: 8px; font-size: 12px; color: var(--color-text-secondary);">'
        + "".join(spans)
        + "</div>"
    )


def legend_toggle_js(chart_var: str, chart_id: str) -> str:
    """Generate JS function for toggling dataset visibility from custom legend clicks."""
    return f"""
function toggleDataset_{chart_id}(index, el) {{
  const chart = {chart_var};
  const meta = chart.getDatasetMeta(index);
  meta.hidden = meta.hidden === null ? !chart.data.datasets[index].hidden : null;
  el.style.opacity = meta.hidden ? '0.3' : '1';
  chart.update();
}}"""


def followup_buttons(buttons: list[dict]) -> str:
    if not buttons:
        return ""
    btns = []
    for b in buttons:
        prompt = b["prompt"].replace("'", "\\'")
        btns.append(f"""<button onclick="sendPrompt('{prompt}')">{b["label"]}</button>""")
    return (
        '<div style="margin-top: 1rem; display: flex; gap: 8px; flex-wrap: wrap;">'
        + "".join(btns)
        + "</div>"
    )


def chart_container(canvas_id: str, height: int = 360) -> str:
    return f'<div style="position: relative; width: 100%; height: {height}px;"><canvas id="{canvas_id}"></canvas></div>'


def interaction_config(mode: str = "nearest", axis: str = "x") -> str:
    """Standard interaction config for hover/tooltip proximity detection."""
    return f"interaction: {{ mode: '{mode}', axis: '{axis}', intersect: false }},"


def animation_config(duration: int = 600, easing: str = "easeOutQuart") -> str:
    """Standard animation config for chart entry animations."""
    return f"animation: {{ duration: {duration}, easing: '{easing}' }},"


def tooltip_config() -> str:
    """Standard tooltip styling config."""
    return """tooltip: {
        backgroundColor: 'rgba(0,0,0,0.8)',
        titleFont: { size: 13 },
        bodyFont: { size: 12 },
        padding: 10,
        cornerRadius: 6,
        displayColors: true,
        boxPadding: 4,
      }"""


def crosshair_plugin_js() -> str:
    """Chart.js inline plugin that draws a vertical crosshair line on hover."""
    return """{
    id: 'crosshair',
    afterDraw(chart) {
      if (chart.tooltip && chart.tooltip.opacity > 0) {
        const x = chart.tooltip.caretX;
        const ctx = chart.ctx;
        const area = chart.chartArea;
        ctx.save();
        ctx.strokeStyle = 'rgba(128,128,128,0.3)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(x, area.top);
        ctx.lineTo(x, area.bottom);
        ctx.stroke();
        ctx.restore();
      }
    }
  }"""


def hover_emphasis_config() -> str:
    """Config to dim non-hovered datasets for visual emphasis."""
    return "hover: { mode: 'dataset', intersect: false },"


def script_tag(js_code: str) -> str:
    return f'<script src="{CHART_JS_CDN}"></script>\n<script>\n{js_code}\n</script>'
