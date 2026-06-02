"""Chart.js widget builders for inline ArviZ visualizations."""

from arviz_mcp.charts.comparison import comparison_chart
from arviz_mcp.charts.density import density_chart
from arviz_mcp.charts.energy import energy_chart
from arviz_mcp.charts.forest import forest_chart
from arviz_mcp.charts.ppc import ppc_chart
from arviz_mcp.charts.prior_posterior import prior_posterior_chart
from arviz_mcp.charts.trace import trace_chart

__all__ = [
    "forest_chart",
    "density_chart",
    "trace_chart",
    "ppc_chart",
    "prior_posterior_chart",
    "comparison_chart",
    "energy_chart",
]
