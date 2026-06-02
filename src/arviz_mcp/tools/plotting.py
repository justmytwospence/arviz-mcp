"""Plotting tools wrapping ArviZ plot functions.

Tools that can be rendered as Chart.js widgets return widget_html for inline
display. Complex multi-panel plots (trace, pair) remain as matplotlib PNGs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import arviz as az
from mcp.server.fastmcp import Context
from mcp.types import CallToolResult, ContentBlock, TextContent
from pydantic import BaseModel, Field

from arviz_mcp.charts import (
    energy_chart,
    forest_chart,
    ppc_chart,
    prior_posterior_chart,
    trace_chart,
)
from arviz_mcp.models import get_group_names
from arviz_mcp.utils import (
    READ_ONLY,
    get_group,
    get_model,
    plot_collection_to_image,
    validate_var_names,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

MAX_VARS_DEFAULT = 12

_WIDGET_INSTRUCTION = (
    "This tool returns a widget_html field containing a complete Chart.js "
    "visualization. Render it inline using show_widget (or equivalent "
    "visualization tool) with the widget_html value as the widget_code. "
    "Do NOT modify the HTML. Also include the text field in your response "
    "to provide interpretation of the results."
)

_IMAGE_DISPLAY_INSTRUCTION = (
    "IMPORTANT: This tool returns an image in the tool result. "
    "The image will not display inline automatically. After calling "
    "this tool, you MUST save the image to a file and use present_files "
    "to show it to the user as an artifact."
)

_PLOT_ANNOTATIONS = READ_ONLY


def _widget_result(chart_output: dict, extra_text: str | None = None) -> CallToolResult:
    """Build a CallToolResult from a chart builder output dict."""
    content: list[ContentBlock] = [TextContent(type="text", text=chart_output["text"])]
    if extra_text:
        content.append(TextContent(type="text", text=extra_text))
    content.append(
        TextContent(
            type="text",
            text=f"WIDGET_HTML_START\n{chart_output['widget_html']}\nWIDGET_HTML_END",
        )
    )
    content.append(TextContent(type="text", text=_WIDGET_INSTRUCTION))
    return CallToolResult(content=content)


def _plot_result(pc, warning: str | None = None) -> CallToolResult:
    """Convert a PlotCollection to ImageContent, with optional text warning."""
    content: list[ContentBlock] = [TextContent(type="text", text=_IMAGE_DISPLAY_INSTRUCTION)]
    if warning:
        content.append(TextContent(type="text", text=warning))
    img = plot_collection_to_image(pc)
    content.append(img.to_image_content())
    return CallToolResult(content=content)


class PlotTraceInput(BaseModel):
    model: str = Field(..., description="Name of the loaded model")
    var_names: list[str] | None = Field(
        default=None,
        description="Variables to plot. Default: all (limited to 12).",
    )


class PlotForestInput(BaseModel):
    models: list[str] = Field(
        ...,
        description="One or more model names.",
        min_length=1,
    )
    var_names: list[str] | None = Field(
        default=None,
        description="Variables to plot.",
    )
    ci_prob: float = Field(
        default=0.94,
        description="Credible interval probability.",
        ge=0.0,
        le=1.0,
    )
    combined: bool = Field(
        default=True,
        description="Combine chains.",
    )


class PlotEnergyInput(BaseModel):
    model: str = Field(..., description="Name of the loaded model")


class PlotPPCDistInput(BaseModel):
    model: str = Field(..., description="Name of the loaded model")
    var_names: list[str] | None = Field(
        default=None,
        description="Observed variables to check.",
    )
    num_samples: int = Field(
        default=50,
        description="Number of posterior predictive samples to overlay.",
        ge=1,
        le=500,
    )
    kind: str | None = Field(
        default=None,
        description="Plot kind: 'kde' or 'cumulative'. Default: ArviZ default.",
    )


class PlotPairInput(BaseModel):
    model: str = Field(..., description="Name of the loaded model")
    var_names: list[str] = Field(
        ...,
        description="Variables to include (recommend 2-6).",
        min_length=2,
    )
    divergences: bool = Field(
        default=True,
        description="Highlight divergent transitions.",
    )


class PlotLOOPITInput(BaseModel):
    model: str = Field(..., description="Name of the loaded model")
    var_names: list[str] | None = Field(
        default=None,
        description="Observed variable(s) to check.",
    )


class PlotPriorPosteriorInput(BaseModel):
    model: str = Field(..., description="Name of the loaded model")
    var_names: list[str] | None = Field(
        default=None,
        description="Variables to plot. Default: all shared between prior and posterior.",
    )


def _limit_var_names(
    dt, var_names: list[str] | None, max_vars: int = MAX_VARS_DEFAULT
) -> tuple[list[str] | None, str | None]:
    """Limit variable names to max_vars, returning a warning if truncated."""
    if var_names is not None:
        if len(var_names) > max_vars:
            warning = f"Limiting to first {max_vars} of {len(var_names)} variables."
            return var_names[:max_vars], warning
        return var_names, None

    groups = get_group_names(dt)
    if "posterior" in groups:
        all_vars = [str(v) for v in dt["posterior"].ds.data_vars]
        if len(all_vars) > max_vars:
            warning = (
                f"Model has {len(all_vars)} variables. "
                f"Showing first {max_vars}. Pass var_names to select specific ones."
            )
            return all_vars[:max_vars], warning

    return None, None


def register(mcp: FastMCP) -> None:
    @mcp.tool(
        name="plot_trace",
        annotations=_PLOT_ANNOTATIONS,
    )
    def plot_trace(params: PlotTraceInput, ctx: Context) -> CallToolResult:
        """Generate trace plots showing MCMC samples per chain.

        Returns a widget_html field containing a Chart.js visualization for
        each requested variable. Render it inline using show_widget with the
        widget_html value as widget_code. Do NOT modify the HTML.

        For a single variable, returns an interactive Chart.js trace plot.
        For multiple variables, falls back to matplotlib PNG.
        """
        dt = get_model(ctx, params.model)
        var_names, warning = _limit_var_names(dt, params.var_names)

        # For single variable, use Chart.js widget
        effective_vars = var_names or [str(v) for v in dt["posterior"].ds.data_vars]
        if len(effective_vars) == 1:
            vname = effective_vars[0]
            post = dt["posterior"].ds[vname]
            chain_values = {}
            for chain_idx in post.coords["chain"].values:
                chain_values[f"Chain {chain_idx}"] = post.sel(chain=chain_idx).values.ravel()
            chart = trace_chart(chain_values, var_name=vname)
            return _widget_result(chart, warning)

        # Multi-variable: matplotlib fallback
        pc = az.plot_trace(dt, var_names=var_names)
        return _plot_result(pc, warning)

    @mcp.tool(
        name="plot_forest",
        annotations=_PLOT_ANNOTATIONS,
    )
    def plot_forest(params: PlotForestInput, ctx: Context) -> CallToolResult:
        """Forest plot for comparing parameter estimates.

        Shows credible intervals for each parameter as a horizontal bar chart.

        Returns a widget_html field containing a Chart.js visualization.
        Render it inline using show_widget with the widget_html value as
        widget_code. Do NOT modify the HTML. Also include the text field
        in your response to provide interpretation.
        """
        if len(params.models) == 1:
            dt = get_model(ctx, params.models[0])
            summary_df = az.summary(
                dt,
                var_names=params.var_names,
                ci_prob=params.ci_prob,
            )
            chart = forest_chart(
                summary_df,
                var_name=params.models[0],
                ci_prob=params.ci_prob,
            )
            return _widget_result(chart)
        else:
            # Multi-model: compute summary per model and combine
            dt_dict = {name: get_model(ctx, name) for name in params.models}
            # Fall back to matplotlib for multi-model forest
            pc = az.plot_forest(
                dt_dict,
                var_names=params.var_names,
                combined=params.combined,
                ci_probs=[params.ci_prob],
            )
            return _plot_result(pc)

    @mcp.tool(
        name="plot_energy",
        annotations=_PLOT_ANNOTATIONS,
    )
    def plot_energy(params: PlotEnergyInput, ctx: Context) -> CallToolResult:
        """Energy diagnostic plot showing marginal energy vs energy transition.

        Large differences indicate poor exploration of the posterior.
        Requires sample_stats with energy data (HMC/NUTS samplers).

        Returns a widget_html field containing a Chart.js visualization.
        Render it inline using show_widget with the widget_html value as
        widget_code. Do NOT modify the HTML.
        """
        dt = get_model(ctx, params.model)
        groups = get_group_names(dt)
        if "sample_stats" not in groups:
            raise ValueError(
                f"Model '{params.model}' has no 'sample_stats' group. "
                "Energy plot requires HMC/NUTS sample_stats."
            )
        stats = dt["sample_stats"].ds
        if "energy" not in stats.data_vars:
            raise ValueError(
                f"Model '{params.model}' has no energy data in sample_stats. "
                "Energy plot requires HMC/NUTS sample_stats with energy."
            )

        energy_vals = stats["energy"].values.ravel()
        chart = energy_chart(energy_vals)
        return _widget_result(chart)

    @mcp.tool(
        name="plot_ppc",
        annotations=_PLOT_ANNOTATIONS,
    )
    def plot_ppc(params: PlotPPCDistInput, ctx: Context) -> CallToolResult:
        """Posterior predictive check plot.

        Overlays posterior predictive samples on observed data to assess
        model fit. Requires 'posterior_predictive' and 'observed_data' groups.

        Returns a widget_html field containing a Chart.js visualization.
        Render it inline using show_widget with the widget_html value as
        widget_code. Do NOT modify the HTML.
        """
        dt = get_model(ctx, params.model)
        groups = get_group_names(dt)

        missing = []
        if "posterior_predictive" not in groups:
            missing.append("posterior_predictive")
        if "observed_data" not in groups:
            missing.append("observed_data")
        if missing:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            f"Error: Model '{params.model}' is missing required groups: {missing}. "
                            f"Available groups: {groups}. "
                            "PPC requires both posterior_predictive and observed_data."
                        ),
                    )
                ],
            )

        obs_ds = dt["observed_data"].ds
        pp_ds = dt["posterior_predictive"].ds

        # Pick first observed variable or use var_names
        if params.var_names:
            vname = params.var_names[0]
        else:
            vname = list(obs_ds.data_vars)[0]

        observed = obs_ds[vname].values.ravel()

        # Stack chains and draws for predicted samples
        pp_var = pp_ds[vname]
        # Shape: (chain, draw, *obs_dims) -> (chain*draw, n_obs)
        pp_stacked = pp_var.stack(sample=("chain", "draw")).values
        if pp_stacked.ndim == 1:
            pp_stacked = pp_stacked.reshape(1, -1)
        else:
            # Transpose so first dim is samples
            pp_stacked = pp_stacked.T

        chart = ppc_chart(
            observed,
            pp_stacked,
            var_name=vname,
            num_overlay=min(params.num_samples, 30),
        )
        return _widget_result(chart)

    @mcp.tool(
        name="plot_pair",
        annotations=_PLOT_ANNOTATIONS,
    )
    def plot_pair(params: PlotPairInput, ctx: Context) -> CallToolResult:
        """Pair plot / corner plot for examining correlations between parameters.

        Useful for detecting posterior correlations and identifying
        divergent transitions in parameter space.

        IMPORTANT: This tool returns an image. Save it to a file and use
        present_files to show it to the user as an artifact.
        """
        dt = get_model(ctx, params.model)
        post_ds = get_group(dt, "posterior", params.model)
        validate_var_names(post_ds, params.var_names, "posterior")

        pc = az.plot_pair(dt, var_names=params.var_names)
        return _plot_result(pc)

    @mcp.tool(
        name="plot_loo_pit",
        annotations=_PLOT_ANNOTATIONS,
    )
    def plot_loo_pit(params: PlotLOOPITInput, ctx: Context) -> CallToolResult:
        """LOO probability integral transform plot for checking calibration.

        A well-calibrated model should show a uniform PIT distribution.
        Requires 'posterior_predictive' group.

        IMPORTANT: This tool returns an image. Save it to a file and use
        present_files to show it to the user as an artifact.
        """
        dt = get_model(ctx, params.model)
        groups = get_group_names(dt)

        if "posterior_predictive" not in groups:
            raise ValueError(
                f"Model '{params.model}' has no 'posterior_predictive' group. "
                "LOO-PIT requires posterior_predictive."
            )

        kwargs = {}
        if params.var_names is not None:
            kwargs["var_names"] = params.var_names

        pc = az.plot_loo_pit(dt, **kwargs)
        return _plot_result(pc)

    @mcp.tool(
        name="plot_prior_posterior",
        annotations=_PLOT_ANNOTATIONS,
    )
    def plot_prior_posterior(params: PlotPriorPosteriorInput, ctx: Context) -> CallToolResult:
        """Plot prior vs posterior distributions for comparison.

        Shows how the data updated the prior beliefs. Requires both
        'prior' and 'posterior' groups.

        Returns a widget_html field containing a Chart.js visualization
        for each variable. Render it inline using show_widget with the
        widget_html value as widget_code. Do NOT modify the HTML.
        """
        dt = get_model(ctx, params.model)
        groups = get_group_names(dt)

        missing = []
        if "prior" not in groups:
            missing.append("prior")
        if "posterior" not in groups:
            missing.append("posterior")
        if missing:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            f"Error: Model '{params.model}' is missing required groups: {missing}. "
                            f"Available groups: {groups}. "
                            "Prior-posterior plot requires both prior and posterior groups."
                        ),
                    )
                ],
            )

        var_names, warning = _limit_var_names(dt, params.var_names)

        # Pick first variable for widget
        effective_vars = var_names or [str(v) for v in dt["posterior"].ds.data_vars]
        if len(effective_vars) == 1:
            vname = effective_vars[0]
            prior_vals = dt["prior"].ds[vname].values.ravel()
            post_vals = dt["posterior"].ds[vname].values.ravel()
            chart = prior_posterior_chart(prior_vals, post_vals, var_name=vname)
            return _widget_result(chart, warning)

        # Multi-variable: matplotlib fallback
        pc = az.plot_prior_posterior(dt, var_names=var_names)
        return _plot_result(pc, warning)
