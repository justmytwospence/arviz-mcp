"""Diagnostic and summary tools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import arviz as az
from mcp.server.fastmcp import Context
from pydantic import BaseModel, Field

from arviz_mcp.diagnostics_core import (
    BFMI_THRESHOLD,
    bad_rhat,
    bfmi_values,
    low_ess,
)
from arviz_mcp.models import get_group_names
from arviz_mcp.utils import (
    READ_ONLY,
    format_dataframe,
    get_group,
    get_model,
    to_json,
    validate_var_names,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


class DiagnosticsInput(BaseModel):
    model: str = Field(..., description="Name of the loaded model")
    var_names: list[str] | None = Field(
        default=None,
        description="Specific variables to check. Default: all posterior variables.",
    )


class SummaryInput(BaseModel):
    model: str = Field(..., description="Name of the loaded model")
    var_names: list[str] | None = Field(
        default=None,
        description="Variables to summarize. Default: all.",
    )
    group: str = Field(
        default="posterior",
        description="Which group to summarize (e.g., 'posterior', 'prior').",
    )
    ci_prob: float = Field(
        default=0.94,
        description="Credible interval probability mass.",
        ge=0.0,
        le=1.0,
    )
    kind: str = Field(
        default="all",
        description="'all', 'stats', or 'diagnostics'.",
    )
    round_to: int | str = Field(
        default="auto",
        description="Number of decimal places, or 'auto'.",
    )


def register(mcp: FastMCP) -> None:
    @mcp.tool(
        name="diagnostics",
        annotations=READ_ONLY,
    )
    def diagnostics(params: DiagnosticsInput, ctx: Context) -> str:
        """Run MCMC diagnostic checks on a fitted model.

        Checks for divergent transitions, R-hat convergence, ESS (bulk and
        tail), tree depth saturation, and energy diagnostics (E-BFMI).
        Returns a structured report with interpretation of each diagnostic.
        """
        dt = get_model(ctx, params.model)
        groups = get_group_names(dt)
        report: dict = {"model": params.model, "checks": [], "warnings": []}

        # -- Divergences --
        if "sample_stats" in groups:
            stats = dt["sample_stats"].ds
            if "diverging" in stats.data_vars:
                n_div = int(stats["diverging"].sum().values)
                total = int(stats["diverging"].size)
                report["divergences"] = {
                    "count": n_div,
                    "total_transitions": total,
                }
                if n_div > 0:
                    pct = n_div / total * 100
                    report["warnings"].append(
                        f"{n_div} divergent transitions ({pct:.1f}%). "
                        "This may indicate model misspecification or that "
                        "the posterior geometry is difficult to explore. "
                        "Consider reparameterization."
                    )
                    report["divergences"]["interpretation"] = (
                        f"The estimation process had trouble exploring some "
                        f"regions of the parameter space ({n_div} out of "
                        f"{total} steps). This may make some estimates less "
                        f"reliable."
                    )
                else:
                    report["divergences"]["interpretation"] = (
                        "The estimation process explored the parameter space "
                        "smoothly with no issues."
                    )
                report["checks"].append("divergences")

            # -- Tree depth --
            if "tree_depth" in stats.data_vars:
                max_td_key = "max_treedepth"
                if max_td_key in stats.attrs:
                    max_td = int(stats.attrs[max_td_key])
                elif "reached_max_treedepth" in stats.data_vars:
                    n_max = int(stats["reached_max_treedepth"].sum().values)
                    if n_max > 0:
                        report["warnings"].append(
                            f"{n_max} transitions reached max tree depth. "
                            "Consider increasing max_treedepth."
                        )
                    report["tree_depth"] = {"saturated_count": n_max}
                    report["checks"].append("tree_depth")
                    max_td = None
                else:
                    max_td = None

                if max_td is not None:
                    n_max = int((stats["tree_depth"] >= max_td).sum().values)
                    if n_max > 0:
                        report["warnings"].append(
                            f"{n_max} transitions hit max tree depth ({max_td}). "
                            "Consider increasing max_treedepth."
                        )
                    report["tree_depth"] = {
                        "max_treedepth": max_td,
                        "saturated_count": n_max,
                    }
                    report["checks"].append("tree_depth")
            elif "reached_max_treedepth" in stats.data_vars:
                n_max = int(stats["reached_max_treedepth"].sum().values)
                if n_max > 0:
                    report["warnings"].append(
                        f"{n_max} transitions reached max tree depth. "
                        "Consider increasing max_treedepth."
                    )
                report["tree_depth"] = {"saturated_count": n_max}
                report["checks"].append("tree_depth")

            # -- E-BFMI --
            if "energy" in stats.data_vars:
                try:
                    values = bfmi_values(dt)
                    low_bfmi = [
                        {"chain": i, "bfmi": v} for i, v in enumerate(values) if v < BFMI_THRESHOLD
                    ]
                    report["energy_bfmi"] = {"values": values}
                    if low_bfmi:
                        report["warnings"].append(
                            f"Low E-BFMI detected in {len(low_bfmi)} chain(s): "
                            f"{low_bfmi}. This may indicate a funnel-like "
                            "posterior geometry."
                        )
                        report["energy_bfmi"]["interpretation"] = (
                            "The estimation process had difficulty with the "
                            "energy landscape in some chains, suggesting the "
                            "model may have challenging geometry."
                        )
                    else:
                        report["energy_bfmi"]["interpretation"] = (
                            "The energy diagnostics look good -- the estimation "
                            "process navigated the parameter space efficiently."
                        )
                    report["checks"].append("energy_bfmi")
                except Exception as e:
                    logger.warning("E-BFMI computation failed: %s", e)
                    report["energy_bfmi"] = {"error": str(e)}

        # -- R-hat --
        if "posterior" in groups:
            try:
                bad_rhat_map = bad_rhat(dt, params.var_names)
                report["rhat"] = {"problematic": bad_rhat_map}
                if bad_rhat_map:
                    report["warnings"].append(
                        f"R-hat > 1.01 for {list(bad_rhat_map.keys())}. "
                        "Chains have not converged. Run longer or "
                        "check for multimodality."
                    )
                    report["rhat"]["interpretation"] = (
                        f"The independent estimation runs disagree on the "
                        f"values of {list(bad_rhat_map.keys())}. This means "
                        f"those estimates have not settled and should not "
                        f"be trusted yet."
                    )
                else:
                    report["rhat"]["interpretation"] = (
                        "All independent estimation runs agree on the "
                        "parameter values, which is a good sign of reliability."
                    )
                report["checks"].append("rhat")
            except Exception as e:
                logger.warning("R-hat computation failed: %s", e)
                report["rhat"] = {"error": str(e)}

        # -- ESS --
        if "posterior" in groups:
            try:
                low_bulk = low_ess(dt, params.var_names, "bulk")
                low_tail = low_ess(dt, params.var_names, "tail")

                report["ess"] = {
                    "low_bulk": low_bulk,
                    "low_tail": low_tail,
                }
                if low_bulk or low_tail:
                    if low_bulk:
                        report["warnings"].append(
                            f"Low bulk ESS (<400) for {list(low_bulk.keys())}. "
                            "Posterior means and medians may be unreliable. "
                            "Run longer chains."
                        )
                    if low_tail:
                        report["warnings"].append(
                            f"Low tail ESS (<400) for {list(low_tail.keys())}. "
                            "Posterior tail quantiles (credible interval bounds) "
                            "may be unreliable. Run longer chains."
                        )
                    problem_vars = list(set(list(low_bulk.keys()) + list(low_tail.keys())))
                    report["ess"]["interpretation"] = (
                        f"Not enough independent samples for "
                        f"{problem_vars}. The estimates for these "
                        f"parameters may be noisy -- longer estimation "
                        f"runs would help."
                    )
                else:
                    report["ess"]["interpretation"] = (
                        "There are enough independent samples for all "
                        "parameters, so the estimates should be stable."
                    )
                report["checks"].append("ess")
            except Exception as e:
                logger.warning("ESS computation failed: %s", e)
                report["ess"] = {"error": str(e)}

        # -- Overall assessment --
        if not report["warnings"]:
            report["assessment"] = {
                "level": "healthy",
                "summary": (
                    "All diagnostic checks passed. The model's results "
                    "are trustworthy and can be used with confidence."
                ),
            }
        elif len(report["warnings"]) <= 2:
            report["assessment"] = {
                "level": "has warnings",
                "summary": (
                    "Most checks passed, but there are minor concerns. "
                    "The results are largely reliable, but some estimates "
                    "should be interpreted with care."
                ),
            }
        else:
            report["assessment"] = {
                "level": "problematic",
                "summary": (
                    "Multiple diagnostic checks flagged issues. These "
                    "results should be interpreted with caution -- the "
                    "model may need adjustment before the findings can "
                    "be trusted."
                ),
            }

        return to_json(report)

    @mcp.tool(
        name="summary",
        annotations=READ_ONLY,
    )
    def summary(params: SummaryInput, ctx: Context) -> str:
        """Compute summary statistics for model parameters.

        Returns a table with mean, sd, credible interval bounds, ESS
        (bulk/tail), R-hat, and MCSE for each parameter. Wraps az.summary().
        """
        dt = get_model(ctx, params.model)
        # Validate group exists
        get_group(dt, params.group, params.model)

        if params.var_names:
            group_ds = get_group(dt, params.group, params.model)
            validate_var_names(group_ds, params.var_names, params.group)

        df = az.summary(
            dt,
            var_names=params.var_names,
            group=params.group,
            ci_prob=params.ci_prob,
            kind=params.kind,
            round_to=params.round_to,
        )
        return format_dataframe(df)
