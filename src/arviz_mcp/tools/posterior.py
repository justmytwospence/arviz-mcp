"""Posterior query tools: HDI, posterior statistics, and parameter comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import arviz as az
import numpy as np
from mcp.server.fastmcp import Context
from pydantic import BaseModel, Field

from arviz_mcp.utils import (
    READ_ONLY,
    _evidence_strength,
    get_group,
    get_model,
    to_json,
    validate_var_names,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


class HDIInput(BaseModel):
    model: str = Field(..., description="Name of the loaded model")
    var_names: list[str] = Field(..., description="Variables to compute HDI for")
    prob: float = Field(
        default=0.94,
        description="Probability mass for the HDI.",
        ge=0.0,
        le=1.0,
    )
    group: str = Field(
        default="posterior",
        description="Group to compute HDI from.",
    )


class PosteriorStatInput(BaseModel):
    model: str = Field(..., description="Name of the loaded model")
    var_name: str = Field(..., description="Variable name")
    stat: str = Field(
        ...,
        description=(
            "Statistic to compute. One of: 'mean', 'median', 'std', "
            "'quantile', 'prob_above', 'prob_below'."
        ),
    )
    value: float | None = Field(
        default=None,
        description=(
            "Required for 'quantile' (the quantile level, e.g. 0.25), "
            "'prob_above' and 'prob_below' (the threshold value)."
        ),
    )
    group: str = Field(
        default="posterior",
        description="Group to compute from.",
    )


class CompareParametersInput(BaseModel):
    model: str = Field(..., description="Name of the loaded model")
    var_a: str = Field(..., description="First variable name")
    var_b: str = Field(..., description="Second variable name")
    group: str = Field(
        default="posterior",
        description="Group to compare from.",
    )


def register(mcp: FastMCP) -> None:
    @mcp.tool(
        name="hdi",
        annotations=READ_ONLY,
    )
    def hdi(params: HDIInput, ctx: Context) -> str:
        """Compute highest density interval for specified parameters.

        Returns the HDI bounds (lower, upper) for each variable at the
        specified probability level.
        """
        dt = get_model(ctx, params.model)
        group_ds = get_group(dt, params.group, params.model)
        validate_var_names(group_ds, params.var_names, params.group)

        hdi_data = az.hdi(
            dt,
            var_names=params.var_names,
            prob=params.prob,
            group=params.group,
        )

        result: dict[str, Any] = {"prob": params.prob, "intervals": {}}

        # hdi returns a DataTree in ArviZ 1.0; data is in the root ds
        hdi_ds = hdi_data.ds
        for var_name in params.var_names:
            vals = hdi_ds[var_name].values
            if vals.ndim == 1:
                result["intervals"][var_name] = {
                    "low": float(vals[0]),
                    "high": float(vals[1]),
                }
            else:
                result["intervals"][var_name] = vals.tolist()

        return to_json(result)

    @mcp.tool(
        name="posterior_stat",
        annotations=READ_ONLY,
    )
    def posterior_stat(params: PosteriorStatInput, ctx: Context) -> str:
        """Compute a statistic on posterior samples for a given parameter.

        Useful for quick queries like "what's the median of beta" or
        "what's the probability that beta > 0".
        """
        dt = get_model(ctx, params.model)
        group_ds = get_group(dt, params.group, params.model)
        validate_var_names(group_ds, [params.var_name], params.group)

        samples = group_ds[params.var_name].values.flatten()
        n_samples = len(samples)

        stat_funcs = {
            "mean": lambda: float(np.mean(samples)),
            "median": lambda: float(np.median(samples)),
            "std": lambda: float(np.std(samples)),
        }

        if params.stat in stat_funcs:
            value = stat_funcs[params.stat]()
            return to_json(
                {
                    "variable": params.var_name,
                    "stat": params.stat,
                    "value": value,
                    "n_samples": n_samples,
                    "group": params.group,
                }
            )
        elif params.stat == "quantile":
            if params.value is None:
                raise ValueError(
                    "'value' is required for stat='quantile' "
                    "(e.g., value=0.25 for 25th percentile)."
                )
            value = float(np.quantile(samples, params.value))
            return to_json(
                {
                    "variable": params.var_name,
                    "stat": params.stat,
                    "value": value,
                    "n_samples": n_samples,
                    "group": params.group,
                }
            )
        elif params.stat == "prob_above":
            if params.value is None:
                raise ValueError("'value' is required for stat='prob_above' (the threshold).")
            prob = float(np.mean(samples > params.value))
            strength = _evidence_strength(prob)
            return to_json(
                {
                    "variable": params.var_name,
                    "stat": params.stat,
                    "threshold": params.value,
                    "value": prob,
                    "complement": round(1 - prob, 6),
                    "strength": strength,
                    "n_samples": n_samples,
                    "group": params.group,
                }
            )
        elif params.stat == "prob_below":
            if params.value is None:
                raise ValueError("'value' is required for stat='prob_below' (the threshold).")
            prob = float(np.mean(samples < params.value))
            strength = _evidence_strength(prob)
            return to_json(
                {
                    "variable": params.var_name,
                    "stat": params.stat,
                    "threshold": params.value,
                    "value": prob,
                    "complement": round(1 - prob, 6),
                    "strength": strength,
                    "n_samples": n_samples,
                    "group": params.group,
                }
            )
        else:
            raise ValueError(
                f"Unknown stat '{params.stat}'. "
                "Supported: mean, median, std, quantile, prob_above, prob_below."
            )

    @mcp.tool(
        name="compare_parameters",
        annotations=READ_ONLY,
    )
    def compare_parameters(params: CompareParametersInput, ctx: Context) -> str:
        """Compare two parameters from the joint posterior.

        Computes the probability that one parameter is larger than another,
        along with statistics on their difference. Useful for questions like
        "is the effect of A bigger than B?"
        """
        dt = get_model(ctx, params.model)
        group_ds = get_group(dt, params.group, params.model)
        validate_var_names(group_ds, [params.var_a, params.var_b], params.group)

        samples_a = group_ds[params.var_a].values.flatten()
        samples_b = group_ds[params.var_b].values.flatten()

        # Use paired samples from the joint posterior
        n = min(len(samples_a), len(samples_b))
        samples_a = samples_a[:n]
        samples_b = samples_b[:n]
        diff = samples_a - samples_b

        prob_a_greater = float(np.mean(diff > 0))
        prob_b_greater = float(np.mean(diff < 0))

        # HDI of the difference
        diff_sorted = np.sort(diff)
        ci_prob = 0.94
        n_samples = len(diff_sorted)
        interval_size = int(np.ceil(ci_prob * n_samples))
        min_width = np.inf
        hdi_low = diff_sorted[0]
        hdi_high = diff_sorted[-1]
        for start in range(n_samples - interval_size):
            width = diff_sorted[start + interval_size] - diff_sorted[start]
            if width < min_width:
                min_width = width
                hdi_low = float(diff_sorted[start])
                hdi_high = float(diff_sorted[start + interval_size])

        # Determine which has stronger evidence
        max_prob = max(prob_a_greater, prob_b_greater)
        strength = _evidence_strength(max_prob)

        return to_json(
            {
                "var_a": params.var_a,
                "var_b": params.var_b,
                "prob_a_greater": round(prob_a_greater, 4),
                "prob_b_greater": round(prob_b_greater, 4),
                "strength": strength,
                "difference": {
                    "description": f"{params.var_a} minus {params.var_b}",
                    "mean": round(float(np.mean(diff)), 4),
                    "median": round(float(np.median(diff)), 4),
                    "hdi_94": {"low": round(hdi_low, 4), "high": round(hdi_high, 4)},
                },
                "n_samples": n,
            }
        )
