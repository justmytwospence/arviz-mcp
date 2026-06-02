"""High-level insight tools for non-technical users."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import arviz as az
import numpy as np
from mcp.server.fastmcp import Context
from pydantic import BaseModel, Field

from arviz_mcp.diagnostics_core import BFMI_THRESHOLD, bad_rhat, bfmi_values, low_ess
from arviz_mcp.models import get_group_names
from arviz_mcp.utils import (
    READ_ONLY,
    _evidence_strength,
    get_group,
    get_model,
    to_json,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


class KeyFindingsInput(BaseModel):
    model: str = Field(..., description="Name of the loaded model")
    ci_prob: float = Field(
        default=0.94,
        description="Credible interval probability.",
        ge=0.0,
        le=1.0,
    )


def _run_trust_assessment(dt, groups: list[str]) -> dict:
    """Run diagnostic checks and return a trust assessment."""
    warnings: list[str] = []
    could_not_check: list[str] = []

    if "sample_stats" in groups:
        stats = dt["sample_stats"].ds
        if "diverging" in stats.data_vars:
            n_div = int(stats["diverging"].sum().values)
            if n_div > 0:
                total = int(stats["diverging"].size)
                pct = n_div / total * 100
                warnings.append(f"{n_div} divergent transitions ({pct:.1f}%)")

        if "energy" in stats.data_vars:
            try:
                low_count = sum(1 for v in bfmi_values(dt) if v < BFMI_THRESHOLD)
                if low_count > 0:
                    warnings.append(f"Low energy diagnostic in {low_count} chain(s)")
            except Exception as e:
                logger.warning("Energy (E-BFMI) trust check failed: %s", e)
                could_not_check.append("energy (E-BFMI)")

    if "posterior" in groups:
        try:
            bad = list(bad_rhat(dt).keys())
            if bad:
                warnings.append(f"Convergence issues for {bad}")
        except Exception as e:
            logger.warning("R-hat trust check failed: %s", e)
            could_not_check.append("R-hat")

        try:
            low = list(low_ess(dt).keys())
            if low:
                warnings.append(f"Low effective samples for {low}")
        except Exception as e:
            logger.warning("ESS trust check failed: %s", e)
            could_not_check.append("ESS")

    if not warnings:
        result: dict = {
            "level": "reliable",
            "detail": (
                "All diagnostic checks passed. These results are "
                "trustworthy and can be used with confidence."
            ),
        }
    elif len(warnings) <= 2:
        result = {
            "level": "caution",
            "detail": (
                "Most checks passed but there are minor concerns: " + "; ".join(warnings) + ". "
                "The results are largely reliable but some estimates "
                "should be interpreted with care."
            ),
        }
    else:
        result = {
            "level": "unreliable",
            "detail": (
                "Multiple issues detected: " + "; ".join(warnings) + ". "
                "These results should be interpreted with significant "
                "caution. The model may need adjustment."
            ),
        }

    if could_not_check:
        result["could_not_check"] = could_not_check
        result["detail"] += (
            " Note: some checks could not be run ("
            + ", ".join(could_not_check)
            + "), so this assessment may be incomplete."
        )
        if result["level"] == "reliable":
            result["level"] = "caution"

    return result


def register(mcp: FastMCP) -> None:
    @mcp.tool(
        name="key_findings",
        annotations=READ_ONLY,
    )
    def key_findings(params: KeyFindingsInput, ctx: Context) -> str:
        """Get a ranked summary of the model's key findings.

        Runs diagnostics to assess trustworthiness, then summarizes each
        parameter with its estimate, credible interval, direction, and
        the strength of evidence. Parameters are ranked by importance
        (signal strength). Designed for non-technical audiences.
        """
        dt = get_model(ctx, params.model)
        groups = get_group_names(dt)

        # Trust assessment
        trust = _run_trust_assessment(dt, groups)

        # Parameter findings
        findings: list[dict[str, Any]] = []
        if "posterior" not in groups:
            return to_json(
                {
                    "model": params.model,
                    "trust": trust,
                    "parameters": [],
                    "note": "No posterior group found in model.",
                }
            )

        posterior = get_group(dt, "posterior", params.model)

        # Compute HDI
        hdi_data = az.hdi(dt, prob=params.ci_prob)
        hdi_ds = hdi_data.ds

        for var_name in posterior.data_vars:
            samples = posterior[var_name].values.flatten()
            if len(samples) == 0:
                continue

            mean_val = float(np.mean(samples))
            sd_val = float(np.std(samples))

            # HDI
            if var_name in hdi_ds.data_vars:
                hdi_vals = hdi_ds[var_name].values
                if hdi_vals.ndim == 1 and len(hdi_vals) == 2:
                    hdi_low = float(hdi_vals[0])
                    hdi_high = float(hdi_vals[1])
                else:
                    # Multi-dimensional parameter -- skip for now
                    continue
            else:
                continue

            # Direction probability
            prob_positive = float(np.mean(samples > 0))
            prob_negative = float(np.mean(samples < 0))

            if prob_positive >= prob_negative:
                direction = "positive"
                direction_prob = prob_positive
            else:
                direction = "negative"
                direction_prob = prob_negative

            if direction_prob < 0.75:
                direction = "unclear"

            # Importance: signal-to-noise ratio
            importance = abs(mean_val) / sd_val if sd_val > 0 else 0.0

            findings.append(
                {
                    "name": str(var_name),
                    "estimate": round(mean_val, 4),
                    "sd": round(sd_val, 4),
                    "hdi_low": round(hdi_low, 4),
                    "hdi_high": round(hdi_high, 4),
                    "direction": direction,
                    "direction_probability": round(direction_prob, 4),
                    "evidence_strength": _evidence_strength(direction_prob),
                    "importance": round(importance, 3),
                }
            )

        # Sort by importance (highest first)
        findings.sort(key=lambda f: f["importance"], reverse=True)

        return to_json(
            {
                "model": params.model,
                "trust": trust,
                "ci_prob": params.ci_prob,
                "n_parameters": len(findings),
                "parameters": findings,
            }
        )
