"""Model comparison tools: compare and LOO."""

from __future__ import annotations

from typing import TYPE_CHECKING

import arviz as az
import numpy as np
from mcp.server.fastmcp import Context
from pydantic import BaseModel, Field

from arviz_mcp.models import get_group_names
from arviz_mcp.utils import READ_ONLY, format_dataframe, get_model, to_json

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


class CompareInput(BaseModel):
    models: list[str] = Field(
        ...,
        description="List of model names to compare (must have log_likelihood group).",
        min_length=2,
    )
    method: str = Field(
        default="stacking",
        description="Weight method: 'stacking' (default), 'bb-pseudo-bma', or 'pseudo-bma'.",
    )


class LOOInput(BaseModel):
    model: str = Field(..., description="Name of the loaded model")
    pointwise: bool = Field(
        default=False,
        description="Return pointwise LOO values.",
    )


def register(mcp: FastMCP) -> None:
    @mcp.tool(
        name="compare",
        annotations=READ_ONLY,
    )
    def compare(params: CompareInput, ctx: Context) -> str:
        """Compare multiple models using LOO-CV.

        Returns a comparison table ranked by expected log pointwise
        predictive density (elpd). Models must have a log_likelihood group.
        """
        model_dict = {}
        for name in params.models:
            dt = get_model(ctx, name)
            groups = get_group_names(dt)
            if "log_likelihood" not in groups:
                raise ValueError(
                    f"Model '{name}' has no 'log_likelihood' group. "
                    f"Available groups: {groups}. "
                    "Model comparison requires log_likelihood."
                )
            model_dict[name] = dt

        df = az.compare(model_dict, method=params.method)
        return format_dataframe(df)

    @mcp.tool(
        name="loo",
        annotations=READ_ONLY,
    )
    def loo(params: LOOInput, ctx: Context) -> str:
        """Compute LOO-CV (Leave-One-Out Cross-Validation) for a single model.

        Returns the LOO estimate, standard error, p_loo, and Pareto k
        diagnostic summary.
        """
        dt = get_model(ctx, params.model)
        groups = get_group_names(dt)
        if "log_likelihood" not in groups:
            raise ValueError(
                f"Model '{params.model}' has no 'log_likelihood' group. "
                f"Available groups: {groups}. LOO requires log_likelihood."
            )

        loo_data = az.loo(dt, pointwise=params.pointwise)

        result = {
            "elpd_loo": float(loo_data.elpd),
            "se": float(loo_data.se),
            "p_loo": float(loo_data.p),
            "n_samples": int(loo_data.n_samples),
            "n_data_points": int(loo_data.n_data_points),
            "warning": loo_data.warning,
        }

        if hasattr(loo_data, "pareto_k") and loo_data.pareto_k is not None:
            k_values = loo_data.pareto_k.values
            result["pareto_k"] = {
                "good (k < 0.5)": int(np.sum(k_values < 0.5)),
                "ok (0.5 <= k < 0.7)": int(np.sum((k_values >= 0.5) & (k_values < 0.7))),
                "bad (0.7 <= k < 1)": int(np.sum((k_values >= 0.7) & (k_values < 1.0))),
                "very bad (k >= 1)": int(np.sum(k_values >= 1.0)),
            }

        return to_json(result)
