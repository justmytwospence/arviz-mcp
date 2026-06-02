"""Raw data access tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mcp.server.fastmcp import Context
from pydantic import BaseModel, Field

from arviz_mcp.utils import READ_ONLY, get_group, get_model, to_json, validate_var_names

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

MAX_VALUES = 10000


class GetValuesInput(BaseModel):
    model: str = Field(..., description="Name of the loaded model")
    group: str = Field(
        ...,
        description="Group name (e.g., 'posterior', 'prior', 'observed_data', 'posterior_predictive').",
    )
    var_name: str = Field(..., description="Variable name within the group")
    sel: dict[str, str | int | float] | None = Field(
        default=None,
        description=(
            'Coordinate selection to subset the data (e.g., {"school": "Choate"} or {"chain": 0}).'
        ),
    )
    stat: str | None = Field(
        default=None,
        description=(
            "If provided, reduce over chain+draw dimensions using this "
            "statistic: 'mean', 'median', 'std'. Otherwise returns shape "
            "info and a sample of values."
        ),
    )


def register(mcp: FastMCP) -> None:
    @mcp.tool(
        name="get_values",
        annotations=READ_ONLY,
    )
    def get_values(params: GetValuesInput, ctx: Context) -> str:
        """Extract raw values from any group/variable in the model.

        For when you need actual numbers rather than summaries. Supports
        coordinate selection and optional statistical reduction.
        """
        dt = get_model(ctx, params.model)
        dataset = get_group(dt, params.group, params.model)
        validate_var_names(dataset, [params.var_name], params.group)

        var_data = dataset[params.var_name]

        # Apply coordinate selection
        if params.sel:
            try:
                var_data = var_data.sel(params.sel)
            except (KeyError, ValueError) as e:
                coords_info = {
                    dim: list(var_data.coords[dim].values)
                    for dim in var_data.dims
                    if dim in var_data.coords
                }
                return to_json(
                    {
                        "error": f"Selection error: {e}",
                        "available_coords": coords_info,
                    }
                )

        result: dict = {
            "variable": params.var_name,
            "group": params.group,
            "dims": list(var_data.dims),
            "shape": list(var_data.shape),
        }

        # Add coordinate info
        coords = {}
        for dim in var_data.dims:
            if dim in var_data.coords:
                coord_vals = var_data.coords[dim].values
                if len(coord_vals) <= 20:
                    coords[dim] = coord_vals.tolist()
                else:
                    coords[dim] = f"{len(coord_vals)} values ({coord_vals[0]} ... {coord_vals[-1]})"
        result["coords"] = coords

        # Apply stat or return raw values
        if params.stat:
            reduce_dims = [d for d in ("chain", "draw") if d in var_data.dims]
            stat_funcs = {
                "mean": lambda: var_data.mean(dim=reduce_dims).values,
                "median": lambda: var_data.median(dim=reduce_dims).values,
                "std": lambda: var_data.std(dim=reduce_dims).values,
            }
            if params.stat not in stat_funcs:
                raise ValueError(f"Unknown stat '{params.stat}'. Supported: mean, median, std.")
            values = stat_funcs[params.stat]()
            result["stat"] = params.stat
            result["values"] = values.tolist() if isinstance(values, np.ndarray) else float(values)
        else:
            total = int(np.prod(var_data.shape))
            if total <= MAX_VALUES:
                result["values"] = var_data.values.tolist()
            else:
                result["total_values"] = total
                result["note"] = (
                    f"Too many values ({total}) to return. "
                    f"Use 'sel' to subset or 'stat' to reduce. "
                    f"Showing first 100 flattened values."
                )
                result["sample_values"] = var_data.values.flatten()[:100].tolist()

        return to_json(result)
