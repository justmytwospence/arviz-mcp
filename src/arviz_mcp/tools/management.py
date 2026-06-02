"""Model management tools: list, load, describe."""

from __future__ import annotations

from typing import TYPE_CHECKING

import arviz as az
from mcp.server.fastmcp import Context
from pydantic import BaseModel, Field

from arviz_mcp.models import load_datatree, model_info
from arviz_mcp.utils import MUTATING, READ_ONLY, get_model, get_registry, to_json

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


class LoadModelInput(BaseModel):
    name: str = Field(
        ...,
        description="Name to assign to this model (used to reference it in other tools)",
        min_length=1,
    )
    path: str = Field(
        ...,
        description=(
            "File path to load. Supports .nc (NetCDF), Zarr directories, "
            "and .csv (CmdStan, requires cmdstanpy). For CmdStan with "
            "multiple chains, use comma-separated paths."
        ),
    )


class LoadExampleInput(BaseModel):
    dataset: str = Field(
        ...,
        description=(
            "Name of the ArviZ example dataset to load. Examples: "
            "'centered_eight', 'non_centered_eight', 'radon', 'rugby', "
            "'glycan_torsion_angles'. Use without arguments to see all available datasets."
        ),
    )
    name: str | None = Field(
        default=None,
        description="Name to assign. Defaults to the dataset name.",
    )


class DescribeModelInput(BaseModel):
    model: str = Field(..., description="Name of the loaded model")
    group: str | None = Field(
        default=None,
        description="If specified, only describe this group (e.g., 'posterior', 'observed_data')",
    )


def register(mcp: FastMCP) -> None:
    @mcp.tool(
        name="list_models",
        annotations=READ_ONLY,
    )
    def list_models(ctx: Context) -> str:
        """List all currently loaded models with their groups and basic info.

        Returns the name, groups present, inference library, number of chains,
        and number of draws for each loaded model.
        """
        registry = get_registry(ctx)
        if not registry:
            return "No models loaded. Use the load_model tool to load a trace file."

        result = {}
        for name, dt in registry.items():
            info = model_info(dt)
            result[name] = {
                "groups": list(info["groups"].keys()),
                "chains": info.get("chains"),
                "draws": info.get("draws"),
                "inference_library": info.get("inference_library"),
            }
        return to_json(result)

    @mcp.tool(
        name="load_model",
        annotations=MUTATING,
    )
    def load_model(params: LoadModelInput, ctx: Context) -> str:
        """Load inference data from a file path.

        Supports NetCDF (.nc), Zarr directories, and CmdStan CSV files
        (requires cmdstanpy). The model is stored under the given name
        and can be referenced by that name in all other tools.
        """
        registry = get_registry(ctx)
        dt = load_datatree(params.path)
        registry[params.name] = dt
        info = model_info(dt)
        return to_json(
            {
                "status": "loaded",
                "name": params.name,
                "groups": list(info["groups"].keys()),
                "chains": info.get("chains"),
                "draws": info.get("draws"),
                "inference_library": info.get("inference_library"),
            }
        )

    @mcp.tool(
        name="load_example",
        annotations=MUTATING,
    )
    def load_example(params: LoadExampleInput, ctx: Context) -> str:
        """Load a built-in ArviZ example dataset.

        Loads one of ArviZ's bundled datasets (e.g., 'centered_eight',
        'non_centered_eight', 'radon', 'rugby') directly into the model
        registry. No file path needed.
        """
        available = az.list_datasets()
        if params.dataset not in available:
            return to_json(
                {
                    "error": f"Unknown dataset '{params.dataset}'.",
                    "available_datasets": available,
                }
            )

        registry = get_registry(ctx)
        model_name = params.name or params.dataset
        dt = az.load_arviz_data(params.dataset)
        registry[model_name] = dt
        info = model_info(dt)
        return to_json(
            {
                "status": "loaded",
                "name": model_name,
                "dataset": params.dataset,
                "groups": list(info["groups"].keys()),
                "chains": info.get("chains"),
                "draws": info.get("draws"),
                "inference_library": info.get("inference_library"),
            }
        )

    @mcp.tool(
        name="describe_model",
        annotations=READ_ONLY,
    )
    def describe_model(params: DescribeModelInput, ctx: Context) -> str:
        """Get detailed structure of a loaded model.

        Shows all groups, variables within each group, their dimensions
        and shapes. Optionally filter to a single group.
        """
        dt = get_model(ctx, params.model)
        info = model_info(dt)

        if params.group:
            if params.group not in info["groups"]:
                available = list(info["groups"].keys())
                raise ValueError(f"Group '{params.group}' not found. Available groups: {available}")
            return to_json({params.group: info["groups"][params.group]})

        return to_json(info)
