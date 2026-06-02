"""Shared utilities for ArviZ MCP server."""

from __future__ import annotations

import io
import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mcp.server.fastmcp import Image
from mcp.types import ToolAnnotations
from xarray import Dataset, DataTree

from arviz_mcp.models import get_group_dataset, get_group_names

# Shared tool annotations. Most tools are read-only and side-effect free; the
# loaders mutate the in-memory model registry.
READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
MUTATING = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


class ModelNotFoundError(Exception):
    pass


class GroupNotFoundError(Exception):
    pass


class VariableNotFoundError(Exception):
    pass


def get_registry(ctx) -> dict[str, DataTree]:
    """Get the model registry from the MCP context."""
    return ctx.request_context.lifespan_context["models"]


def get_model(ctx, name: str) -> DataTree:
    """Get a model by name from the registry. Raises with available models on miss."""
    registry = get_registry(ctx)
    if name not in registry:
        available = list(registry.keys())
        raise ModelNotFoundError(f"Model '{name}' not found. Available models: {available}")
    return registry[name]


def get_group(dt: DataTree, group: str, model_name: str = "") -> Dataset:
    """Get a group dataset from a DataTree. Raises with available groups on miss."""
    try:
        return get_group_dataset(dt, group)
    except KeyError:
        available = get_group_names(dt)
        raise GroupNotFoundError(
            f"Group '{group}' not found in model '{model_name}'. Available groups: {available}"
        ) from None


def validate_var_names(dataset, var_names: list[str], group: str = "") -> None:
    """Validate that variable names exist in a dataset."""
    available = list(dataset.data_vars)
    missing = [v for v in var_names if v not in available]
    if missing:
        raise VariableNotFoundError(
            f"Variables not found in group '{group}': {missing}. Available variables: {available}"
        )


def plot_collection_to_image(pc) -> Image:
    """Convert an ArviZ PlotCollection to a FastMCP Image."""
    fig = pc.viz["figure"].item()
    return fig_to_image(fig)


def fig_to_image(fig: plt.Figure) -> Image:
    """Convert a matplotlib figure to a FastMCP Image."""
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        return Image(data=buf.getvalue(), format="png")
    finally:
        plt.close(fig)


def format_dataframe(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Format a DataFrame as a readable string table."""
    if len(df) > max_rows:
        return f"Showing first {max_rows} of {len(df)} rows:\n\n" + df.head(max_rows).to_string()
    return df.to_string()


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def to_json(obj: Any) -> str:
    """Serialize to JSON with NumPy type support."""
    return json.dumps(obj, cls=NumpyEncoder, indent=2)


def _evidence_strength(prob: float) -> str:
    """Categorize a probability into an evidence strength label."""
    if prob >= 0.99:
        return "very strong"
    elif prob >= 0.95:
        return "strong"
    elif prob >= 0.90:
        return "moderate"
    elif prob >= 0.75:
        return "weak"
    else:
        return "inconclusive"
