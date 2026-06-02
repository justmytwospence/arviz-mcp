"""Shared MCMC diagnostic computations.

Both the ``diagnostics`` tool and the ``key_findings`` trust assessment rely on
these helpers so the two never drift on thresholds or how a metric is computed.
Each function raises on failure; callers decide how to surface that.
"""

from __future__ import annotations

import arviz as az
import numpy as np
from xarray import DataTree

from arviz_mcp.models import get_group_dataset

# Diagnostic thresholds -- single source of truth.
RHAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400
BFMI_THRESHOLD = 0.3


def bfmi_values(dt: DataTree) -> list[float]:
    """Return the per-chain E-BFMI values (rounded).

    In ArviZ 1.0 ``az.bfmi`` returns a DataTree carrying the per-chain values
    under ``sample_stats/energy`` (not a bare array).
    """
    bfmi = az.bfmi(dt)
    arr = np.asarray(get_group_dataset(bfmi, "sample_stats")["energy"].values)
    return [round(float(v), 4) for v in arr.flatten()]


def bad_rhat(dt: DataTree, var_names: list[str] | None = None) -> dict[str, float]:
    """Map each variable whose max R-hat exceeds RHAT_THRESHOLD to that value."""
    rhat = az.rhat(dt, var_names=var_names)
    bad: dict[str, float] = {}
    for name in rhat.data_vars:
        vals = rhat[name].values
        max_val = float(vals) if vals.ndim == 0 else float(np.nanmax(vals))
        if max_val > RHAT_THRESHOLD:
            bad[str(name)] = round(max_val, 4)
    return bad


def low_ess(
    dt: DataTree, var_names: list[str] | None = None, method: str = "bulk"
) -> dict[str, float]:
    """Map each variable whose minimum ESS falls below ESS_THRESHOLD to that value."""
    ess = az.ess(dt, var_names=var_names, method=method)
    low: dict[str, float] = {}
    for name in ess.data_vars:
        min_val = float(np.nanmin(ess[name].values))
        if min_val < ESS_THRESHOLD:
            low[str(name)] = round(min_val, 0)
    return low
