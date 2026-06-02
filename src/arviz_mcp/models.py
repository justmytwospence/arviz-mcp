"""Model loading, registry, and format detection."""

from __future__ import annotations

import os
from pathlib import Path

import arviz as az
from xarray import Dataset, DataTree


def detect_format(path: str) -> str:
    """Detect InferenceData format from file path.

    Returns one of: "netcdf", "zarr", "cmdstan".
    """
    p = Path(path)

    if p.suffix == ".nc":
        return "netcdf"
    if p.is_dir() and (p / ".zgroup").exists():
        return "zarr"
    if p.suffix == ".csv":
        return "cmdstan"

    raise ValueError(
        f"Cannot detect format for '{path}'. "
        "Supported: .nc (NetCDF), .csv (CmdStan), or Zarr directory."
    )


def load_datatree(path: str, fmt: str | None = None) -> DataTree:
    """Load a DataTree from a file path.

    For CmdStan CSV format, path can be a comma-separated list of per-chain
    CSV file paths (requires cmdstanpy).
    """
    # Handle comma-separated CmdStan CSVs
    if "," in path:
        csv_paths = [p.strip() for p in path.split(",")]
        for p in csv_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"CmdStan CSV not found: {p}")
        return az.from_cmdstanpy(csv_paths)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if fmt is None:
        fmt = detect_format(path)

    loaders = {
        "netcdf": az.from_netcdf,
        "zarr": az.from_zarr,
        "cmdstan": lambda p: az.from_cmdstanpy([p]),
    }

    loader = loaders.get(fmt)
    if loader is None:
        raise ValueError(f"Unknown format: {fmt}. Supported: {list(loaders)}")

    return loader(path)


def scan_directory(dir_path: str) -> dict[str, str]:
    """Scan a directory for loadable files.

    Returns a mapping of model name (filename without extension) to file path.
    """
    result: dict[str, str] = {}
    p = Path(dir_path)

    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    for f in sorted(p.iterdir()):
        if f.suffix == ".nc":
            result[f.stem] = str(f)
        elif f.is_dir() and (f / ".zgroup").exists():
            result[f.name] = str(f)

    return result


def _clean_group_name(name: str) -> str:
    """Strip leading slash from DataTree group names."""
    return name.lstrip("/")


def get_group_names(dt: DataTree) -> list[str]:
    """Get clean group names from a DataTree (excluding root)."""
    return [_clean_group_name(g) for g in dt.groups if g != "/"]


def get_group_dataset(dt: DataTree, group: str) -> Dataset:
    """Get the xarray Dataset for a group in the DataTree."""
    # Try with and without leading slash
    for key in [f"/{group}", group]:
        if key in dt.groups:
            return dt[key].ds
    available = get_group_names(dt)
    raise KeyError(f"Group '{group}' not found. Available groups: {available}")


def model_info(dt: DataTree) -> dict:
    """Extract metadata about a DataTree."""
    info: dict = {"groups": {}}
    group_names = get_group_names(dt)

    for group_name in group_names:
        ds = get_group_dataset(dt, group_name)
        group_info: dict = {"variables": {}}

        for var_name in ds.data_vars:
            var = ds[var_name]
            group_info["variables"][var_name] = {
                "dims": list(var.dims),
                "shape": list(var.shape),
            }

        group_info["dims"] = dict(ds.sizes)

        if "chain" in ds.sizes and "draw" in ds.sizes:
            group_info["chains"] = int(ds.sizes["chain"])
            group_info["draws"] = int(ds.sizes["draw"])

        info["groups"][group_name] = group_info

    # Top-level summary from posterior
    if "posterior" in group_names:
        post_ds = get_group_dataset(dt, "posterior")
        info["chains"] = int(post_ds.sizes.get("chain", 0))
        info["draws"] = int(post_ds.sizes.get("draw", 0))

        attrs = dict(post_ds.attrs)
        if "inference_library" in attrs:
            info["inference_library"] = attrs["inference_library"]
        if "inference_library_version" in attrs:
            info["inference_library_version"] = attrs["inference_library_version"]

    return info
