"""Tests for model loading and metadata."""

import pytest

from arviz_mcp.models import detect_format, get_group_dataset, get_group_names, model_info


def test_detect_format_netcdf():
    assert detect_format("trace.nc") == "netcdf"


def test_detect_format_csv():
    assert detect_format("output.csv") == "cmdstan"


def test_detect_format_unknown():
    with pytest.raises(ValueError, match="Cannot detect format"):
        detect_format("data.pkl")


def test_model_info(centered_eight):
    info = model_info(centered_eight)
    assert "posterior" in info["groups"]
    assert "sample_stats" in info["groups"]
    assert info["chains"] == 4
    assert info["draws"] == 500
    assert info["inference_library"] == "pymc"

    post = info["groups"]["posterior"]
    assert "mu" in post["variables"]
    assert "tau" in post["variables"]
    assert "theta" in post["variables"]


def test_get_group_names(centered_eight):
    names = get_group_names(centered_eight)
    assert "posterior" in names
    assert "sample_stats" in names
    assert "observed_data" in names


def test_get_group_dataset(centered_eight):
    ds = get_group_dataset(centered_eight, "posterior")
    assert "mu" in ds.data_vars
    assert "chain" in ds.sizes
    assert "draw" in ds.sizes


def test_get_group_dataset_missing(centered_eight):
    with pytest.raises(KeyError, match="not found"):
        get_group_dataset(centered_eight, "nonexistent")
