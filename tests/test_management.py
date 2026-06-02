"""Tests for model management tools."""

import json

import pytest

from arviz_mcp.tools.management import DescribeModelInput, LoadExampleInput


def test_load_example(ctx_empty, get_tool):
    fn = get_tool("load_example")
    out = json.loads(fn(LoadExampleInput(dataset="centered_eight"), ctx_empty))
    assert out["status"] == "loaded"
    assert out["name"] == "centered_eight"
    assert "posterior" in out["groups"]


def test_load_example_custom_name(ctx_empty, get_tool):
    fn = get_tool("load_example")
    out = json.loads(fn(LoadExampleInput(dataset="centered_eight", name="eight"), ctx_empty))
    assert out["name"] == "eight"


def test_load_example_unknown_dataset(ctx_empty, get_tool):
    fn = get_tool("load_example")
    out = json.loads(fn(LoadExampleInput(dataset="not_a_real_dataset"), ctx_empty))
    assert "error" in out
    assert "available_datasets" in out


def test_describe_model(ctx_with_models, get_tool):
    fn = get_tool("describe_model")
    out = json.loads(fn(DescribeModelInput(model="centered"), ctx_with_models))
    assert "groups" in out
    assert "posterior" in out["groups"]


def test_describe_model_unknown_group_raises(ctx_with_models, get_tool):
    fn = get_tool("describe_model")
    with pytest.raises(ValueError):
        fn(DescribeModelInput(model="centered", group="nope"), ctx_with_models)
