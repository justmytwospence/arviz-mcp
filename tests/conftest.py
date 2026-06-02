"""Test fixtures for ArviZ MCP server."""

import matplotlib

matplotlib.use("Agg")

import arviz as az
import pytest


@pytest.fixture(scope="session")
def centered_eight():
    return az.load_arviz_data("centered_eight")


@pytest.fixture(scope="session")
def non_centered_eight():
    return az.load_arviz_data("non_centered_eight")


class MockLifespanContext:
    """Mock for ctx.request_context.lifespan_context."""

    def __init__(self, models):
        self.lifespan_context = {"models": models}


class MockContext:
    """Mock MCP Context that provides access to model registry."""

    def __init__(self, models):
        self.request_context = MockLifespanContext(models)


@pytest.fixture
def ctx_with_models(centered_eight, non_centered_eight):
    """Context with centered_eight and non_centered_eight loaded."""
    models = {
        "centered": centered_eight,
        "non_centered": non_centered_eight,
    }
    return MockContext(models)


@pytest.fixture
def ctx_empty():
    """Context with no models loaded."""
    return MockContext({})


@pytest.fixture(scope="session")
def registered_mcp():
    """A FastMCP instance with all tools registered, for tool-level tests."""
    from mcp.server.fastmcp import FastMCP

    from arviz_mcp.tools import register_all

    mcp = FastMCP("arviz_mcp_test")
    register_all(mcp)
    return mcp


@pytest.fixture
def get_tool(registered_mcp):
    """Return a getter that fetches a registered tool's underlying function by name."""

    def _get(name):
        tool = registered_mcp._tool_manager.get_tool(name)
        assert tool is not None, f"Tool '{name}' is not registered"
        return tool.fn

    return _get
