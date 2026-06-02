"""Tool registration for ArviZ MCP server."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def register_all(mcp: FastMCP) -> None:
    """Register all tool modules with the MCP server."""
    from arviz_mcp.tools import (
        comparison,
        data,
        diagnostics,
        insights,
        management,
        plotting,
        posterior,
    )

    management.register(mcp)
    diagnostics.register(mcp)
    posterior.register(mcp)
    comparison.register(mcp)
    plotting.register(mcp)
    data.register(mcp)
    insights.register(mcp)
