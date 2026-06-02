"""ArviZ MCP server — conversational exploration of Bayesian model outputs."""

from __future__ import annotations

import argparse
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import matplotlib

matplotlib.use("Agg")

from mcp.server.fastmcp import FastMCP  # noqa: E402
from xarray import DataTree  # noqa: E402

from arviz_mcp.models import (  # noqa: E402
    load_datatree,
    model_info,
    scan_directory,
)
from arviz_mcp.tools import register_all  # noqa: E402
from arviz_mcp.utils import to_json  # noqa: E402

logger = logging.getLogger("arviz_mcp")

INSTRUCTIONS = """\
You are helping a non-statistician explore the results of a Bayesian statistical \
model. Your goal is to give rigorous, correctly nuanced answers in plain language \
that a business person can act on.

## Communication style

- Lead with what the results mean, not how they were computed.
- Use plain language. When a technical term is unavoidable, define it in parentheses \
on first use (e.g., "credible interval (the range of plausible values)").
- Frame findings in terms of confidence and practical implications, not p-values or \
raw numbers.
- Keep responses concise. Offer to go deeper rather than overwhelming upfront.

## Trustworthiness first

Before presenting any findings, always run the `diagnostics` tool to check whether \
the model's results are reliable. Translate the assessment for the user:
- "healthy" -> "The model's results are reliable -- the estimation process ran well."
- "has warnings" -> "The results are mostly reliable, but there are minor concerns \
worth noting."
- "problematic" -> "There are significant concerns about these results -- interpret \
with caution."

Present this verdict briefly before diving into findings. Do not skip this step.

## How to present findings

- For each parameter estimate, report the best estimate (posterior mean) and the \
range of plausible values (94% credible interval) in natural language.
- Proactively compute the probability that a parameter is positive or negative \
(using `posterior_stat` with `prob_above` or `prob_below` and value 0) whenever \
the sign matters for the user's question.
- Use the `strength` field from probability results to calibrate your language: \
"very strong evidence" (>99%), "strong evidence" (>95%), "moderate evidence" (>90%), \
"weak evidence" (>75%), "inconclusive" (<75%).
- When comparing parameters, use the `compare_parameters` tool rather than asking \
the user to eyeball overlapping intervals.

## Answering common questions

- "Is this model reliable?" / "Can I trust these results?" -> Run `diagnostics`. \
Use the interpretation hints in the response to explain each check.
- "What are the key findings?" / "What does this model tell me?" -> Use \
`key_findings` to get a ranked summary, then narrate the top findings with \
confidence levels.
- "Is parameter A bigger than B?" -> Use `compare_parameters`.
- "What's the probability that X is positive/negative/above Y?" -> Use \
`posterior_stat` with `prob_above` or `prob_below`.
- "Show me the results" -> Use `summary` for a table, or `plot_forest` for a \
visual overview. Explain what the visualization shows.

## Uncertainty communication

Never present a point estimate without its uncertainty. Instead of "the effect is \
2.3", say "the effect is estimated at 2.3, with plausible values ranging from 1.3 \
to 3.2 (94% credible interval)". This is the core value of Bayesian analysis -- \
always honor it.

## What to avoid

- Do not use "statistically significant" -- this is a frequentist concept that \
does not apply here.
- Do not present raw JSON or tables without interpretation.
- Do not assume the user knows what "posterior", "HDI", "R-hat", or "ESS" mean.
- Do not hedge excessively. If the evidence is strong, say so clearly.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ArviZ MCP server for exploring Bayesian model outputs",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME:PATH",
        help=(
            "Load a model as name:path. For CmdStan CSVs with multiple chains, "
            "use name:/path/chain1.csv,/path/chain2.csv. Can be repeated."
        ),
    )
    parser.add_argument(
        "--dir",
        default=None,
        metavar="PATH",
        help="Scan a directory for .nc files (filenames become model names).",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
    )
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def _parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse 'name:path' model specification."""
    if ":" not in spec:
        raise ValueError(f"Invalid model spec '{spec}'. Expected format: name:/path/to/file")
    name, path = spec.split(":", 1)
    if not name or not path:
        raise ValueError(f"Invalid model spec '{spec}'. Both name and path are required.")
    return name, path


def make_lifespan(
    model_specs: list[str], trace_dir: str | None
) -> Callable[[FastMCP], AbstractAsyncContextManager[dict]]:
    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[dict]:
        models: dict[str, DataTree] = {}

        # Load explicit --model args
        for spec in model_specs:
            name, path = _parse_model_spec(spec)
            logger.info(f"Loading model '{name}' from {path}")
            models[name] = load_datatree(path)
            info = model_info(models[name])
            logger.info(
                f"  Loaded: {list(info['groups'].keys())}, "
                f"chains={info.get('chains')}, draws={info.get('draws')}"
            )

        # Load from --dir
        if trace_dir:
            found = scan_directory(trace_dir)
            for name, path in found.items():
                if name not in models:
                    logger.info(f"Loading model '{name}' from {path}")
                    models[name] = load_datatree(path)

        if models:
            logger.info(f"Ready with {len(models)} model(s): {list(models.keys())}")
        else:
            logger.info("No models pre-loaded. Use load_model tool to load data.")

        yield {"models": models}

    return lifespan


def create_server(lifespan) -> FastMCP:
    mcp = FastMCP(
        "arviz_mcp",
        instructions=INSTRUCTIONS,
        lifespan=lifespan,
    )

    # Register all tools
    register_all(mcp)

    # -- Resources --
    @mcp.resource("arviz://models")
    def list_model_resources() -> str:
        """List all loaded models."""
        return to_json({"note": "Use the list_models tool for model info."})

    @mcp.resource("arviz://models/{name}")
    def get_model_resource(name: str) -> str:
        """Get structure of a specific model."""
        return to_json({"note": f"Use describe_model tool with model='{name}'."})

    @mcp.resource("arviz://models/{name}/{group}")
    def get_model_group_resource(name: str, group: str) -> str:
        """Get variables and dimensions in a specific group."""
        return to_json({"note": f"Use describe_model tool with model='{name}', group='{group}'."})

    # -- Prompts --
    @mcp.prompt()
    def explore_model(model: str) -> str:
        """Explore a model's findings in plain language for a non-technical audience."""
        return (
            f"I have a Bayesian model called '{model}' and I'd like to understand "
            "what it tells me. Please:\n\n"
            "1. First check if the model's results are trustworthy (run diagnostics "
            "and just tell me the verdict briefly)\n"
            "2. Show me the key findings -- what are the main effects and how "
            "confident should I be in each one?\n"
            "3. Use plain language throughout -- no statistical jargon\n"
            "4. For each finding, tell me the probability and practical meaning\n"
            "5. Highlight anything surprising or that I should pay special attention to"
        )

    @mcp.prompt()
    def bayesian_workflow(model: str) -> str:
        """Walk through a complete diagnostic and analysis workflow."""
        return (
            f"I have a fitted Bayesian model called '{model}'. Please walk me "
            "through a full analysis:\n\n"
            "1. Describe the model structure so I know what was estimated\n"
            "2. Check if the results are trustworthy (run diagnostics and explain "
            "the verdict in plain language)\n"
            "3. Show trace plots for key parameters to visualize the estimation\n"
            "4. Summarize the main findings with confidence levels\n"
            "5. If posterior predictive data is available, check how well the model "
            "fits the observed data\n"
            "6. Give me an overall assessment: what can I confidently conclude, and "
            "what remains uncertain?"
        )

    @mcp.prompt()
    def model_comparison(model_names: str) -> str:
        """Compare multiple models to find which best explains the data."""
        return (
            f"I have the following fitted models: {model_names}. "
            "Please compare them and help me decide which to use:\n\n"
            "1. Briefly describe what each model includes\n"
            "2. Check that each model's results are trustworthy\n"
            "3. Compare them to determine which fits the data better and by how much\n"
            "4. Show a visual comparison of key parameters\n"
            "5. Give me a clear recommendation: which model should I use, and how "
            "confident should I be in that choice?"
        )

    return mcp


def main() -> None:
    args = parse_args()
    lifespan = make_lifespan(args.model, args.dir)
    mcp = create_server(lifespan)

    if args.transport == "streamable-http":
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
