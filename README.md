# arviz-mcp

An [MCP](https://modelcontextprotocol.io) server for exploring Bayesian model
outputs with [ArviZ](https://www.arviz.org/). It lets an AI assistant load
`InferenceData` (NetCDF, Zarr, or CmdStan CSV), run MCMC diagnostics, query the
posterior, compare models, and produce plain-language findings and charts — so a
non-statistician can interrogate a fitted model in conversation.

## Requirements

- Python **3.12+** (the tools use the ArviZ 1.0 API).
- [uv](https://docs.astral.sh/uv/) for environment management.

## Installation

```bash
git clone https://github.com/spencerboucher/arviz-mcp.git
cd arviz-mcp
uv sync
```

## Running the server

```bash
# stdio transport (default) — what MCP clients launch
uv run arviz-mcp

# preload models at startup
uv run arviz-mcp --model centered:/path/to/centered.nc --model school:/path/to/school.nc

# scan a directory of .nc / Zarr files (filenames become model names)
uv run arviz-mcp --dir /path/to/models

# HTTP transport
uv run arviz-mcp --transport streamable-http --port 8000
```

CLI flags:

| Flag | Description |
| --- | --- |
| `--model NAME:PATH` | Load a model as `name:path`. For multi-chain CmdStan CSVs, use `name:/path/a.csv,/path/b.csv`. Repeatable. |
| `--dir PATH` | Scan a directory for `.nc` files and Zarr stores. |
| `--transport` | `stdio` (default) or `streamable-http`. |
| `--port` | Port for `streamable-http` (default `8000`). |

## Connecting an MCP client

Register the server under the name `arviz`. For a local checkout:

```json
{
  "mcpServers": {
    "arviz": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/arviz-mcp", "arviz-mcp", "--dir", "/path/to/models"]
    }
  }
}
```

If you install the package onto your `PATH` (e.g. `uv tool install .`), the entry
point simplifies to:

```json
{
  "mcpServers": {
    "arviz": { "command": "arviz-mcp", "args": ["--dir", "/path/to/models"] }
  }
}
```

No data files? Ask the assistant to call `load_example` (e.g. `centered_eight`)
to pull in a bundled ArviZ dataset.

## Tools

| Group | Tools |
| --- | --- |
| **Models** | `list_models`, `load_model`, `load_example`, `describe_model` |
| **Diagnostics** | `diagnostics` (divergences, tree depth, E-BFMI, R-hat, ESS), `summary` |
| **Posterior** | `hdi`, `posterior_stat` (mean/median/std/quantile/prob_above/prob_below), `compare_parameters` |
| **Model comparison** | `compare` (LOO-CV), `loo` |
| **Insights** | `key_findings` (ranked, with a trust assessment) |
| **Plots** | `plot_trace`, `plot_forest`, `plot_energy`, `plot_ppc`, `plot_pair`, `plot_loo_pit`, `plot_prior_posterior` |
| **Raw data** | `get_values` |

The server also exposes `arviz://models[/{name}[/{group}]]` resources and the
`explore_model`, `bayesian_workflow`, and `model_comparison` prompts.

## Example questions

- "Can I trust these results?" → runs `diagnostics` and translates the verdict.
- "What are the key findings?" → `key_findings`, narrated with confidence levels.
- "What's the probability the effect is positive?" → `posterior_stat` with `prob_above`.
- "Is parameter A bigger than B?" → `compare_parameters`.
- "Which model fits best?" → `compare`.

## Development

```bash
uv sync --all-groups        # install dev tools (pytest, ruff, mypy)
uv run python -m pytest     # run the test suite
uv run ruff check           # lint
uv run ruff format          # format
uv run mypy src             # type-check
```

Optionally enable the pre-commit hooks:

```bash
uv run pre-commit install
```

## License

[MIT](LICENSE)
