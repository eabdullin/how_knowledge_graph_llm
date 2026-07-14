# How Should LLMs See a Knowledge Graph?

This repository is the public research artifact for experiments comparing four ways of giving a language model domain knowledge during information-elicitation dialogue:

- a rendered knowledge-graph image;
- a Mermaid-like text serialization;
- graph-navigation tools; and
- a no-graph control.

The bundled benchmark is the exact 30-problem MAMO subset used for the reported experiment: 15 Production Planning problems and 15 Traveling Salesman Problem problems. The same two graphs are used in image, text, and tool form.

## Install

Python 3.12 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync --locked
cp .env.example .env
```

Add `OPENAI_API_KEY` to `.env` for GPT runs. Gemma can be served by any provider exposing an OpenAI-compatible Chat Completions API: set `OPENAI_COMPATIBLE_BASE_URL` and, when required by the provider, `OPENAI_COMPATIBLE_API_KEY`; then adjust the two model IDs if your provider uses different names. Local services that do not authenticate can leave the compatible API key empty.

Providers with separate deployments for the two Gemma variants can instead set the optional `GEMMA_31B_*` and `GEMMA_E4B_*` base-URL/API-key overrides shown in `.env.example`. The shared values remain the fallback.

## Validate without API calls

The dry run loads all benchmark records, validates the 15/15 topic split and every graph representation, resolves the requested models, and prints the planned manifest without creating results or calling a provider:

```bash
uv run python -m experiments.run_experiment --dry-run
```

Run the offline test suite with:

```bash
uv run python -m unittest discover -s tests -v
```

## Small experiment

This command runs one no-graph dialogue on one benchmark problem. It makes paid API calls for the modeller, simulated user, and judge.

```bash
uv run python -m experiments.run_experiment \
  --modality no_graph \
  --model gpt-5.4-mini-medium \
  --problem mamo_complex_tsp_courier_5_cities_e_i \
  --runs 1 \
  --output results/smoke
```

Results are written incrementally as `raw_results.json`, with a final `summary.json` and readable dialogue files under the selected output directory. Use `--resume` to continue an interrupted run.

To generate the fuller Markdown report from a completed direct or sharded run:

```bash
uv run python -m experiments.report_results \
  --run-dir results/paper_run \
  --expected-total 2400 \
  --runs-per-problem 5 \
  --max-turns 40 \
  --model multi-model
```

## Reported experiment matrix

The following command defines the paper's four modalities, four modeller labels, 30 problems, and five trials: 2,400 planned conversations in total. It is intentionally not run as part of repository verification.

```bash
uv run python -m experiments.run_experiment \
  --dataset benchmark \
  --modality image text tools no_graph \
  --model gpt-5.4-nano-medium gpt-5.4-mini-medium gemma-4-E4B-it gemma-4-31b-it \
  --runs 5 \
  --max-turns 40 \
  --user-model gpt-5.4-none \
  --judge-model gpt-5.4-medium \
  --output results/paper_run \
  --resume
```

Add `--dry-run` first to validate this full matrix and confirm `planned_conversations` is 2400.

### Historical model-label disclosure

The configuration recorded as `gpt-5.4-nano-medium` in the original experiment code actually invoked backend model `gpt-5.4-mini` with `low` reasoning effort. For reproducibility, the historical label remains accepted and emits a warning. The honest canonical name for that executed configuration is `gpt-5.4-mini-low`. The public artifact does not silently replace it with a different nano configuration because doing so would no longer reproduce the recorded run.

## Repository contents

```text
data/benchmark.jsonl       exact 30-problem benchmark
knowledge_graphs/          two graphs in DOT, Mermaid, and PNG formats
experiments/               runner, prompts, modalities, evaluation, reporting
knowledge_graph.py         graph parser and traversal API
tests/                     offline regression tests
```

Raw model outputs, paper submission/reviewer files, logs, local operational scripts, deprecated assets, and third-party PDFs are intentionally excluded.

## Data and asset provenance

The benchmark is a selected and normalized derivative of [MAMO](https://github.com/FreedomIntelligence/Mamo), licensed CC BY-SA 4.0. The two domain graphs are adapted from [AgentMILO](https://github.com/arc2022-deakin/AgentMILO), licensed Apache 2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [data/README.md](data/README.md) for attribution and transformation details.

The repository's original code is licensed under the Apache License 2.0. The MAMO-derived benchmark remains under CC BY-SA 4.0.

## Citation

If you use this repository, please cite the associated paper. Replace the placeholders below once the publication details are available:

```bibtex
@inproceedings{citation-key-to-be-updated,
  title     = {How Should LLMs See a Knowledge Graph? Comparing Text, Image, and Tool-Based Modalities for Information Elicitation},
  author    = {Authors to be added},
  booktitle = {Venue to be added},
  year      = {Year to be added},
  url       = {Paper URL to be added}
}
```
