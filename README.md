# PageRank Lab

Two independent ways to compute PageRank, compared rigorously on reproducible synthetic graphs:

1. **Monte Carlo:** a seeded random surfer walks the Markov chain, and visit frequencies estimate PageRank.
2. **Iteration:** simultaneous fixed-point updates until max |change| < 1e-8, then checked against
   a direct linear solve before being used as the reference.

This is a study of probabilistic algorithms, numerical convergence, stochastic estimation and
computational cost. All graphs are generated locally; nothing is crawled from the internet.
The small HTML crawler reads only local files.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.12 and the locked numpy, scipy
and matplotlib (plus pytest).

```bash
cd pagerank-lab
uv sync
uv run pytest                                  # fast unit + integration tests
uv run python scripts/run_all.py               # full experiment (100,000-sample benchmark), about 1 minute
uv run python scripts/run_all.py --quick       # reduced samples for development -> reports/quick/
uv run python scripts/run_benchmark.py         # only the 21-scenario benchmark
```

## What is measured

- **21-scenario benchmark:** graph sizes 8, 16, 32, 64, 128, 256 and 1,000 pages × damping 0.70,
  0.85 and 0.95, with 100,000 samples each. For every scenario it records:
  - pages, links and dangling pages;
  - seeds;
  - iterations;
  - the iterative-vs-direct check;
  - mean absolute difference (MAD);
  - Spearman's rho (tie-aware; undefined values never replaced);
  - the exact-tie fraction;
  - sampling and iteration time, kept in a separate timing file.
- **Sampling-budget experiment:** 1,000, 10,000 and 100,000 samples × 5 seeds on 32-, 256- and
  1,000-page graphs. It measures error, rank correlation, runtime and the error-vs-budget slope.
- **Graph-structure sensitivity:** many dangling pages, hub-dominated, two weakly linked
  communities and uniform targets, each at three damping factors. It explains convergence with
  the second eigenvalue |λ2| of the Google matrix.
- **Numerical vs ranking accuracy:** real examples of small error with low rank correlation, and
  of large error with perfect rank correlation.

## Outputs (`reports/`)

| File | Content | Deterministic? |
|---|---|---|
| `results_summary.md` | methodology, headline statistics, all 21 scenarios, experiments, limitations | yes |
| `benchmark_results.csv` | one row per scenario (21 rows) | yes |
| `budget_results.csv`, `budget_summary.csv` | sampling-budget experiment | yes |
| `structure_results.csv`, `numeric_vs_rank.csv` | structure experiment and illustration | yes |
| `inputs/benchmark_graph_N.json` | the exact benchmark graphs | yes |
| `figures/*.png` | sampling error and rank correlation by size and damping, plus the budget chart | yes |
| `cv_evidence.json` | metric definitions, values, seeds, commands, evidence files, CV-figure check | yes |
| `*_timings.csv`, `timing_report.md` | wall-clock timings | **no** (runtime metadata) |

## Layout

```
src/pagerank_lab/
  crawl.py       local HTML crawler (no network; safe file-name resolution)
  core.py        transition model, Monte Carlo sampler, iterative PageRank, direct-solve check
  graphs.py      deterministic synthetic graph generators
  metrics.py     mean absolute difference, tie-aware Spearman (undefined handled), aggregation
  benchmark.py   21 scenarios, budget and structure experiments, seed strategy
  reporting.py   CSV and chart writers
data/html_demo/  4 hand-written pages with crawler edge cases
scripts/         run_benchmark.py, run_all.py
docs/            methodology.md, interview_preparation.md
```