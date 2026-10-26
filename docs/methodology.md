# Methodology

## Source

This project follows the CS50 AI PageRank specification
(https://cs50.harvard.edu/ai/projects/2/pagerank/) for the transition model, the random-surfer
sampler, the iterative formula and the dangling-page rule. It is an independently structured
project, not a copy of a course solution.

Deliberate differences from the course:

- the iteration stops at a 1e-8 tolerance rather than 0.001;
- the iteration count is recorded, and a `ConvergenceError` is raised at the iteration limit;
- links to unknown pages are rejected rather than silently dropped;
- the sampler takes explicit seeds;
- every result is checked against a direct linear solve.

## Algorithms (`src/pagerank_lab/core.py`)

| Function | Definition |
|---|---|
| `transition_model` | P(i → j) = (1−d)/N + d·1[i→j]/L(i); uniform 1/N from a dangling page |
| `sample_pagerank` | first state uniform; then `random_surfer_step` (a Bernoulli(d) coin, then a uniform link or a uniform page); estimate = visits / samples |
| `iterate_pagerank` | r₀ = 1/N; r_{k+1}(p) = (1−d)/N + d·[Σ_{i→p} r_k(i)/L(i) + Σ_{i dangling} r_k(i)/N]; simultaneous updates; stop when max\|r_{k+1} − r_k\| < tol |
| `solve_pagerank_exact` | solve (G − I)x = 0 with Σx = 1 (dense); used only as a check |
| `second_eigenvalue_modulus` | \|λ2\| of G, which controls the convergence rate |

The two-stage draw in the sampler is exactly the transition distribution:
P(j) = d·1[i→j]/L(i) + (1−d)/N, and 1/N from a dangling page.
`test_sampler_step_distribution_matches_transition_model` checks it empirically.

## Graphs (`src/pagerank_lab/graphs.py`)

**Benchmark graph** for N pages, with seed 20261026 + N:

- Zipf(1.0) popularity weights, randomly permuted over pages;
- each page is dangling with probability 0.10, with at least one guaranteed;
- every other page gets 1 + Poisson(3) distinct out-links;
- targets are drawn in proportion to popularity, never to the page itself.

Each graph is saved to `reports/inputs/`.

**Structure experiment** (256 pages):

- the benchmark mix;
- 50% dangling;
- Zipf(2.0) hubs;
- two communities joined by 4 bridge links;
- uniform targets with no dangling pages.

## Seeds

| Use | Seed |
|---|---|
| benchmark graph | 20261026 + N |
| benchmark sampling | 10,000,000 + 1000·N + round(100·d) |
| budget experiment | 30,000,000 + 1000·N + 100·log10(samples) + replicate |
| structure graphs | 20261026 + 7·(k+1) for structure k |
| structure sampling | 50,000,000 + 1000·k + 100·round(10·d) + replicate |
| numeric-vs-rank example | 70,000,001 |

## Metrics (`src/pagerank_lab/metrics.py`)

- **MAD** = mean over pages of |MC − iterative|; also the maximum absolute difference.
- **Spearman's ρ:** Pearson correlation of the average ranks (`scipy.stats.rankdata`, method
  "average"). It is `None`/"undefined" when either vector is constant. Undefined values are
  excluded from the median and minimum, and their count is reported.
- **Exact-tie fraction:** the share of pages whose iterative rank equals another page's rank,
  to 12 decimal places.

## Reproducibility

`scripts/run_all.py` writes deterministic files (metrics, inputs, figures, summary, CV evidence)
and separate timing files. Re-running reproduces the deterministic files byte for byte. This was
verified by running the full experiment twice and comparing SHA-256 hashes of every non-timing output.
