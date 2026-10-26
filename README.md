# PageRank Lab

PageRank can be computed in two completely different ways. You can simulate a random surfer
clicking links and count where they end up, or you can solve for the fixed point directly with
iteration. I implemented both from scratch and put them head to head on reproducible graphs. I
wanted to see how fast each converges, how noisy the sampling is, and when the two actually
agree on the ranking.

It's based on the PageRank project from Harvard's CS50 AI course, which I extended into a proper
benchmark and a few targeted experiments.

## How it works

- **Random surfer (Monte Carlo):** with probability d the surfer follows a random link on the
  current page, otherwise they jump to any page at random. A page with no links sends them
  anywhere. Visit frequencies over 100,000 seeded steps estimate PageRank.
- **Iteration:** start everyone at 1/N and apply the PageRank update to all pages at once until
  nothing changes by more than 1e-8.
- **Trust, but verify:** before I use the iterative answer as the reference, I check it against
  a direct linear solve of the stationary equations. Stopping when the changes get small is not
  the same thing as being accurate.
- **Crawler:** a small crawler reads a local folder of HTML pages, with no network access at
  all. It safely ignores external links, fragments, self-links, duplicates and missing pages.

## Results

**Benchmark:** 21 scenarios, made of 7 graph sizes (8 to 1,000 pages) times 3 damping factors
(0.70, 0.85 and 0.95), with 100,000 samples each.

| What I measured | Result |
|---|---|
| Iterations to converge (tolerance 1e-8) | median **18**, max **26** |
| Mean absolute gap between sampling and iteration | median **0.00027** |
| Spearman rank correlation | median **0.994**, min **0.852** |
| Iterative vs direct solve | within **7.8e-9** in every scenario |

The interesting part is what happens as the graphs grow:

- **Numbers get closer, rankings get worse.** On the 1,000-page graphs, sampling has the
  smallest numerical error of all, yet the lowest rank correlation. The cause is exact ties. In
  the 1,000-page graphs, half the pages have *identical* PageRank, because every page with no
  incoming links gets the same value. Random sampling can't reproduce a tie, so it shuffles those
  pages. Small error and accurate ranking are different things.
- **The error falls like 1/√samples.** I ran budgets of 1,000, 10,000 and 100,000 samples with 5
  seeds each. The fitted log-log slope came out at -0.50, so 10x more samples buys about 3x less
  error.
- **Graph structure drives convergence.** Two dense communities joined by just 4 links needed 52
  iterations at d = 0.95, because the Google matrix's second eigenvalue sits close to d.
- **The stopping rule can miss the true answer.** On those weakly connected graphs, the
  iterative answer landed up to 3.7e-8 from the true answer even with a 1e-8 tolerance. The
  stopping rule bounds the last change, not the error, which is exactly why the direct-solve
  check is there.

## Running it

You need [uv](https://docs.astral.sh/uv/), which sets up Python 3.12 and the locked dependencies.

```bash
cd pagerank-lab
uv sync
uv run pytest                              # 65 tests, a few seconds
uv run python scripts/run_all.py           # the full experiment, about a minute
uv run python scripts/run_all.py --quick   # small sample counts while developing
```

Everything lands in `reports/`:
- the 21-scenario CSV;
- the sampling-budget and graph-structure results;
- the exact input graphs;
- charts of sampling error and rank correlation by graph size and damping;
- `results_summary.md`;
- `cv_evidence.json`.

Every seed is written down, so re-running gives byte-identical results. Only the timing files
differ between runs.

## Layout

```
src/pagerank_lab/
  crawl.py       local HTML crawler
  core.py        transition model, random surfer, iteration, direct-solve check
  graphs.py      seeded synthetic graph generators
  metrics.py     mean absolute error, tie-aware Spearman, aggregation
  benchmark.py   the 21 scenarios and the extra experiments
  reporting.py   CSVs and charts
data/html_demo/  four hand-written pages full of crawler edge cases
scripts/         run_benchmark.py, run_all.py
tests/
docs/methodology.md
```
