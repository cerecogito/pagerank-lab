"""Run the 21-scenario benchmark (7 graph sizes x 3 damping factors).

Usage:
    uv run python scripts/run_benchmark.py            # full: 100,000 samples per scenario -> reports/
    uv run python scripts/run_benchmark.py --quick    # development: 2,000 samples -> reports/quick/

Writes benchmark_results.csv (deterministic metrics), benchmark_timings.csv (runtime metadata),
the benchmark graphs as edge lists (inputs/), and the two benchmark charts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pagerank_lab.benchmark import (
    FULL_SAMPLES, GRAPH_SIZES, QUICK_SAMPLES, TOLERANCE, benchmark_graph, run_benchmark,
)
from pagerank_lab.reporting import chart_by_size_and_damping, write_csv

ROOT = Path(__file__).resolve().parents[1]


def output_dir(quick: bool) -> Path:
    return ROOT / "reports" / ("quick" if quick else "")


def save_inputs(directory: Path) -> None:
    """Save every benchmark graph so the exact inputs can be inspected and reused."""
    inputs = directory / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for n_pages in GRAPH_SIZES:
        corpus = benchmark_graph(n_pages)
        edges = {page: sorted(links) for page, links in sorted(corpus.items())}
        (inputs / f"benchmark_graph_{n_pages}.json").write_text(json.dumps(edges, indent=0) + "\n",
                                                                encoding="utf-8")


def main(quick: bool = False) -> Path:
    directory = output_dir(quick)
    samples = QUICK_SAMPLES if quick else FULL_SAMPLES
    print(f"running 21 scenarios with {samples:,} samples each (tolerance {TOLERANCE})")
    rows, timings = run_benchmark(samples=samples, tolerance=TOLERANCE)
    write_csv(directory / "benchmark_results.csv", rows)
    write_csv(directory / "benchmark_timings.csv", timings)
    save_inputs(directory)
    chart_by_size_and_damping(rows, "mean_abs_difference", "mean |Monte Carlo - iterative|",
                              f"Sampling error by graph size ({samples:,} samples per scenario)",
                              directory / "figures" / "sampling_error_by_size.png", log_y=True)
    chart_by_size_and_damping(rows, "spearman", "Spearman rank correlation",
                              f"Rank agreement by graph size ({samples:,} samples per scenario)",
                              directory / "figures" / "rank_correlation_by_size.png")
    print(f"wrote {len(rows)} scenario rows to {directory / 'benchmark_results.csv'}")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="2,000 samples per scenario, output to reports/quick/")
    main(parser.parse_args().quick)
