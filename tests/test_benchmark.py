"""Tests for the benchmark design, aggregation and a reduced-sample integration run."""

import pytest

from pagerank_lab.benchmark import (
    DAMPING_FACTORS, GRAPH_SIZES, aggregate_benchmark, benchmark_graph, benchmark_scenarios, run_benchmark,
    tied_fraction,
)
from pagerank_lab.core import compile_corpus
from pagerank_lab.graphs import count_dangling, skewed_random_graph, two_community_graph


def test_exactly_21_unique_scenarios_with_unique_seeds():
    scenarios = benchmark_scenarios()
    assert len(scenarios) == 21 == len(GRAPH_SIZES) * len(DAMPING_FACTORS)
    assert len({s.scenario_id for s in scenarios}) == 21
    assert len({s.sampling_seed for s in scenarios}) == 21


def test_graphs_are_deterministic_varied_and_have_dangling_pages():
    for n in GRAPH_SIZES:
        first, second = benchmark_graph(n), benchmark_graph(n)
        assert first == second
        assert count_dangling(first) >= 1
        compile_corpus(first)                                # valid: no unknown links
        assert all(page not in links for page, links in first.items())
    in_degrees = {}
    for links in benchmark_graph(256).values():
        for target in links:
            in_degrees[target] = in_degrees.get(target, 0) + 1
    assert max(in_degrees.values()) >= 5 * (sum(in_degrees.values()) / 256)   # skewed, not uniform


def test_two_community_graph_has_few_bridges():
    corpus = two_community_graph(64, seed=1, bridges=4)
    half = {f"p{i:04d}" for i in range(32)}
    crossing = sum(1 for page, links in corpus.items() for t in links if (page in half) != (t in half))
    assert crossing == 4


def test_aggregation_excludes_undefined_spearman():
    rows = [{"iterations": "10", "mean_abs_difference": "0.001", "spearman": "0.9",
             "iterative_vs_direct_max_abs": "1e-9"},
            {"iterations": "12", "mean_abs_difference": "0.002", "spearman": "undefined",
             "iterative_vs_direct_max_abs": "2e-9"},
            {"iterations": "20", "mean_abs_difference": "0.003", "spearman": "0.7",
             "iterative_vs_direct_max_abs": "3e-9"}]
    aggregate = aggregate_benchmark(rows)
    assert aggregate["iterations_median"] == 12 and aggregate["iterations_max"] == 20
    assert aggregate["spearman"] == {"defined": 2, "undefined": 1, "median": 0.8, "min": 0.7, "max": 0.9}


def test_tied_fraction():
    assert tied_fraction({"a": 0.2, "b": 0.2, "c": 0.6}) == pytest.approx(2 / 3)
    assert tied_fraction({"a": 0.1, "b": 0.3, "c": 0.6}) == 0.0


def test_integration_reduced_sample_benchmark():
    """Full 21-scenario pipeline with 500 samples (the 100,000-sample run is the benchmark script)."""
    rows, timings = run_benchmark(samples=500)
    assert len(rows) == len(timings) == 21
    for row in rows:
        assert row["samples"] == 500
        assert row["sampled_rank_sum"] == pytest.approx(1.0) and row["iterative_rank_sum"] == pytest.approx(1.0)
        assert row["iterative_vs_direct_max_abs"] < 1e-7
        assert row["spearman"] == "undefined" or -1.0 <= row["spearman"] <= 1.0
    again, _ = run_benchmark(samples=500)
    assert again == rows                                    # same seeds -> same metrics (timings excluded)
    assert skewed_random_graph(16, 1) != skewed_random_graph(16, 2)
