"""Reproducible experiments comparing Monte Carlo and iterative PageRank.

Seed strategy (every seed is a plain function of the scenario, so it is easy to audit):
    graph seed     = BASE_SEED + n_pages
    sampling seed  = 10_000_000 + 1000 * n_pages + round(100 * damping)
    budget study   = 30_000_000 + 1000 * n_pages + 100 * log10(samples) + replicate
    structure study: graph seed = BASE_SEED + 7 * (structure number + 1),
                     sampling seed = 50_000_000 + 1000 * structure number + 100 * round(10 * damping) + replicate

Before the iterative ranks are used as the reference, every run also solves
the linear system directly and records max |iterative - direct|. This checks
that the iteration really converged rather than assuming it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from pagerank_lab.core import (
    CompiledCorpus, compile_corpus, iterate_pagerank, sample_pagerank, second_eigenvalue_modulus,
    solve_pagerank_exact,
)
from pagerank_lab.graphs import BASE_SEED, count_dangling, count_links, skewed_random_graph, two_community_graph
from pagerank_lab.metrics import (
    max_absolute_difference, mean_absolute_difference, spearman, summarise_defined,
)

GRAPH_SIZES = (8, 16, 32, 64, 128, 256, 1000)
DAMPING_FACTORS = (0.70, 0.85, 0.95)
FULL_SAMPLES = 100_000
QUICK_SAMPLES = 2_000
TOLERANCE = 1e-8


# ---------------------------------------------------------------- main 21-scenario benchmark

@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    n_pages: int
    damping: float
    graph_seed: int
    sampling_seed: int


def graph_seed_for(n_pages: int) -> int:
    return BASE_SEED + n_pages


def sampling_seed_for(n_pages: int, damping: float) -> int:
    return 10_000_000 + 1000 * n_pages + round(100 * damping)


def benchmark_scenarios() -> list[Scenario]:
    return [Scenario(f"n{n}_d{round(100 * d)}", n, d, graph_seed_for(n), sampling_seed_for(n, d))
            for n in GRAPH_SIZES for d in DAMPING_FACTORS]


def benchmark_graph(n_pages: int) -> dict[str, set[str]]:
    return skewed_random_graph(n_pages, graph_seed_for(n_pages))


def reference_ranks(compiled: CompiledCorpus, damping: float, tolerance: float = TOLERANCE):
    """Iterative ranks plus an independent direct-solve check of their accuracy."""
    started = time.perf_counter()
    iteration = iterate_pagerank(compiled, damping, tolerance)
    iteration_seconds = time.perf_counter() - started
    direct = compiled.to_dict(solve_pagerank_exact(compiled, damping))
    check = max_absolute_difference(iteration.ranks, direct)
    return iteration, check, iteration_seconds


def run_scenario(scenario: Scenario, samples: int, tolerance: float = TOLERANCE) -> tuple[dict, dict]:
    """Return (deterministic metrics row, timing row) for one scenario."""
    corpus = benchmark_graph(scenario.n_pages)
    compiled = compile_corpus(corpus)
    iteration, check, iteration_seconds = reference_ranks(compiled, scenario.damping, tolerance)
    started = time.perf_counter()
    sampled = sample_pagerank(compiled, scenario.damping, samples, scenario.sampling_seed)
    sampling_seconds = time.perf_counter() - started
    rho = spearman(sampled, iteration.ranks)
    metrics = {
        "scenario_id": scenario.scenario_id,
        "pages": compiled.size,
        "links": count_links(corpus),
        "dangling_pages": count_dangling(corpus),
        "damping": scenario.damping,
        "graph_seed": scenario.graph_seed,
        "sampling_seed": scenario.sampling_seed,
        "samples": samples,
        "tolerance": tolerance,
        "iterations": iteration.iterations,
        "final_max_change": iteration.final_max_change,
        "iterative_vs_direct_max_abs": check,
        "mean_abs_difference": mean_absolute_difference(sampled, iteration.ranks),
        "max_abs_difference": max_absolute_difference(sampled, iteration.ranks),
        "spearman": "undefined" if rho is None else rho,
        "reference_tied_fraction": tied_fraction(iteration.ranks),
        "sampled_rank_sum": float(sum(sampled.values())),
        "iterative_rank_sum": float(sum(iteration.ranks.values())),
    }
    timing = {"scenario_id": scenario.scenario_id, "sampling_seconds": sampling_seconds,
              "iteration_seconds": iteration_seconds}
    return metrics, timing


def run_benchmark(samples: int = FULL_SAMPLES, tolerance: float = TOLERANCE) -> tuple[list[dict], list[dict]]:
    rows, timings = [], []
    for scenario in benchmark_scenarios():
        metrics, timing = run_scenario(scenario, samples, tolerance)
        rows.append(metrics)
        timings.append(timing)
    return rows, timings


def aggregate_benchmark(rows: list[dict]) -> dict:
    """Headline statistics, computed only from the saved scenario rows."""
    rhos = [None if r["spearman"] == "undefined" else float(r["spearman"]) for r in rows]
    iterations = [int(r["iterations"]) for r in rows]
    mads = [float(r["mean_abs_difference"]) for r in rows]
    return {
        "scenarios": len(rows),
        "iterations_median": float(np.median(iterations)),
        "iterations_min": min(iterations),
        "iterations_max": max(iterations),
        "mean_abs_difference_median": float(np.median(mads)),
        "mean_abs_difference_max": float(np.max(mads)),
        "spearman": summarise_defined(rhos),
        "iterative_vs_direct_max_abs_worst": max(float(r["iterative_vs_direct_max_abs"]) for r in rows),
    }


# ---------------------------------------------------------------- sampling-budget experiment

BUDGET_GRAPH_SIZES = (32, 256, 1000)
BUDGET_DAMPING = 0.85
BUDGETS = (1_000, 10_000, 100_000)
BUDGET_REPLICATES = 5


def budget_seed(n_pages: int, samples: int, replicate: int) -> int:
    return 30_000_000 + 1000 * n_pages + 100 * round(math.log10(samples)) + replicate


def run_budget_experiment(budgets=BUDGETS, replicates: int = BUDGET_REPLICATES) -> tuple[list[dict], list[dict]]:
    rows, timings = [], []
    for n_pages in BUDGET_GRAPH_SIZES:
        compiled = compile_corpus(benchmark_graph(n_pages))
        iteration, check, _ = reference_ranks(compiled, BUDGET_DAMPING)
        for samples in budgets:
            for replicate in range(replicates):
                seed = budget_seed(n_pages, samples, replicate)
                started = time.perf_counter()
                sampled = sample_pagerank(compiled, BUDGET_DAMPING, samples, seed)
                seconds = time.perf_counter() - started
                rho = spearman(sampled, iteration.ranks)
                rows.append({"pages": n_pages, "damping": BUDGET_DAMPING, "samples": samples,
                             "replicate": replicate, "seed": seed,
                             "mean_abs_difference": mean_absolute_difference(sampled, iteration.ranks),
                             "spearman": "undefined" if rho is None else rho,
                             "reference_check_max_abs": check})
                timings.append({"pages": n_pages, "samples": samples, "replicate": replicate,
                                "sampling_seconds": seconds})
    return rows, timings


def summarise_budget(rows: list[dict]) -> list[dict]:
    """Mean and spread over replicates, plus the fitted log-log slope of error vs budget."""
    summary = []
    for n_pages in sorted({r["pages"] for r in rows}):
        group = [r for r in rows if r["pages"] == n_pages]
        budgets = sorted({r["samples"] for r in group})
        mean_errors = []
        for samples in budgets:
            cell = [r for r in group if r["samples"] == samples]
            errors = [r["mean_abs_difference"] for r in cell]
            rhos = summarise_defined([None if r["spearman"] == "undefined" else float(r["spearman"]) for r in cell])
            mean_errors.append(float(np.mean(errors)))
            summary.append({"pages": n_pages, "samples": samples, "replicates": len(cell),
                            "mad_mean": float(np.mean(errors)), "mad_sd": float(np.std(errors, ddof=1)),
                            "spearman_median": rhos["median"], "spearman_min": rhos["min"],
                            "spearman_undefined": rhos["undefined"], "slope_log_mad_vs_log_samples": None})
        slope = float(np.polyfit(np.log10(budgets), np.log10(mean_errors), 1)[0])
        for row in summary:
            if row["pages"] == n_pages:
                row["slope_log_mad_vs_log_samples"] = slope
    return summary


# ---------------------------------------------------------------- graph-structure sensitivity

STRUCTURE_PAGES = 256
STRUCTURE_DAMPINGS = (0.70, 0.85, 0.95)
STRUCTURE_SAMPLES = 100_000
STRUCTURE_REPLICATES = 3


def structure_graphs() -> dict[str, dict[str, set[str]]]:
    """Deliberately different structures, all with 256 pages."""
    n = STRUCTURE_PAGES
    seed = lambda k: BASE_SEED + 7 * (k + 1)  # noqa: E731
    return {
        "benchmark_mix (Zipf 1.0, 10% dangling)": skewed_random_graph(n, seed(0)),
        "many_dangling (50% dangling)": skewed_random_graph(n, seed(1), dangling_fraction=0.5),
        "hub_dominated (Zipf 2.0)": skewed_random_graph(n, seed(2), zipf_exponent=2.0),
        "two_communities (4 bridge links)": two_community_graph(n, seed(3)),
        "uniform_targets (no popularity skew, no dangling)": skewed_random_graph(
            n, seed(4), zipf_exponent=0.0, dangling_fraction=0.0),
    }


def tied_fraction(ranks: dict[str, float], decimals: int = 12) -> float:
    """Fraction of pages whose rank exactly equals another page's rank (to `decimals` places).

    Exact ties are common: every page with no incoming links has the same PageRank,
    (1-d)/N plus the shared dangling contribution. Monte Carlo cannot reproduce a
    tie, so it orders tied pages by noise, which lowers Spearman's rho.
    """
    _, counts = np.unique(np.round(np.array(list(ranks.values())), decimals), return_counts=True)
    return float(counts[counts > 1].sum() / len(ranks))


def rank_gap_statistics(ranks: dict[str, float]) -> dict:
    """How crowded the ranking is: exact ties and gaps between neighbouring sorted values."""
    values = np.sort(np.array(list(ranks.values())))
    gaps = np.diff(values)
    return {"rank_max_over_min": float(values[-1] / values[0]),
            "median_neighbour_gap": float(np.median(gaps)),
            "tied_fraction": tied_fraction(ranks),
            "rank_coefficient_of_variation": float(values.std() / values.mean())}


def run_structure_experiment(samples: int = STRUCTURE_SAMPLES,
                             replicates: int = STRUCTURE_REPLICATES) -> tuple[list[dict], list[dict]]:
    rows, timings = [], []
    for number, (name, corpus) in enumerate(structure_graphs().items()):
        compiled = compile_corpus(corpus)
        for damping in STRUCTURE_DAMPINGS:
            iteration, check, iteration_seconds = reference_ranks(compiled, damping)
            mads, rhos, seconds = [], [], []
            for replicate in range(replicates):
                seed = 50_000_000 + 1000 * number + 100 * round(10 * damping) + replicate
                started = time.perf_counter()
                sampled = sample_pagerank(compiled, damping, samples, seed)
                seconds.append(time.perf_counter() - started)
                mads.append(mean_absolute_difference(sampled, iteration.ranks))
                rhos.append(spearman(sampled, iteration.ranks))
            rho_summary = summarise_defined(rhos)
            rows.append({
                "structure": name, "pages": compiled.size, "links": count_links(corpus),
                "dangling_pages": count_dangling(corpus), "damping": damping,
                "iterations": iteration.iterations,
                "second_eigenvalue_modulus": second_eigenvalue_modulus(compiled, damping),
                "iterative_vs_direct_max_abs": check,
                "samples": samples, "replicates": replicates,
                "mad_mean": float(np.mean(mads)),
                "spearman_median": rho_summary["median"], "spearman_undefined": rho_summary["undefined"],
                **rank_gap_statistics(iteration.ranks),
            })
            timings.append({"structure": name, "damping": damping, "iteration_seconds": iteration_seconds,
                            "sampling_seconds_mean": float(np.mean(seconds))})
    return rows, timings


# ---------------------------------------------------------------- numerical vs ranking accuracy

def numeric_vs_rank_illustration(samples: int = STRUCTURE_SAMPLES) -> list[dict]:
    """Two real comparisons showing that MAD and Spearman measure different things.

    A. hub-dominated graph: Monte Carlo estimate vs iterative ranks. Many pages share
       exactly the same rank (no in-links), so sampling noise reorders them: small MAD, low rho.
    B. hub-dominated graph: the iterative vector vs a monotone distortion of itself
       (ranks squared, renormalised). The order is unchanged (rho = 1) while the
       numbers move a lot (large MAD).
    """
    graphs = structure_graphs()
    hub = compile_corpus(graphs["hub_dominated (Zipf 2.0)"])
    hub_ranks = iterate_pagerank(hub, 0.85).ranks
    hub_sample = sample_pagerank(hub, 0.85, samples, 70_000_001)
    squared = {page: value ** 2 for page, value in hub_ranks.items()}
    total = sum(squared.values())
    distorted = {page: value / total for page, value in squared.items()}
    return [
        {"case": f"A: hub-dominated graph, Monte Carlo ({samples:,} samples) vs iterative",
         "mean_abs_difference": mean_absolute_difference(hub_sample, hub_ranks),
         "spearman": spearman(hub_sample, hub_ranks),
         **{f"reference_{k}": v for k, v in rank_gap_statistics(hub_ranks).items()}},
        {"case": "B: hub-dominated graph, iterative vs (iterative^2 renormalised)",
         "mean_abs_difference": mean_absolute_difference(distorted, hub_ranks),
         "spearman": spearman(distorted, hub_ranks),
         **{f"reference_{k}": v for k, v in rank_gap_statistics(hub_ranks).items()}},
    ]
