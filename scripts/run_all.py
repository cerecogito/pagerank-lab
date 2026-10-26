"""Run everything: the 21-scenario benchmark, the extra experiments, and the reports.

Usage:
    uv run python scripts/run_all.py            # full experiment (100,000-sample benchmark) -> reports/
    uv run python scripts/run_all.py --quick    # reduced samples for development -> reports/quick/

Deterministic outputs (identical on every re-run): benchmark_results.csv, budget_results.csv,
budget_summary.csv, structure_results.csv, numeric_vs_rank.csv, inputs/*.json, figures/*.png,
results_summary.md, cv_evidence.json.
Runtime metadata (varies between runs): *_timings.csv, timing_report.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from pagerank_lab.benchmark import (
    BUDGET_DAMPING, BUDGET_GRAPH_SIZES, BUDGET_REPLICATES, BUDGETS, DAMPING_FACTORS, FULL_SAMPLES, GRAPH_SIZES,
    STRUCTURE_PAGES, STRUCTURE_REPLICATES, STRUCTURE_SAMPLES, TOLERANCE, aggregate_benchmark,
    numeric_vs_rank_illustration, run_budget_experiment, run_structure_experiment, summarise_budget,
)
from pagerank_lab.reporting import chart_budget, markdown_table, read_csv, write_csv
from run_benchmark import main as run_main_benchmark

COMMAND = "uv run python scripts/run_all.py"
# the figures proposed for the CV, treated as hypotheses to check (never as targets)
CV_HYPOTHESES = {"iterations_median": 11, "iterations_max": 14, "mean_abs_difference_median": 0.000244,
                 "spearman_median": 0.989, "spearman_min": 0.935}


def as_float(value: str) -> float | None:
    return None if value in ("undefined", "", "None") else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true")
    quick = parser.parse_args().quick

    directory = run_main_benchmark(quick)

    budgets = (1_000, 10_000) if quick else BUDGETS
    replicates = 2 if quick else BUDGET_REPLICATES
    print(f"budget experiment: graphs {BUDGET_GRAPH_SIZES}, budgets {budgets}, {replicates} seeds each")
    budget_rows, budget_timings = run_budget_experiment(budgets, replicates)
    write_csv(directory / "budget_results.csv", budget_rows)
    write_csv(directory / "budget_timings.csv", budget_timings)
    write_csv(directory / "budget_summary.csv", summarise_budget(budget_rows))

    structure_samples = 10_000 if quick else STRUCTURE_SAMPLES
    structure_replicates = 1 if quick else STRUCTURE_REPLICATES
    print(f"structure experiment: {STRUCTURE_PAGES} pages, {structure_samples:,} samples x {structure_replicates}")
    structure_rows, structure_timings = run_structure_experiment(structure_samples, structure_replicates)
    write_csv(directory / "structure_results.csv", structure_rows)
    write_csv(directory / "structure_timings.csv", structure_timings)
    write_csv(directory / "numeric_vs_rank.csv", numeric_vs_rank_illustration(structure_samples))

    # ---- everything below is derived from the SAVED files
    benchmark = read_csv(directory / "benchmark_results.csv")
    budget_summary = read_csv(directory / "budget_summary.csv")
    chart_budget(budget_summary, directory / "figures" / "sampling_budget_error.png")
    structure = read_csv(directory / "structure_results.csv")
    illustration = read_csv(directory / "numeric_vs_rank.csv")
    aggregate = aggregate_benchmark(benchmark)

    write_summary(directory, quick, benchmark, aggregate, budget_summary, structure, illustration)
    write_cv_evidence(directory, quick, benchmark, aggregate, budget_summary)
    write_timing_report(directory)
    print(f"done. iterations median {aggregate['iterations_median']}, max {aggregate['iterations_max']}; "
          f"median MAD {aggregate['mean_abs_difference_median']:.6f}; Spearman median "
          f"{aggregate['spearman']['median']:.4f}, min {aggregate['spearman']['min']:.4f}")


def hypothesis_rows(aggregate: dict) -> list[list]:
    actual = {"iterations_median": aggregate["iterations_median"], "iterations_max": aggregate["iterations_max"],
              "mean_abs_difference_median": aggregate["mean_abs_difference_median"],
              "spearman_median": aggregate["spearman"]["median"], "spearman_min": aggregate["spearman"]["min"]}
    # compare at the precision the hypothesis was stated in
    precision = {"iterations_median": 0, "iterations_max": 0, "mean_abs_difference_median": 6,
                 "spearman_median": 3, "spearman_min": 3}
    rows = []
    for key, proposed in CV_HYPOTHESES.items():
        value = actual[key]
        matches = value is not None and round(value, precision[key]) == round(proposed, precision[key])
        shown = "n/a" if value is None else f"{value:.{max(precision[key], 0)}f}" if precision[key] else f"{value:g}"
        rows.append([key, proposed, shown, "yes" if matches else "**no**"])
    return rows


def write_summary(directory: Path, quick: bool, benchmark, aggregate, budget_summary, structure,
                  illustration) -> None:
    samples = int(benchmark[0]["samples"])
    rho = aggregate["spearman"]
    lowest_mad = min(benchmark, key=lambda r: float(r["mean_abs_difference"]))
    defined = [r for r in benchmark if r["spearman"] != "undefined"]
    lowest_rho = min(defined, key=lambda r: float(r["spearman"]))
    slowest = max(structure, key=lambda r: int(r["iterations"]))
    worst_structure_rho = min((r for r in structure if r["spearman_median"] not in ("", "None")),
                              key=lambda r: float(r["spearman_median"]))
    lines = [
        "# PageRank Lab: results summary (deterministic)",
        "",
        f"Generated by `{COMMAND}{' --quick' if quick else ''}` from the saved CSV files in this folder. "
        "Re-running with the same seeds reproduces this file exactly. Timings are kept separately in "
        "`timing_report.md`.",
        "",
        "> All graphs are synthetic and generated locally; no web pages were crawled from the internet. "
        "This is a study of probabilistic algorithms, numerical convergence and stochastic estimation.",
        "" if not quick else "> **QUICK MODE: reduced sample counts. Not the final experiment.**",
        "",
        "## 1. Method",
        "",
        "* **Random surfer (Markov chain):** with probability d follow a uniformly chosen out-link, "
        "otherwise jump to a uniformly chosen page. A dangling page jumps uniformly to any page.",
        f"* **Monte Carlo:** the first page is uniform, then the chain runs for {samples:,} states in total. "
        "Estimate = visit count / samples, with an explicit seed per scenario.",
        f"* **Iterative:** start at 1/N, simultaneous (Jacobi) updates, stop when max |change| < {TOLERANCE:g}. "
        "Each run is also checked against a direct linear solve of the stationary equations.",
        "* **Metrics:**",
        "  * mean absolute difference (MAD) between the two rank vectors;",
        "  * Spearman's rho with average ranks for ties. It is *undefined* if either vector is constant, "
        "and undefined values are excluded from medians and minima, never replaced by 0 or 1.",
        "* **Graphs:**",
        "  * pages get Zipf(1.0) popularity weights, randomly permuted;",
        "  * each page is dangling with probability 0.10, with at least one dangling page guaranteed;",
        "  * other pages get 1 + Poisson(3) distinct out-links, drawn in proportion to popularity, never "
        "to themselves.",
        "* **Seeds:**",
        "  * graph seed = 20261026 + N;",
        "  * sampling seed = 10,000,000 + 1000·N + 100·d;",
        "  * the experiment seeds are listed in `src/pagerank_lab/benchmark.py`;",
        "  * the exact graphs are saved in `inputs/`.",
        "",
        "## 2. Headline statistics (21 scenarios: sizes "
        f"{', '.join(str(n) for n in GRAPH_SIZES)} × damping {', '.join(f'{d:.2f}' for d in DAMPING_FACTORS)})",
        "",
        markdown_table(["statistic", "value"], [
            ["scenarios", aggregate["scenarios"]],
            ["Monte Carlo samples per scenario", f"{samples:,}"],
            ["iterations to tolerance 1e-8: median / min / max",
             f"{aggregate['iterations_median']:g} / {aggregate['iterations_min']} / {aggregate['iterations_max']}"],
            ["mean absolute difference: median / max",
             f"{aggregate['mean_abs_difference_median']:.6f} / {aggregate['mean_abs_difference_max']:.6f}"],
            ["Spearman rho: median / min (defined only)",
             f"{rho['median']:.4f} / {rho['min']:.4f}" if rho["median"] is not None else "n/a"],
            ["Spearman defined / undefined", f"{rho['defined']} / {rho['undefined']}"],
            ["worst max |iterative − direct solve| (convergence check)",
             f"{aggregate['iterative_vs_direct_max_abs_worst']:.2e}"],
        ]),
        "",
        "### Proposed CV figures checked against the actual outputs",
        "",
        markdown_table(["statistic", "proposed", "actual", "matches"], hypothesis_rows(aggregate)),
        "",
        "Every mismatch is reported as measured. The graph design was fixed before the benchmark was run "
        "and was not tuned towards the proposed figures. The iteration count depends on the stopping rule "
        "(max |change| < 1e-8), on d (the error shrinks by at most a factor d per step), and on the graph's "
        "second eigenvalue.",
        "",
        "## 3. All 21 scenarios",
        "",
        markdown_table(["scenario", "pages", "links", "dangling", "d", "sampling seed", "iterations",
                        "|iter − direct|", "MAD", "max abs diff", "Spearman", "exact ties"],
                       [[r["scenario_id"], r["pages"], r["links"], r["dangling_pages"], r["damping"],
                         r["sampling_seed"], r["iterations"], f"{float(r['iterative_vs_direct_max_abs']):.1e}",
                         f"{float(r['mean_abs_difference']):.6f}", f"{float(r['max_abs_difference']):.6f}",
                         r["spearman"] if r["spearman"] == "undefined" else f"{float(r['spearman']):.4f}",
                         f"{100 * float(r['reference_tied_fraction']):.0f}%"]
                        for r in benchmark]),
        "",
        "![sampling error](figures/sampling_error_by_size.png)",
        "",
        "![rank correlation](figures/rank_correlation_by_size.png)",
        "",
        "## 4. Sampling budget vs accuracy",
        "",
        f"Graphs with {', '.join(str(n) for n in BUDGET_GRAPH_SIZES)} pages, d = {BUDGET_DAMPING}. "
        f"Each budget is run with independent seeds, and each estimate is compared with the iterative "
        "reference (itself checked against the direct solve).",
        "",
        markdown_table(["pages", "samples", "seeds", "MAD mean", "MAD sd", "Spearman median", "Spearman min",
                        "log-log slope"],
                       [[r["pages"], f"{int(r['samples']):,}", r["replicates"], f"{float(r['mad_mean']):.6f}",
                         f"{float(r['mad_sd']):.6f}",
                         "undefined" if r["spearman_median"] in ("", "None") else f"{float(r['spearman_median']):.4f}",
                         "undefined" if r["spearman_min"] in ("", "None") else f"{float(r['spearman_min']):.4f}",
                         f"{float(r['slope_log_mad_vs_log_samples']):.3f}"] for r in budget_summary]),
        "",
        budget_slope_sentence(budget_summary),
        "",
        "![budget](figures/sampling_budget_error.png)",
        "",
        f"## 5. Graph structure and damping ({STRUCTURE_PAGES} pages each)",
        "",
        markdown_table(["structure", "links", "dangling", "d", "iterations", "|λ2|", "|iter − direct|",
                        "MAD mean", "Spearman median", "max/min rank", "pages in exact ties"],
                       [[r["structure"], r["links"], r["dangling_pages"], r["damping"], r["iterations"],
                         f"{float(r['second_eigenvalue_modulus']):.3f}",
                         f"{float(r['iterative_vs_direct_max_abs']):.1e}", f"{float(r['mad_mean']):.6f}",
                         "undefined" if r["spearman_median"] in ("", "None") else f"{float(r['spearman_median']):.4f}",
                         f"{float(r['rank_max_over_min']):.1f}", f"{100 * float(r['tied_fraction']):.0f}%"]
                        for r in structure]),
        "",
        f"* Slowest convergence: **{slowest['structure']}** at d = {slowest['damping']} "
        f"({slowest['iterations']} iterations, |λ2| = {float(slowest['second_eigenvalue_modulus']):.3f}). "
        "The error of the iteration shrinks roughly like |λ2|^k, and |λ2| ≤ d. A graph that splits into "
        "weakly linked parts has |λ2| close to d, so higher damping and weak links both slow convergence.",
        f"* Lowest rank agreement: **{worst_structure_rho['structure']}** at d = {worst_structure_rho['damping']} "
        f"(median Spearman {float(worst_structure_rho['spearman_median']):.4f}, MAD "
        f"{float(worst_structure_rho['mad_mean']):.6f}). In that graph "
        f"{100 * float(worst_structure_rho['tied_fraction']):.0f}% of pages share an exactly equal PageRank "
        "(pages without in-links all get (1−d)/N plus the dangling share). Monte Carlo cannot reproduce a "
        "tie, so it orders tied pages by noise: the numerical error is small but the rank correlation is low.",
        *tolerance_vs_error_sentence(structure),
        "",
        "## 6. Numerical accuracy ≠ ranking accuracy",
        "",
        markdown_table(["comparison", "MAD", "Spearman", "max/min rank of reference",
                        "pages in exact ties (reference)"],
                       [[r["case"], f"{float(r['mean_abs_difference']):.6f}",
                         "undefined" if r["spearman"] in ("", "None") else f"{float(r['spearman']):.4f}",
                         f"{float(r['reference_rank_max_over_min']):.2f}",
                         f"{100 * float(r['reference_tied_fraction']):.0f}%"] for r in illustration]),
        "",
        f"In the benchmark itself, the lowest MAD is **{lowest_mad['scenario_id']}** "
        f"(MAD {float(lowest_mad['mean_abs_difference']):.6f}, Spearman "
        f"{lowest_mad['spearman'] if lowest_mad['spearman'] == 'undefined' else format(float(lowest_mad['spearman']), '.4f')}). "
        f"The lowest Spearman is **{lowest_rho['scenario_id']}** ({float(lowest_rho['spearman']):.4f}). Larger "
        "graphs spread the same probability over more pages, so the absolute errors shrink. But the share "
        "of pages in exact ties in the iterative reference rises with size (from "
        f"{100 * min(float(r['reference_tied_fraction']) for r in benchmark):.0f}% to "
        f"{100 * max(float(r['reference_tied_fraction']) for r in benchmark):.0f}% across the benchmark), and "
        "sampling noise breaks those ties at random. A low numerical error therefore does not imply a "
        "high rank correlation, and vice versa.",
        "",
        "## 7. Limitations",
        "",
        "* Synthetic graphs only. Real web graphs have different degree distributions, spam structure "
        "and scale (billions of pages).",
        "* The iterative ranks are a reference only after the convergence check (the direct-solve "
        "difference above). They are not exact: the stopping rule bounds the last change, not the error.",
        "* Monte Carlo uses one long chain per run, so its samples are correlated. The error depends on "
        "mixing as well as on the number of samples.",
        "* The direct solve and eigenvalues use dense linear algebra, which is fine for ≤ 1,000 pages but "
        "not for web scale.",
    ]
    (directory / "results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def budget_slope_sentence(budget_summary) -> str:
    """Interpretation of the fitted error-vs-budget slopes, read from the saved summary."""
    slopes = {}
    for row in budget_summary:
        slopes[int(row["pages"])] = float(row["slope_log_mad_vs_log_samples"])
    listed = ", ".join(f"{pages} pages: {slope:.3f}" for pages, slope in sorted(slopes.items()))
    return (f"Measured log-log slopes of mean error against budget: {listed}. For independent draws the "
            "error falls like 1/sqrt(samples), a slope of -0.5. The surfer's states are correlated, but "
            "the chain mixes quickly, so the measured rate is essentially the same. 10x more samples buys "
            "about 3.2x less error (10^0.5). Sampling time grows roughly in proportion to the budget once "
            "fixed setup costs are small; at 1,000 samples the setup dominates (see `timing_report.md`). "
            "Rank correlation improves much more slowly on the large graphs.")


def tolerance_vs_error_sentence(structure) -> list[str]:
    """Report (from the saved rows) any iterative result further than the tolerance from the direct solve."""
    over = [r for r in structure if float(r["iterative_vs_direct_max_abs"]) > TOLERANCE]
    if not over:
        return ["* Every iterative result is within the 1e-8 tolerance of the direct solve."]
    worst = max(over, key=lambda r: float(r["iterative_vs_direct_max_abs"]))
    return [f"* In {len(over)} of {len(structure)} structure runs, the converged iterative ranks differ from "
            f"the direct solve by more than the 1e-8 tolerance (worst {float(worst['iterative_vs_direct_max_abs']):.1e}: "
            f"{worst['structure']}, d = {worst['damping']}). The stopping rule bounds the last *change*, "
            "not the error, which can be about |λ2|/(1−|λ2|) times larger. This is why the iterative reference "
            "is checked against the direct solve before being trusted."]


def write_cv_evidence(directory: Path, quick: bool, benchmark, aggregate, budget_summary) -> None:
    rho = aggregate["spearman"]
    evidence = {
        "project": "pagerank-lab",
        "mode": "quick" if quick else "full",
        "reproduce_command": COMMAND + (" --quick" if quick else ""),
        "test_command": "uv run pytest",
        "deterministic": True,
        "seed_strategy": {"graph_seed": "20261026 + N", "sampling_seed": "10_000_000 + 1000*N + round(100*d)",
                          "budget_seed": "30_000_000 + 1000*N + 100*log10(samples) + replicate"},
        "benchmark": {"scenarios": len(benchmark), "graph_sizes": list(GRAPH_SIZES),
                      "damping_factors": list(DAMPING_FACTORS), "samples": int(benchmark[0]["samples"]),
                      "tolerance": TOLERANCE, "evidence_file": "reports/benchmark_results.csv"},
        "metrics": [
            {"metric": "iterations_median", "value": aggregate["iterations_median"],
             "definition": "median over 21 scenarios of simultaneous updates until max |change| < 1e-8"},
            {"metric": "iterations_max", "value": aggregate["iterations_max"], "definition": "maximum over 21 scenarios"},
            {"metric": "mean_abs_difference_median", "value": aggregate["mean_abs_difference_median"],
             "definition": "median over scenarios of mean_p |MC(p) - iterative(p)|"},
            {"metric": "spearman_median", "value": rho["median"], "defined_scenarios": rho["defined"],
             "undefined_scenarios": rho["undefined"],
             "definition": "Spearman rho (average ranks for ties) between MC and iterative; undefined excluded"},
            {"metric": "spearman_min", "value": rho["min"], "definition": "minimum over defined scenarios"},
            {"metric": "iterative_vs_direct_worst", "value": aggregate["iterative_vs_direct_max_abs_worst"],
             "definition": "max over scenarios of max_p |iterative(p) - direct linear solve(p)|"},
        ],
        "proposed_cv_figures_check": [dict(zip(["statistic", "proposed", "actual", "matches"], row))
                                      for row in hypothesis_rows(aggregate)],
        "budget_experiment": {"graph_sizes": list(BUDGET_GRAPH_SIZES), "damping": BUDGET_DAMPING,
                              "evidence_file": "reports/budget_summary.csv",
                              "slopes": {r["pages"]: float(r["slope_log_mad_vs_log_samples"])
                                         for r in budget_summary}},
        "timings": {"note": "runtime metadata, varies by run", "files": ["reports/benchmark_timings.csv",
                                                                           "reports/timing_report.md"]},
    }
    (directory / "cv_evidence.json").write_text(json.dumps(evidence, indent=1) + "\n", encoding="utf-8")


def write_timing_report(directory: Path) -> None:
    timings = read_csv(directory / "benchmark_timings.csv")
    results = {r["scenario_id"]: r for r in read_csv(directory / "benchmark_results.csv")}
    budget = read_csv(directory / "budget_timings.csv")
    lines = ["# Timing report (runtime metadata, varies between runs)", "",
             f"Python {sys.version.split()[0]} on {platform.platform()}. Single run per scenario.", "",
             markdown_table(["scenario", "pages", "links", "samples", "sampling s", "iteration s",
                             "µs per sample"],
                            [[t["scenario_id"], results[t["scenario_id"]]["pages"], results[t["scenario_id"]]["links"],
                              f"{int(results[t['scenario_id']]['samples']):,}", f"{float(t['sampling_seconds']):.4f}",
                              f"{float(t['iteration_seconds']):.5f}",
                              f"{1e6 * float(t['sampling_seconds']) / int(results[t['scenario_id']]['samples']):.2f}"]
                             for t in timings]), "",
             "## Sampling time by budget (mean over seeds)", "",
             markdown_table(["pages", "samples", "mean sampling s"],
                            [[pages, f"{samples:,}", f"{np.mean([float(b['sampling_seconds']) for b in budget if int(b['pages']) == pages and int(b['samples']) == samples]):.4f}"]
                             for pages in sorted({int(b["pages"]) for b in budget})
                             for samples in sorted({int(b["samples"]) for b in budget})])]
    (directory / "timing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
