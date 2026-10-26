"""Writing CSV files and charts. Only formatting here - no PageRank maths."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# reference categorical palette, first three slots (fixed order)
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _style(axes, title: str, x_label: str, y_label: str) -> None:
    axes.set_title(title, loc="left", fontsize=10, color=TEXT)
    axes.set_xlabel(x_label, color=MUTED)
    axes.set_ylabel(y_label, color=MUTED)
    axes.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=MUTED)


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, metadata={"Software": None})      # no version stamp -> byte-stable
    plt.close(figure)


def chart_by_size_and_damping(rows: list[dict], metric: str, y_label: str, title: str, path: Path,
                              log_y: bool = False) -> None:
    """One line per damping factor, graph size on a log x-axis."""
    figure, axes = plt.subplots(figsize=(7.5, 4.3), dpi=130)
    dampings = sorted({float(r["damping"]) for r in rows})
    for colour, damping in zip(SERIES_COLOURS, dampings):
        points = sorted((int(r["pages"]), r[metric]) for r in rows if float(r["damping"]) == damping)
        xs = [p for p, v in points if v != "undefined"]
        ys = [float(v) for p, v in points if v != "undefined"]
        axes.plot(xs, ys, marker="o", markersize=5, linewidth=2, color=colour, label=f"d = {damping:.2f}")
    axes.set_xscale("log")
    if log_y:
        axes.set_yscale("log")
    sizes = sorted({int(r["pages"]) for r in rows})
    axes.set_xticks(sizes, [str(s) for s in sizes])
    axes.minorticks_off()
    _style(axes, title, "pages in graph (log scale)", y_label)
    axes.legend(frameon=False, title="damping factor")
    _save(figure, path)


def chart_budget(summary: list[dict], path: Path) -> None:
    """Mean absolute error vs sample budget (log-log), one line per graph size, with replicate spread."""
    figure, axes = plt.subplots(figsize=(7.5, 4.3), dpi=130)
    sizes = sorted({int(r["pages"]) for r in summary})
    for colour, pages in zip(SERIES_COLOURS, sizes):
        cells = sorted((int(r["samples"]), float(r["mad_mean"]), float(r["mad_sd"]))
                       for r in summary if int(r["pages"]) == pages)
        xs, ys, sds = zip(*cells)
        axes.errorbar(xs, ys, yerr=sds, marker="o", markersize=5, linewidth=2, capsize=3, color=colour,
                      label=f"{pages} pages")
    axes.set_xscale("log")
    axes.set_yscale("log")
    _style(axes, "Sampling budget vs error (d = 0.85, mean ± sd over replicate seeds)",
           "Monte Carlo samples (log scale)", "mean absolute difference vs iterative (log scale)")
    axes.legend(frameon=False)
    _save(figure, path)


def markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)
