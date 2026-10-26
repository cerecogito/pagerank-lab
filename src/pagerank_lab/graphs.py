"""Deterministic synthetic directed graphs (no real web data).

Benchmark generator (`skewed_random_graph`), one seed per graph:
1. pages are named p0000, p0001, ...;
2. each page gets a "popularity" weight from a Zipf law, w = 1 / rank^alpha, with the
   ranks randomly permuted over pages. This gives skewed, varied in-degrees;
3. each page is dangling (no out-links) with probability `dangling_fraction`. At least
   one dangling page is guaranteed, so the dangling rule is always exercised;
4. every other page gets 1 + Poisson(`mean_extra_links`) distinct out-links (capped at
   N - 1). The targets are drawn without replacement with probability proportional to
   popularity, never itself.

`two_community_graph` builds two equal halves with links inside each half plus
a few bridge links between them, to study weakly connected structure.
"""

from __future__ import annotations

import numpy as np

BASE_SEED = 20261026


def page_names(n_pages: int) -> list[str]:
    width = max(4, len(str(n_pages - 1)))
    return [f"p{i:0{width}d}" for i in range(n_pages)]


def skewed_random_graph(n_pages: int, seed: int, dangling_fraction: float = 0.10,
                        zipf_exponent: float = 1.0, mean_extra_links: float = 3.0) -> dict[str, set[str]]:
    if n_pages < 2:
        raise ValueError("need at least 2 pages")
    rng = np.random.default_rng(seed)
    names = page_names(n_pages)
    popularity = 1.0 / np.arange(1, n_pages + 1) ** zipf_exponent
    popularity = popularity[rng.permutation(n_pages)]

    dangling = rng.random(n_pages) < dangling_fraction
    if not dangling.any():
        dangling[rng.integers(n_pages)] = True          # guarantee at least one dangling page

    corpus: dict[str, set[str]] = {}
    for page in range(n_pages):
        if dangling[page]:
            corpus[names[page]] = set()
            continue
        out_degree = min(1 + int(rng.poisson(mean_extra_links)), n_pages - 1)
        weights = popularity.copy()
        weights[page] = 0.0                              # no self-links
        targets = rng.choice(n_pages, size=out_degree, replace=False, p=weights / weights.sum())
        corpus[names[page]] = {names[t] for t in targets}
    return corpus


def two_community_graph(n_pages: int, seed: int, links_per_page: int = 4, bridges: int = 4,
                        dangling_fraction: float = 0.10) -> dict[str, set[str]]:
    """Two equal communities with uniform links inside each and `bridges` links between them."""
    if n_pages < 4 or n_pages % 2:
        raise ValueError("need an even number of pages, at least 4")
    rng = np.random.default_rng(seed)
    names = page_names(n_pages)
    half = n_pages // 2
    dangling = rng.random(n_pages) < dangling_fraction
    corpus: dict[str, set[str]] = {}
    for page in range(n_pages):
        if dangling[page]:
            corpus[names[page]] = set()
            continue
        start = 0 if page < half else half
        candidates = [p for p in range(start, start + half) if p != page]
        chosen = rng.choice(candidates, size=min(links_per_page, len(candidates)), replace=False)
        corpus[names[page]] = {names[t] for t in chosen}
    # a few bridges, from non-dangling pages, alternating direction
    linkers = [p for p in range(n_pages) if not dangling[p]]
    left = [p for p in linkers if p < half]
    right = [p for p in linkers if p >= half]
    for b in range(bridges):
        source_side, target_side = (left, range(half, n_pages)) if b % 2 == 0 else (right, range(half))
        source = int(rng.choice(source_side))
        target = int(rng.choice(list(target_side)))
        corpus[names[source]].add(names[target])
    return corpus


def count_links(corpus: dict[str, set[str]]) -> int:
    return sum(len(links) for links in corpus.values())


def count_dangling(corpus: dict[str, set[str]]) -> int:
    return sum(1 for links in corpus.values() if not links)
