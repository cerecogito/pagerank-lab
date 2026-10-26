"""PageRank: transition model, Monte Carlo random surfer, iterative fixed point.

The random surfer is a Markov chain on the N pages. From page i:
* with probability d it follows one of i's outgoing links, chosen uniformly;
* with probability 1 - d it "teleports" to a page chosen uniformly from all N.
A page with no outgoing links (a dangling page) is treated as linking to every
page, including itself, so from it the next page is uniform over all N.

PageRank is the stationary distribution of this chain, i.e. the fixed point of
    PR(p) = (1 - d)/N + d * [ sum_{i -> p} PR(i)/L(i) + sum_{i dangling} PR(i)/N ]
where L(i) is the number of distinct outgoing links of page i.

The corpus format is {page: set of linked pages}. Links are a set, so duplicates
count once. Self-links are removed and links to pages outside the corpus are
rejected (see normalise_corpus).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

Corpus = Mapping[str, Iterable[str]]


class ConvergenceError(RuntimeError):
    """Iterative PageRank hit its iteration limit before reaching the tolerance."""


# ---------------------------------------------------------------- input checks

def check_damping(damping: float) -> float:
    """The damping factor is a probability; 0 < d < 1 keeps the chain irreducible and convergent."""
    if isinstance(damping, bool) or not isinstance(damping, (int, float)) or not np.isfinite(damping):
        raise ValueError(f"damping factor must be a finite number, got {damping!r}")
    if not 0.0 < damping < 1.0:
        raise ValueError(f"damping factor must satisfy 0 < d < 1, got {damping}")
    return float(damping)


def normalise_corpus(corpus: Corpus) -> dict[str, frozenset[str]]:
    """Copy the corpus with self-links removed and links stored as sets.

    Raises ValueError for an empty corpus or a link to a page that is not in it,
    because silently dropping such links would change the Markov chain.
    """
    if not corpus:
        raise ValueError("corpus is empty")
    pages = set(corpus)
    clean: dict[str, frozenset[str]] = {}
    for page, links in corpus.items():
        if not isinstance(page, str) or not page:
            raise ValueError(f"page names must be non-empty strings, got {page!r}")
        if isinstance(links, str):
            raise ValueError(f"links of {page!r} must be a collection of page names, not a string")
        targets = set(links)
        unknown = targets - pages
        if unknown:
            raise ValueError(f"page {page!r} links to pages not in the corpus: {sorted(unknown)}")
        clean[page] = frozenset(targets - {page})          # a self-link is ignored
    return clean


@dataclass(frozen=True)
class CompiledCorpus:
    """Index-based form of a corpus for fast sampling and iteration."""
    names: tuple[str, ...]              # sorted page names; index i <-> names[i]
    links: tuple[np.ndarray, ...]       # links[i] = indices of pages i links to (sorted)
    edge_sources: np.ndarray            # one entry per directed link
    edge_targets: np.ndarray
    out_degree: np.ndarray

    @property
    def size(self) -> int:
        return len(self.names)

    @property
    def dangling(self) -> np.ndarray:
        return self.out_degree == 0

    def to_dict(self, values: np.ndarray) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.names, values)}


def compile_corpus(corpus: Corpus) -> CompiledCorpus:
    clean = normalise_corpus(corpus)
    names = tuple(sorted(clean))
    index = {name: i for i, name in enumerate(names)}
    links = tuple(np.array(sorted(index[t] for t in clean[name]), dtype=np.int64) for name in names)
    sources = np.concatenate([np.full(len(l), i, dtype=np.int64) for i, l in enumerate(links)])
    targets = np.concatenate(links) if sources.size else np.array([], dtype=np.int64)
    out_degree = np.array([len(l) for l in links], dtype=np.int64)
    return CompiledCorpus(names, links, sources, targets, out_degree)


# ---------------------------------------------------------------- transition model

def transition_model(corpus: Corpus, page: str, damping_factor: float) -> dict[str, float]:
    """P(next page | current page) for the random surfer, over every page in the corpus."""
    d = check_damping(damping_factor)
    clean = normalise_corpus(corpus)
    if page not in clean:
        raise ValueError(f"page {page!r} is not in the corpus")
    n = len(clean)
    links = clean[page]
    if not links:                                   # dangling: behave as if linking to every page
        return {p: 1.0 / n for p in clean}
    distribution = {p: (1.0 - d) / n for p in clean}  # teleport share for every page
    for target in links:
        distribution[target] += d / len(links)      # follow-a-link share
    return distribution


# ---------------------------------------------------------------- Monte Carlo

def sample_pagerank(corpus: Corpus | CompiledCorpus, damping_factor: float, samples: int,
                    seed: int) -> dict[str, float]:
    """Estimate PageRank as the fraction of `samples` random-surfer states spent on each page.

    The first state is uniform over all pages. Each later state is drawn from
    the transition model of the previous state. We draw it as a two-stage
    mixture, which is exactly the same distribution:
        with prob. d and if the page has links -> uniform over its links
        otherwise (teleport, or dangling page)  -> uniform over all pages.
    A dangling page always goes to the second branch, which gives the uniform
    distribution the transition model prescribes. tests/test_core.py checks the
    empirical one-step frequencies against transition_model.
    """
    check_damping(damping_factor)
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError(f"samples must be a positive integer, got {samples!r}")
    compiled = corpus if isinstance(corpus, CompiledCorpus) else compile_corpus(corpus)
    n = compiled.size
    rng = np.random.default_rng(seed)

    # draw all random numbers up front (fast); the chain itself is still sequential
    follow_link = (rng.random(samples) < damping_factor).tolist()
    choice = rng.random(samples).tolist()
    link_lists = [l.tolist() for l in compiled.links]

    counts = [0] * n
    current = int(choice[0] * n)                # first sample: uniform page
    counts[current] += 1
    for step in range(1, samples):
        current = random_surfer_step(link_lists[current], n, follow_link[step], choice[step])
        counts[current] += 1
    return compiled.to_dict(np.array(counts, dtype=float) / samples)


def random_surfer_step(links: list[int], n_pages: int, follow_link: bool, uniform: float) -> int:
    """One move of the surfer, given a coin (follow_link ~ Bernoulli(d)) and a Uniform[0, 1) draw.

    floor(uniform * k) is uniform on {0, ..., k-1}, so this picks a uniform out-link
    or a uniform page. A dangling page (no links) always takes the uniform-page branch.
    """
    if follow_link and links:
        return links[int(uniform * len(links))]
    return int(uniform * n_pages)


# ---------------------------------------------------------------- iteration

@dataclass(frozen=True)
class IterationResult:
    ranks: dict[str, float]
    iterations: int                 # number of simultaneous updates performed
    final_max_change: float         # max |PR_new - PR_old| at the last update
    tolerance: float


def iterate_pagerank(corpus: Corpus | CompiledCorpus, damping_factor: float, tolerance: float = 1e-8,
                     max_iterations: int = 10_000) -> IterationResult:
    """Repeat the PageRank update from PR = 1/N until no page changes by `tolerance` or more.

    Every update uses only the previous iteration's full vector (a simultaneous,
    Jacobi-style update), never values already overwritten in the same pass.
    The update keeps the total at exactly 1 in exact arithmetic, so no
    renormalisation is applied; tests check that the sum stays at 1.
    """
    d = check_damping(damping_factor)
    if not tolerance > 0:
        raise ValueError("tolerance must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    compiled = corpus if isinstance(corpus, CompiledCorpus) else compile_corpus(corpus)
    n = compiled.size
    dangling = compiled.dangling
    share_per_link = np.zeros(n)
    has_links = compiled.out_degree > 0
    share_per_link[has_links] = 1.0 / compiled.out_degree[has_links]

    ranks = np.full(n, 1.0 / n)
    for iteration in range(1, max_iterations + 1):
        flow = np.bincount(compiled.edge_targets,
                           weights=(ranks * share_per_link)[compiled.edge_sources], minlength=n)
        dangling_mass = ranks[dangling].sum()           # spread evenly over all N pages
        new_ranks = (1.0 - d) / n + d * (flow + dangling_mass / n)
        max_change = float(np.max(np.abs(new_ranks - ranks)))
        ranks = new_ranks
        if max_change < tolerance:
            return IterationResult(compiled.to_dict(ranks), iteration, max_change, tolerance)
    raise ConvergenceError(f"no convergence to {tolerance} within {max_iterations} iterations "
                           f"(last max change {max_change:.3e})")


# ---------------------------------------------------------------- independent check of the reference

def google_matrix(compiled: CompiledCorpus, damping_factor: float) -> np.ndarray:
    """Dense column-stochastic matrix G with G[j, i] = P(next = j | current = i)."""
    d = check_damping(damping_factor)
    n = compiled.size
    follow = np.zeros((n, n))
    for i, links in enumerate(compiled.links):
        if links.size:
            follow[links, i] = 1.0 / links.size
        else:
            follow[:, i] = 1.0 / n                      # dangling column: uniform
    return d * follow + (1.0 - d) / n


def solve_pagerank_exact(compiled: CompiledCorpus, damping_factor: float) -> np.ndarray:
    """Stationary distribution by solving the linear system (G - I) x = 0 with sum(x) = 1.

    Used only to CHECK the iterative reference; it shares no code with the
    iteration other than the corpus.
    """
    g = google_matrix(compiled, damping_factor)
    n = compiled.size
    system = g - np.eye(n)
    system[-1, :] = 1.0                                 # replace one equation by sum(x) = 1
    rhs = np.zeros(n)
    rhs[-1] = 1.0
    return np.linalg.solve(system, rhs)


def second_eigenvalue_modulus(compiled: CompiledCorpus, damping_factor: float) -> float:
    """|lambda_2| of the Google matrix: the asymptotic error-shrink factor per iteration (<= d)."""
    moduli = np.sort(np.abs(np.linalg.eigvals(google_matrix(compiled, damping_factor))))[::-1]
    return float(moduli[1]) if moduli.size > 1 else 0.0
