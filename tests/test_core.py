"""Tests for the transition model, sampler and iterative algorithm."""

import numpy as np
import pytest

from pagerank_lab.core import (
    ConvergenceError, compile_corpus, iterate_pagerank, normalise_corpus, random_surfer_step, sample_pagerank,
    solve_pagerank_exact, transition_model,
)
from pagerank_lab.metrics import mean_absolute_difference

CS50_EXAMPLE = {"1.html": {"2.html", "3.html"}, "2.html": {"3.html"}, "3.html": {"2.html"}}
# same link structure as the CS50 "corpus0" example
CORPUS0 = {"1.html": {"2.html"}, "2.html": {"1.html", "3.html"}, "3.html": {"2.html", "4.html"},
           "4.html": {"2.html"}}


def test_official_transition_model_example():
    result = transition_model(CS50_EXAMPLE, "1.html", 0.85)
    assert result == pytest.approx({"1.html": 0.05, "2.html": 0.475, "3.html": 0.475})


@pytest.mark.parametrize("page", ["1.html", "2.html", "3.html", "4.html"])
@pytest.mark.parametrize("damping", [0.3, 0.85, 0.99])
def test_transition_probabilities_sum_to_one(page, damping):
    assert sum(transition_model(CORPUS0, page, damping).values()) == pytest.approx(1.0)


def test_dangling_page_transitions_uniformly():
    corpus = {"a": {"b"}, "b": set(), "c": {"a"}}
    assert transition_model(corpus, "b", 0.85) == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})


def test_duplicates_and_self_links_are_ignored():
    with_extras = {"a": ["b", "b", "a"], "b": ["a"]}
    plain = {"a": {"b"}, "b": {"a"}}
    assert transition_model(with_extras, "a", 0.85) == transition_model(plain, "a", 0.85)
    assert normalise_corpus(with_extras)["a"] == frozenset({"b"})
    # a page whose only link is to itself is dangling
    assert transition_model({"a": {"a"}, "b": {"a"}}, "a", 0.85) == pytest.approx({"a": 0.5, "b": 0.5})


@pytest.mark.parametrize("bad", [{"a": {"zzz"}}, {}, {"a": "b"}])
def test_unknown_links_and_bad_corpora_are_rejected(bad):
    with pytest.raises(ValueError):
        transition_model(bad, "a", 0.85)


@pytest.mark.parametrize("damping", [0.0, 1.0, -0.1, 1.5, float("nan"), "0.85", True])
def test_invalid_damping_is_rejected(damping):
    with pytest.raises(ValueError):
        transition_model(CS50_EXAMPLE, "1.html", damping)


def test_unknown_page_and_bad_sample_count():
    with pytest.raises(ValueError):
        transition_model(CS50_EXAMPLE, "9.html", 0.85)
    with pytest.raises(ValueError):
        sample_pagerank(CS50_EXAMPLE, 0.85, 0, seed=1)


def test_symmetric_cycle_has_equal_ranks():
    cycle = {f"p{i}": {f"p{(i + 1) % 5}"} for i in range(5)}
    result = iterate_pagerank(cycle, 0.85)
    assert all(v == pytest.approx(0.2, abs=1e-12) for v in result.ranks.values())


def test_analytic_two_page_graph_with_a_dangling_page():
    """a -> b, b dangling. Solving the two equations gives PR(a) = 1/(2+d), PR(b) = (1+d)/(2+d)."""
    for d in (0.5, 0.85, 0.95):
        ranks = iterate_pagerank({"a": {"b"}, "b": set()}, d, tolerance=1e-14).ranks
        assert ranks["a"] == pytest.approx(1 / (2 + d), abs=1e-12)
        assert ranks["b"] == pytest.approx((1 + d) / (2 + d), abs=1e-12)


def test_corpus0_closed_form_and_published_course_values():
    """Pages 1 and 3 have identical in-links, so they get equal rank.

    The course printed 0.2202/0.4289/0.2202/0.1307, iterated only to a 0.001 tolerance, so we
    allow 0.001.
    """
    ranks = iterate_pagerank(CORPUS0, 0.85).ranks
    exact = compile_corpus(CORPUS0).to_dict(solve_pagerank_exact(compile_corpus(CORPUS0), 0.85))
    assert ranks == pytest.approx(exact, abs=1e-8)
    assert ranks["1.html"] == pytest.approx(ranks["3.html"], abs=1e-12)
    published = {"1.html": 0.2202, "2.html": 0.4289, "3.html": 0.2202, "4.html": 0.1307}
    assert all(abs(ranks[p] - published[p]) <= 0.001 for p in published)


def test_iteration_converges_and_stays_normalised():
    corpus = {"a": {"b", "c"}, "b": {"c"}, "c": {"a"}, "d": set(), "e": {"d", "a"}}
    result = iterate_pagerank(corpus, 0.85, tolerance=1e-10)
    assert sum(result.ranks.values()) == pytest.approx(1.0, abs=1e-12)   # no renormalisation needed
    assert result.final_max_change < 1e-10 and result.iterations > 1


def test_iteration_limit_raises():
    corpus = {"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": set()}
    with pytest.raises(ConvergenceError):
        iterate_pagerank(corpus, 0.85, tolerance=1e-15, max_iterations=3)


def test_simultaneous_update_is_order_independent():
    """Renaming pages (a different internal order) must not change any rank."""
    corpus = {"a": {"b", "c"}, "b": {"c"}, "c": {"a"}, "d": {"a"}}
    renamed = {"z" + k: {"z" + t for t in v} for k, v in corpus.items()}
    renamed = {"zz" if k == "za" else k: {"zz" if t == "za" else t for t in v} for k, v in renamed.items()}
    original = iterate_pagerank(corpus, 0.85).ranks
    other = iterate_pagerank(renamed, 0.85).ranks
    mapping = {"a": "zz", "b": "zb", "c": "zc", "d": "zd"}
    for page, value in original.items():
        assert other[mapping[page]] == pytest.approx(value, abs=1e-15)


def test_sampler_is_reproducible_and_normalised():
    first = sample_pagerank(CORPUS0, 0.85, 5_000, seed=11)
    second = sample_pagerank(CORPUS0, 0.85, 5_000, seed=11)
    other = sample_pagerank(CORPUS0, 0.85, 5_000, seed=12)
    assert first == second and first != other
    assert sum(first.values()) == pytest.approx(1.0)


def test_sampler_step_distribution_matches_transition_model():
    """The sampler's own step function, fed random draws, reproduces transition_model."""
    corpus = {"a": {"b", "c"}, "b": set(), "c": {"a"}}
    compiled = compile_corpus(corpus)
    trials = 40_000
    for start_index, start in enumerate(compiled.names):
        expected = transition_model(corpus, start, 0.85)
        rng = np.random.default_rng(100 + start_index)
        follow, uniform = rng.random(trials) < 0.85, rng.random(trials)
        links = compiled.links[start_index].tolist()
        counts = np.bincount([random_surfer_step(links, compiled.size, f, x) for f, x in zip(follow, uniform)],
                             minlength=compiled.size)
        for index, page in enumerate(compiled.names):
            p = expected[page]
            standard_error = (p * (1 - p) / trials) ** 0.5
            assert abs(counts[index] / trials - p) < 5 * standard_error + 1e-9


def test_sampling_error_shrinks_as_samples_grow():
    """Average over several seeds so the test is not a single lucky draw."""
    corpus = {"a": {"b", "c"}, "b": {"c"}, "c": {"a"}, "d": set(), "e": {"d", "a"}}
    reference = iterate_pagerank(corpus, 0.85).ranks
    errors = []
    for samples in (500, 5_000, 50_000):
        errors.append(np.mean([mean_absolute_difference(sample_pagerank(corpus, 0.85, samples, seed), reference)
                               for seed in range(5)]))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.01
