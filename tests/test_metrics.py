"""Tests for MAD, Spearman (ties, undefined cases) and aggregation."""

import pytest
from scipy.stats import spearmanr

from pagerank_lab.metrics import mean_absolute_difference, spearman, summarise_defined


def test_mean_absolute_difference_by_hand():
    assert mean_absolute_difference({"a": 0.5, "b": 0.5}, {"a": 0.4, "b": 0.6}) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        mean_absolute_difference({"a": 1.0}, {"b": 1.0})


def test_spearman_with_ties_matches_scipy():
    a = {"p1": 0.1, "p2": 0.1, "p3": 0.3, "p4": 0.5, "p5": 0.5}
    b = {"p1": 0.2, "p2": 0.1, "p3": 0.3, "p4": 0.6, "p5": 0.4}
    expected = spearmanr([a[k] for k in sorted(a)], [b[k] for k in sorted(b)]).statistic
    assert spearman(a, b) == pytest.approx(expected, abs=1e-12)


def test_spearman_perfect_and_reversed():
    a = {"x": 0.1, "y": 0.2, "z": 0.7}
    assert spearman(a, {"x": 0.2, "y": 0.3, "z": 0.5}) == pytest.approx(1.0)
    assert spearman(a, {"x": 0.7, "y": 0.2, "z": 0.1}) == pytest.approx(-1.0)


def test_spearman_is_undefined_for_constant_vectors_and_single_pages():
    uniform = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    other = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}
    assert spearman(uniform, other) is None
    assert spearman(other, uniform) is None
    assert spearman({"a": 1.0}, {"a": 1.0}) is None


def test_small_numerical_error_can_still_reorder_close_pages():
    """MAD and Spearman measure different things."""
    reference = {"a": 0.2501, "b": 0.2500, "c": 0.2499, "d": 0.2500}
    noisy = {"a": 0.2499, "b": 0.2500, "c": 0.2501, "d": 0.2500}       # tiny changes, order reversed
    assert mean_absolute_difference(reference, noisy) < 1e-4
    assert spearman(reference, noisy) < 0
    distorted = {"a": 0.7, "b": 0.2, "c": 0.05, "d": 0.05}
    base = {"a": 0.4, "b": 0.3, "c": 0.15, "d": 0.15}
    assert spearman(base, distorted) == pytest.approx(1.0) and mean_absolute_difference(base, distorted) > 0.1


def test_summarise_defined_excludes_undefined_values():
    summary = summarise_defined([0.9, None, 0.95, None, 0.99])
    assert summary == {"defined": 3, "undefined": 2, "median": 0.95, "min": 0.9, "max": 0.99}
    assert summarise_defined([None, None])["median"] is None
