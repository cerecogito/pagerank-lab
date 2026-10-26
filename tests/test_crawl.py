"""Tests for the local HTML crawler."""

import socket

import pytest

from pagerank_lab.crawl import crawl, resolve_href
from pagerank_lab.core import iterate_pagerank

DEMO = "data/html_demo"
CORPUS0 = {"1.html": {"2.html"}, "2.html": {"1.html", "3.html"}, "3.html": {"2.html", "4.html"},
           "4.html": {"2.html"}}


def test_demo_corpus_matches_corpus0_structure_without_network(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("the crawler tried to open a network connection")
    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    corpus, report = crawl(DEMO)
    assert corpus == CORPUS0
    assert report.pages == 4 and report.links_kept == 6
    for reason in ("external", "fragment_only", "self_link", "duplicate", "missing_target", "outside_folder"):
        assert report.ignored[reason] >= 1, reason


def test_crawled_corpus_ranks_like_the_course_example():
    corpus, _ = crawl(DEMO)
    ranks = iterate_pagerank(corpus, 0.85).ranks
    assert ranks["1.html"] == pytest.approx(ranks["3.html"], abs=1e-12)
    assert max(ranks, key=ranks.get) == "2.html"


@pytest.mark.parametrize("href, expected", [
    ("2.html", "2.html"), ("./2.html", "2.html"), ("2.html#part", "2.html"), ("2.html?x=1", "2.html"),
    ("https://example.com/2.html", None), ("//cdn.example.com/2.html", None), ("mailto:a@b.c", None),
    ("#top", None), ("../2.html", None), ("sub/2.html", None), ("/2.html", None), ("", None),
    ("%32.html", "2.html"),
])
def test_resolve_href(href, expected):
    assert resolve_href(href)[0] == expected


def test_dangling_page_and_duplicates(tmp_path):
    (tmp_path / "a.html").write_text('<a href="b.html">b</a><a href="b.html">b</a><a href="a.html">self</a>')
    (tmp_path / "b.html").write_text("<p>no links here</p>")
    corpus, report = crawl(tmp_path)
    assert corpus == {"a.html": {"b.html"}, "b.html": set()}
    assert report.ignored["duplicate"] == 1 and report.ignored["self_link"] == 1


def test_missing_targets_and_non_html_files_are_ignored(tmp_path):
    (tmp_path / "a.html").write_text('<a href="ghost.html">x</a><a href="notes.txt">y</a>')
    (tmp_path / "notes.txt").write_text("not a page")
    corpus, report = crawl(tmp_path)
    assert corpus == {"a.html": set()}
    assert report.ignored["missing_target"] == 2


def test_bad_directories(tmp_path):
    with pytest.raises(NotADirectoryError):
        crawl(tmp_path / "nope")
    with pytest.raises(ValueError):
        crawl(tmp_path)            # empty folder
