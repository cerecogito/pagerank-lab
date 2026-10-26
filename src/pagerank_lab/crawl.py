"""Read a local folder of HTML files into a link corpus (no network access, ever).

Rules for a link <a href="..."> found in page P:
* links with a scheme or host (http://, https://, mailto:, //host/...) are external -> ignored;
* the fragment (#section) and query (?x=1) are dropped; a bare "#top" is not a link to a page;
* the remaining path is normalised and must be a plain file name in the same folder;
  anything pointing into or out of a sub-folder ("a/b.html", "../x.html") is ignored, so a
  crafted href can never make us read outside the corpus;
* the target must be one of the corpus pages, otherwise it is a missing target -> ignored;
* links from a page to itself are ignored, and duplicates count once (links form a set).
Every ignored link is counted in the CrawlReport so nothing is dropped silently.
"""

from __future__ import annotations

import posixpath
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class _LinkParser(HTMLParser):
    """Collects the href of every <a> tag, in document order."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value is not None:
                self.hrefs.append(value)


@dataclass
class CrawlReport:
    """Why links were ignored, counted per reason (for transparency and tests)."""
    ignored: Counter = field(default_factory=Counter)
    pages: int = 0
    links_kept: int = 0


def resolve_href(href: str) -> tuple[str | None, str]:
    """Turn an href into a same-folder file name, or (None, reason) if it is not one."""
    href = href.strip()
    if not href:
        return None, "empty"
    parts = urlsplit(href)
    if parts.scheme or parts.netloc:
        return None, "external"
    path = unquote(parts.path)
    if path == "":
        return None, "fragment_only"            # e.g. "#top" or "?page=2"
    normalised = posixpath.normpath(path)
    if normalised.startswith("/") or "/" in normalised or normalised in (".", ".."):
        return None, "outside_folder"
    return normalised, "ok"


def crawl(directory: str | Path) -> tuple[dict[str, set[str]], CrawlReport]:
    """Return ({page: set of linked pages}, report) for every *.html file in `directory`."""
    folder = Path(directory)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a directory")
    files = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".html")
    if not files:
        raise ValueError(f"no .html files found in {folder}")

    page_names = {path.name for path in files}
    corpus: dict[str, set[str]] = {}
    report = CrawlReport(pages=len(files))
    for path in files:
        parser = _LinkParser()
        parser.feed(path.read_text(encoding="utf-8", errors="replace"))
        links: set[str] = set()
        for href in parser.hrefs:
            target, reason = resolve_href(href)
            if target is None:
                report.ignored[reason] += 1
            elif target == path.name:
                report.ignored["self_link"] += 1
            elif target not in page_names:
                report.ignored["missing_target"] += 1
            elif target in links:
                report.ignored["duplicate"] += 1
            else:
                links.add(target)
        corpus[path.name] = links
        report.links_kept += len(links)
    return corpus, report
