"""
Literature Search Tool (Stage 2) -- the "Literature Search" box in the
architecture diagram.

Searches arXiv's public API. The HTTP fetch is injected (`fetch_fn`) so
this can be exercised in tests without any real network call -- production
code just uses the default, which calls the real arXiv API.

Note on scope: this returns titles, authors, and a short excerpt only, not
full abstracts. This tool is meant to point the agent at sources for the
literature-retrieval step in the design (e.g. checking a derived formula
against a published result), not to ingest and quote full paper text.
"""
from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, Optional

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
EXCERPT_MAX_CHARS = 200


def _default_fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read().decode("utf-8")


class LiteratureSearchTool:
    name = "literature_search"

    def __init__(self, fetch_fn: Optional[Callable[[str], str]] = None):
        self._fetch = fetch_fn or _default_fetch

    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expected input shape:
            {"query": "damped harmonic oscillator", "max_results": 3}

        Returns:
            {"query": "...", "results": [{"id", "title", "authors", "excerpt"}, ...]}
        Raises ValueError if the request or response parsing fails.
        """
        query = input.get("query")
        max_results = input.get("max_results", 3)
        if not query:
            raise ValueError("literature_search requires 'query'")

        params = urllib.parse.urlencode(
            {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
        )
        url = f"{ARXIV_API_URL}?{params}"

        try:
            raw_xml = self._fetch(url)
        except Exception as e:
            raise ValueError(f"Literature search request failed: {e}")

        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError as e:
            raise ValueError(f"Could not parse arXiv response: {e}")

        results = []
        for entry in root.findall("atom:entry", ATOM_NS):
            title_el = entry.find("atom:title", ATOM_NS)
            summary_el = entry.find("atom:summary", ATOM_NS)
            id_el = entry.find("atom:id", ATOM_NS)
            authors = [
                name_el.text
                for a in entry.findall("atom:author", ATOM_NS)
                if (name_el := a.find("atom:name", ATOM_NS)) is not None
            ]
            title = (title_el.text or "").strip() if title_el is not None else ""
            summary = (summary_el.text or "").strip() if summary_el is not None else ""
            excerpt = (
                summary[:EXCERPT_MAX_CHARS] + "..."
                if len(summary) > EXCERPT_MAX_CHARS
                else summary
            )
            results.append(
                {
                    "id": id_el.text.strip() if id_el is not None else "",
                    "title": title,
                    "authors": authors,
                    "excerpt": excerpt,
                }
            )

        return {"query": query, "results": results}
