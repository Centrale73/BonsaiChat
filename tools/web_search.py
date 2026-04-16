"""
tools/web_search.py — DuckDuckGo web search tool for BonsaiChat agents.

Inspired by Agent Zero's search_engine tool.  Requires no API key.
Returns the top N results as a formatted string the LLM can reason over.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from agno.tools import Toolkit

try:
    from duckduckgo_search import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    _DDGS_AVAILABLE = False


MAX_RESULTS = 8


def _ddg_search(query: str, max_results: int = MAX_RESULTS) -> str:
    """Synchronous DuckDuckGo text search returning a formatted string."""
    if not _DDGS_AVAILABLE:
        return (
            "duckduckgo_search is not installed. "
            "Run: pip install duckduckgo-search"
        )

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            title = r.get("title", "")
            href  = r.get("href", "")
            body  = r.get("body", "")
            results.append(f"**{title}**\n{href}\n{body}")

    if not results:
        return f"No results found for: {query}"

    header = f"Web search results for: {query}\n{'=' * 60}\n\n"
    return header + "\n\n".join(results)


class WebSearchTools(Toolkit):
    """Agno Toolkit wrapping DuckDuckGo — zero API key required."""

    def __init__(self, max_results: int = MAX_RESULTS):
        super().__init__(name="web_search")
        self.max_results = max_results
        self.register(self.web_search)

    def web_search(self, query: str) -> str:
        """
        Search the web using DuckDuckGo and return the top results.

        Args:
            query: The search query string.

        Returns:
            Formatted search results with title, URL and snippet per result.
        """
        return _ddg_search(query, self.max_results)
