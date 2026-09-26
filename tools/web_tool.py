"""
Tool: Web Search
================

Searches the internet via DuckDuckGo (no API key required).
"""

from llama_index.core.tools import FunctionTool


def web_search(query: str, max_results: int = 5) -> str:
    """Search the internet and return top results."""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(r)

        if not results:
            return f"No results found for: {query}"

        output = []
        for i, r in enumerate(results, 1):
            output.append(
                f"[{i}] {r.get('title', '')}\n"
                f"    URL: {r.get('href', '')}\n"
                f"    {r.get('body', '')}"
            )
        return "\n\n".join(output)

    except ImportError:
        return "Error: duckduckgo-search not installed. Run: pip install duckduckgo-search"
    except Exception as exc:
        return f"Web search failed: {exc}"


web_search_tool = FunctionTool.from_defaults(
    fn=web_search,
    name="web_search",
    description=(
        "Search the internet for real-time information, news, tutorials, "
        "technical documentation, or anything NOT in the company documents. "
        "Use this when the user needs external or up-to-date information."
    ),
)
