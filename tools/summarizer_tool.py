"""
Tool: Text Summarizer
=====================
"""

from llama_index.core import Settings
from llama_index.core.tools import FunctionTool

_STYLES = {
    "concise": "Summarize in 3-5 clear sentences.",
    "bullets": "Summarize as 5-8 bullet points, each starting with •.",
    "detailed": "Provide a comprehensive summary in 2-3 paragraphs covering all key points.",
}


def summarize_text(text: str, style: str = "concise") -> str:
    """Summarize a piece of text using the LLM.

    Args:
        text: The text to summarize
        style: 'concise', 'bullets', or 'detailed'
    """
    instruction = _STYLES.get(style.lower(), _STYLES["concise"])
    prompt = f"""Summarize the following text.

{instruction}

Text:
{text[:5000]}

Summary:
"""
    try:
        return Settings.llm.complete(prompt).text.strip()
    except Exception as exc:
        return f"Summarization failed: {exc}"


summarizer_tool = FunctionTool.from_defaults(
    fn=summarize_text,
    name="summarize_text",
    description=(
        "Summarize a piece of text. Provide the text and optionally "
        "style: 'concise' (3-5 sentences), 'bullets' (bullet points), "
        "or 'detailed' (comprehensive). Use when the user asks to summarize something."
    ),
)
