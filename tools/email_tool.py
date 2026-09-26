"""
Tool: Email Drafter
===================
"""

from llama_index.core import Settings
from llama_index.core.tools import FunctionTool


def draft_email(to: str, subject: str, context: str, tone: str = "professional") -> str:
    """Draft a professional email using the LLM.

    Args:
        to: Recipient name or email address
        subject: Email subject line
        context: What the email should be about
        tone: 'professional', 'friendly', 'formal', or 'casual'
    """
    prompt = f"""Draft a {tone} email with the following details:

To: {to}
Subject: {subject}
Context: {context}

Rules:
- Write the complete email including greeting and sign-off.
- Keep it concise and well-structured.
- Use a {tone} tone.
- Sign off with "Best regards" followed by a blank line.

Email:
"""
    try:
        result = Settings.llm.complete(prompt).text.strip()
        return f"Subject: {subject}\nTo: {to}\n\n{result}"
    except Exception as exc:
        return f"Email drafting failed: {exc}"


email_tool = FunctionTool.from_defaults(
    fn=draft_email,
    name="draft_email",
    description=(
        "Draft a professional email. Provide: to (recipient), subject, "
        "context (what to say), and optionally tone "
        "('professional', 'friendly', 'formal', 'casual'). "
        "Use when the user asks to write or compose an email."
    ),
)
