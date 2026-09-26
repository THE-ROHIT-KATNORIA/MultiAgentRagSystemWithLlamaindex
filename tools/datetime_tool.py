"""
Tool: Date & Time
=================
"""

from datetime import datetime, timezone
from llama_index.core.tools import FunctionTool


def get_date_time() -> str:
    """Return current date, time, day of week, and UTC timestamp."""
    now = datetime.now()
    utc = datetime.now(timezone.utc)
    return (
        f"Date:        {now.strftime('%A, %B %d, %Y')}\n"
        f"Time:        {now.strftime('%I:%M:%S %p')}\n"
        f"Day:         {now.strftime('%A')}\n"
        f"Week Number: {now.strftime('%W')}\n"
        f"UTC:         {utc.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"Timestamp:   {int(now.timestamp())}"
    )


datetime_tool = FunctionTool.from_defaults(
    fn=get_date_time,
    name="date_time",
    description=(
        "Get the current date, time, day of week, week number, and UTC time. "
        "Use when the user asks what time or date it is."
    ),
)
