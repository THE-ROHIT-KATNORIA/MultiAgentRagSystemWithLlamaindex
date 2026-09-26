"""
Tool: Company Information
=========================
"""

from llama_index.core.tools import FunctionTool

_INFO = {
    "name": "Katnoria Technologies",
    "working_days": "Monday to Friday",
    "working_hours": "9:30 AM to 6:00 PM",
    "lunch_break": "1:00 PM to 2:00 PM",
    "timezone": "IST (UTC+5:30)",
    "hr_email": "hr@katnoria.com",
    "it_support": "it-support@katnoria.com",
    "office": "Bengaluru, India",
    "holidays_2026": [
        "Jan 26 — Republic Day",
        "Mar 14 — Holi",
        "Apr 14 — Ambedkar Jayanti",
        "May 1  — May Day",
        "Aug 15 — Independence Day",
        "Oct 2  — Gandhi Jayanti",
        "Oct 21 — Dussehra",
        "Nov 11 — Diwali",
        "Dec 25 — Christmas",
    ],
}


def get_company_info(topic: str = "all") -> str:
    """Return company information by topic.

    Args:
        topic: 'working_hours', 'holidays', 'contacts', or 'all'
    """
    t = topic.lower().strip()

    if t in ("hours", "working_hours", "schedule", "timing"):
        return (
            f"Company: {_INFO['name']}\n"
            f"Working Days: {_INFO['working_days']}\n"
            f"Working Hours: {_INFO['working_hours']}\n"
            f"Lunch Break: {_INFO['lunch_break']}\n"
            f"Timezone: {_INFO['timezone']}"
        )
    if t in ("holidays", "holiday"):
        return "Company Holidays 2026:\n" + "\n".join(f"  • {h}" for h in _INFO["holidays_2026"])
    if t in ("contacts", "contact", "email"):
        return (
            f"HR Email:   {_INFO['hr_email']}\n"
            f"IT Support: {_INFO['it_support']}\n"
            f"Office:     {_INFO['office']}"
        )

    holidays = "\n".join(f"  • {h}" for h in _INFO["holidays_2026"])
    return (
        f"Company: {_INFO['name']}\n"
        f"Working Days: {_INFO['working_days']}\n"
        f"Working Hours: {_INFO['working_hours']}\n"
        f"Lunch Break: {_INFO['lunch_break']}\n"
        f"Timezone: {_INFO['timezone']}\n\n"
        f"Contacts:\n"
        f"  HR: {_INFO['hr_email']}\n"
        f"  IT: {_INFO['it_support']}\n"
        f"  Office: {_INFO['office']}\n\n"
        f"Holidays 2026:\n{holidays}"
    )


company_info_tool = FunctionTool.from_defaults(
    fn=get_company_info,
    name="company_info",
    description=(
        "Get company working hours, holidays, contact emails, and office info. "
        "Pass topic='working_hours', 'holidays', 'contacts', or 'all'."
    ),
)
