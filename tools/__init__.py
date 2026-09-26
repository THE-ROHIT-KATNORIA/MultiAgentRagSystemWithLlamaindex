# Tools package — each tool as its own module
from tools.rag_tool import rag_tool, build_rag_tool
from tools.web_tool import web_search_tool
from tools.calculator_tool import calculator_tool
from tools.datetime_tool import datetime_tool
from tools.company_tool import company_info_tool
from tools.email_tool import email_tool
from tools.summarizer_tool import summarizer_tool
from tools.notes_tool import notes_tool

ALL_TOOLS = [
    rag_tool,
    web_search_tool,
    calculator_tool,
    datetime_tool,
    company_info_tool,
    email_tool,
    summarizer_tool,
    notes_tool,
]
