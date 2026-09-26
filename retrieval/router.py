"""
Dynamic LLM Router + Metadata Filter
=====================================

A single LLM call that decides BOTH:
  • route      → RAG | COMPLEX_RAG | DIRECT_LLM
  • department → HR | IT | Finance | NONE

Returns a dict with: route, department, filters, reasoning, confidence
"""

import json
import re
import logging

from llama_index.core import Settings
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from config.settings import ALLOWED_DEPARTMENTS

log = logging.getLogger(__name__)

ALLOWED_ROUTES = {"RAG", "COMPLEX_RAG", "DIRECT_LLM"}

_DEPT_DISPLAY = {"HR": "HR", "IT": "IT", "FINANCE": "Finance"}


def dynamic_route_and_filter(question: str) -> dict:
    """Single LLM call that decides routing AND department filtering."""
    prompt = f"""You are the intelligent routing and filtering system of an enterprise knowledge assistant.

Analyse the user's question and decide TWO things:

═══ STEP 1: ROUTE ═══

RAG
  → Question needs information from company documents (policies, rules, procedures).
  → Examples: "What is the leave policy?", "What are the IT security rules?"

COMPLEX_RAG
  → Question requires comparing, contrasting, or combining information from
    MULTIPLE sections/documents. It needs sub-questions.
  → Examples: "Compare leave policy and IT policy",
    "What are differences between HR and Finance procedures?"

DIRECT_LLM
  → General knowledge NOT in company documents.
  → Examples: "What is Python?", "Explain recursion", "What is the capital of France?"

═══ STEP 2: DEPARTMENT ═══

If route is RAG or COMPLEX_RAG, pick the most relevant department:
  HR       → leave, attendance, remote work, resignation
  IT       → passwords, security, accounts, IT infrastructure
  Finance  → budget, expenses, payroll, financial reports
  NONE     → spans multiple departments or none specifically

If route is DIRECT_LLM, department must be NONE.

═══ OUTPUT FORMAT ═══

Respond with EXACTLY this JSON (no extra text):

{{
  "reasoning": "<one sentence explaining your decision>",
  "route": "<RAG or COMPLEX_RAG or DIRECT_LLM>",
  "department": "<HR or IT or Finance or NONE>",
  "confidence": <0.0 to 1.0>
}}

Question: {question}
"""

    result = {
        "route": "RAG",
        "department": "NONE",
        "filters": None,
        "reasoning": "fallback",
        "confidence": 0.0,
    }

    try:
        raw = Settings.llm.complete(prompt).text.strip()

        # Extract JSON even if LLM wraps it in markdown
        match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        parsed = json.loads(match.group() if match else raw)

        # Validate route
        route = str(parsed.get("route", "RAG")).strip().upper()
        if route not in ALLOWED_ROUTES:
            log.warning("Unknown route '%s', defaulting to RAG.", route)
            route = "RAG"
        result["route"] = route

        # Validate department
        dept = str(parsed.get("department", "NONE")).strip().upper()
        if dept in ALLOWED_DEPARTMENTS:
            dept_value = _DEPT_DISPLAY.get(dept, dept)
            result["department"] = dept_value
            result["filters"] = MetadataFilters(
                filters=[MetadataFilter(key="department", value=dept_value)]
            )
        else:
            result["department"] = "NONE"
            result["filters"] = None

        # Force NONE for DIRECT_LLM
        if route == "DIRECT_LLM":
            result["department"] = "NONE"
            result["filters"] = None

        result["reasoning"] = str(parsed.get("reasoning", ""))
        result["confidence"] = float(parsed.get("confidence", 0.5))

    except json.JSONDecodeError as exc:
        log.warning("Could not parse routing JSON: %s — raw: %.200s", exc, raw)
        raw_upper = raw.upper()
        for r in ("COMPLEX_RAG", "DIRECT_LLM", "RAG"):
            if r in raw_upper:
                result["route"] = r
                break
    except Exception as exc:
        log.error("LLM routing call failed: %s — defaulting to RAG", exc)

    return result
