"""
Tool: Calculator
================

Evaluates mathematical expressions safely using a restricted eval.
"""

import math
from llama_index.core.tools import FunctionTool

_SAFE_MATH = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "pow": pow,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e,
    "ceil": math.ceil, "floor": math.floor,
    "factorial": math.factorial,
}


def calculator(expression: str) -> str:
    """Evaluate a math expression.

    Supports: +, -, *, /, **, %, //, parentheses, sqrt, log,
    sin, cos, tan, factorial, abs, round, min, max, pi, e.

    Examples: "25 * 40", "sqrt(144) + 10", "factorial(10)"
    """
    try:
        result = eval(expression, {"__builtins__": {}}, _SAFE_MATH)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


calculator_tool = FunctionTool.from_defaults(
    fn=calculator,
    name="calculator",
    description=(
        "Evaluate mathematical expressions. Supports arithmetic, "
        "sqrt, log, trig, factorial, and constants pi and e. "
        "Pass the expression as a string e.g. '25 * 40' or 'sqrt(144)'."
    ),
)
