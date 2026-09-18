"""Shared tools/model for the agentic-pattern demos (agent_*.py).

Every pattern (reflex, ReAct, plan-execute, ...) solves the same kind of
arithmetic question with the same three tools. What changes between files
is only the *control flow* around the LLM, not the domain.
"""
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

MODEL = "gemini-3.1-flash-lite"


@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y


@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the 'y'."""
    return x**y


TOOLS = [add, multiply, exponentiate]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def new_llm(temperature: float = 0):
    return ChatGoogleGenerativeAI(model=MODEL, temperature=temperature)


def as_text(content) -> str:
    """Gemini returns content as either a plain string or a list of parts."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))
