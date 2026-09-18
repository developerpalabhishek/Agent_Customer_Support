"""
Decomposition + Plan-Execute (hybrid)
======================================
Decomposition answers "what are the independent pieces of this question?"
Plan-Execute answers "how do I solve one piece that needs several
dependent steps?" These aren't competing patterns — they operate at
different levels and compose naturally:

    [LLM call: split query into independent sub-questions]
    for each sub-question:
        [LLM call: write a step-by-step plan FOR THIS SUB-QUESTION]
        [execute that plan's steps in code, no LLM in between]
    [LLM call: combine all sub-question results into one final answer]

Compare with agent_query_decomposition.py, which assumes each sub-question
is answerable with a single tool call. Here, a sub-question like "add 4
and 6, then square it" is itself a small dependency chain — exactly the
kind of thing a single decomposition-solver call would get wrong, but a
nested plan-execute call handles correctly.
"""
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from calc_tools import TOOLS_BY_NAME, as_text, new_llm

load_dotenv()


class SubQuestions(BaseModel):
    questions: list[str]


class Step(BaseModel):
    tool: Literal["add", "multiply", "exponentiate"]
    x: str = Field(description="A number, or '$stepN' to reuse an earlier step's result")
    y: str = Field(description="A number, or '$stepN' to reuse an earlier step's result")


class Plan(BaseModel):
    steps: list[Step]


decomposer = new_llm().with_structured_output(SubQuestions)
planner = new_llm().with_structured_output(Plan)
synthesizer = new_llm()


def resolve(value: str, results: dict[str, float]) -> float:
    return results[value[1:]] if value.startswith("$") else float(value)


def plan_execute(sub_question: str) -> float:
    """Same mechanism as agent_plan_execute.py, scoped to one sub-question."""
    plan = planner.invoke([HumanMessage(
        f"Break this into add/multiply/exponentiate steps: {sub_question}\n"
        "Reference an earlier step's result as $step1, $step2, etc."
    )])

    results: dict[str, float] = {}
    for i, step in enumerate(plan.steps, start=1):
        x, y = resolve(step.x, results), resolve(step.y, results)
        value = TOOLS_BY_NAME[step.tool].invoke({"x": x, "y": y})
        results[f"step{i}"] = value
        print(f"    step{i}: {step.tool}({x}, {y}) = {value}")

    return results[f"step{len(plan.steps)}"]


def run(query: str) -> str:
    decomposed = decomposer.invoke(
        [HumanMessage(
            f"Split this into independent, self-contained sub-questions: {query}\n\n"
            "IMPORTANT: only split off a sub-question if it can be answered without "
            "knowing the numeric result of any other sub-question. If one part depends "
            "on another part's result (e.g. 'add X and Y, then raise that result to Z'), "
            "keep them together as a SINGLE combined sub-question — do not pre-compute "
            "any intermediate number yourself."
        )]
    )
    print(f"Sub-questions: {decomposed.questions}")

    answers = []
    for q in decomposed.questions:
        print(f"  solving: {q}")
        result = plan_execute(q)
        print(f"  -> {result}")
        answers.append(f"Q: {q}\nA: {result}")

    combined = "\n".join(answers)
    final = synthesizer.invoke([HumanMessage(
        f"Original question: {query}\n\nSub-answers:\n{combined}\n\nGive one combined final answer."
    )])
    return as_text(final.content)


if __name__ == "__main__":
    answer = run(
        "First: add 4 and 6, then raise that result to the power of 2. "
        "Separately: what is 100 multiplied by 3?"
    )
    print("\nFinal answer:", answer)
