"""
Plan-and-Execute Agent
======================
Separates *planning* from *doing*. One LLM call produces a complete,
ordered plan up front (every step, with its tool and arguments); the
plan is then executed mechanically, step by step, with NO further LLM
calls in between (contrast with ReAct, which re-invokes the LLM after
every single action).

    [LLM call: "write the whole plan"] -> [execute step 1, 2, 3, ... in code]
        -> [LLM call: synthesize final answer from the results]

Steps can reference earlier results via a placeholder ("$step1"), so the
plan can express a dependency chain without needing the LLM back in the
loop to resolve it.

Cheaper than ReAct for tasks whose shape is predictable up front (one LLM
call for planning + one for the final summary, instead of one per step).
Weaker when the right next step truly depends on *reasoning about* a
result (e.g. "if the result is negative, do X instead") — that requires
looking at the result before deciding, which is exactly what ReAct does
and plan-execute does not.
"""
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from calc_tools import TOOLS_BY_NAME, as_text, new_llm

load_dotenv()


class Step(BaseModel):
    tool: Literal["add", "multiply", "exponentiate"]
    x: str = Field(description="A number, or '$stepN' to reuse an earlier step's result")
    y: str = Field(description="A number, or '$stepN' to reuse an earlier step's result")


class Plan(BaseModel):
    steps: list[Step]


planner = new_llm().with_structured_output(Plan)
synthesizer = new_llm()


def resolve(value: str, results: dict[str, float]) -> float:
    return results[value[1:]] if value.startswith("$") else float(value)


def run(query: str) -> str:
    plan = planner.invoke(
        [HumanMessage(
            f"Break this into add/multiply/exponentiate steps: {query}\n"
            "Reference an earlier step's result as $step1, $step2, etc."
        )]
    )

    print(f"Plan: {[s.model_dump() for s in plan.steps]}")

    results: dict[str, float] = {}
    for i, step in enumerate(plan.steps, start=1):
        x, y = resolve(step.x, results), resolve(step.y, results)
        value = TOOLS_BY_NAME[step.tool].invoke({"x": x, "y": y})
        results[f"step{i}"] = value
        print(f"  step{i}: {step.tool}({x}, {y}) = {value}")

    summary = "\n".join(f"step{i} = {v}" for i, v in results.items())
    final = synthesizer.invoke(
        [HumanMessage(f"Question: {query}\nComputed results:\n{summary}\nAnswer the question in one sentence.")]
    )
    return as_text(final.content)


if __name__ == "__main__":
    answer = run("Add 4 and 6, then raise that result to the power of 2.")
    print("\nFinal answer:", answer)
