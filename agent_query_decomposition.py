"""
Query-Decomposition Agent
=========================
Splits one compound question into several *independent* sub-questions,
answers each one separately, then combines the answers.

    [LLM call: split into sub-questions] -> for each sub-question:
        [LLM call w/ tools: answer it, on its own, no shared context]
    -> [LLM call: combine all sub-answers into one final answer]

The key difference from Plan-and-Execute: decomposition assumes the
sub-questions are INDEPENDENT of each other (no step references another
step's result, they could be answered in any order or in parallel).
Plan-and-Execute instead assumes a DEPENDENCY chain (step 2 needs step
1's number). Use decomposition for "and also" questions; use
plan-execute for "then use that to" questions.
"""
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from calc_tools import TOOLS, TOOLS_BY_NAME, as_text, new_llm

load_dotenv()


class SubQuestions(BaseModel):
    questions: list[str]


decomposer = new_llm().with_structured_output(SubQuestions)
solver = new_llm().bind_tools(TOOLS)
synthesizer = new_llm()


def answer_sub_question(question: str) -> str:
    """One-shot tool call per sub-question — each is solved with zero
    knowledge of the others, which is what makes them independent."""
    ai_msg = solver.invoke([HumanMessage(question)])
    if not ai_msg.tool_calls:
        return as_text(ai_msg.content)
    call = ai_msg.tool_calls[0]
    tool_msg = TOOLS_BY_NAME[call["name"]].invoke(call)
    return str(tool_msg.content)


def run(query: str) -> str:
    decomposed = decomposer.invoke(
        [HumanMessage(f"Split this into independent, self-contained sub-questions: {query}")]
    )
    print(f"Sub-questions: {decomposed.questions}")

    answers = []
    for q in decomposed.questions:
        a = answer_sub_question(q)
        print(f"  Q: {q}\n  A: {a}")
        answers.append(f"Q: {q}\nA: {a}")

    combined = "\n".join(answers)
    final = synthesizer.invoke(
        [HumanMessage(f"Original question: {query}\n\nSub-answers:\n{combined}\n\nGive one combined final answer.")]
    )
    return as_text(final.content)


if __name__ == "__main__":
    answer = run("What is 12 * 8, and separately, what is 100 raised to the power of 2?")
    print("\nFinal answer:", answer)
