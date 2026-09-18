"""
Deep-Research Agent
===================
An iterative gather-and-reflect loop that decides, ROUND BY ROUND, what
it still needs to find out — rather than deciding everything up front
(Plan-and-Execute) or splitting the question into a fixed, known set of
parts (Query-Decomposition).

    findings = []
    loop:
        [LLM call: "given the query and findings so far, what's the next
                    thing to look into, or are we DONE?"]
        done? -> break
        [LLM call w/ tools: research that one sub-question] -> append to findings
    [LLM call: write a final report from all accumulated findings]

The number and order of research steps is open-ended and chosen by the
agent itself as it goes — the defining trait of "deep research" style
agents versus the fixed-shape patterns above.
"""
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from calc_tools import TOOLS, TOOLS_BY_NAME, as_text, new_llm

load_dotenv()

MAX_ROUNDS = 5


class NextStep(BaseModel):
    done: bool
    next_question: str | None = None


router = new_llm().with_structured_output(NextStep)
researcher = new_llm().bind_tools(TOOLS)
reporter = new_llm()


def research_one(question: str) -> str:
    ai_msg = researcher.invoke([HumanMessage(question)])
    if not ai_msg.tool_calls:
        return as_text(ai_msg.content)
    call = ai_msg.tool_calls[0]
    tool_msg = TOOLS_BY_NAME[call["name"]].invoke(call)
    return f"{question} -> {call['name']}{call['args']} = {tool_msg.content}"


def run(query: str) -> str:
    findings: list[str] = []

    for round_num in range(1, MAX_ROUNDS + 1):
        notes = "\n".join(findings) or "(none yet)"
        decision = router.invoke([HumanMessage(
            f"Overall task: {query}\nFindings so far:\n{notes}\n\n"
            "What is the single next sub-question we need to research to fully "
            "answer the overall task? If every part has already been covered "
            "by the findings, set done=true instead."
        )])

        if decision.done or not decision.next_question:
            print(f"[round {round_num}] agent decided it has enough. Stopping.")
            break

        print(f"[round {round_num}] researching: {decision.next_question}")
        result = research_one(decision.next_question)
        print(f"  finding: {result}")
        findings.append(result)
    else:
        print("Hit max rounds — stopping and reporting with what we have.")

    report = reporter.invoke([HumanMessage(
        f"Task: {query}\n\nResearch findings:\n" + "\n".join(findings) +
        "\n\nWrite a short final report answering the task using these findings."
    )])
    return as_text(report.content)


if __name__ == "__main__":
    answer = run(
        "I need a report covering: 12 * 8, separately 45 + 30, "
        "and separately 2 raised to the power of 10."
    )
    print("\nFinal report:\n", answer)
