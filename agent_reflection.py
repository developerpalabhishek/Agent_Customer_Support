"""
Reflection Agent
================
Draft first, verify second. The LLM drafts an answer from reasoning
alone — no tools — which is fast but error-prone (LLMs are notoriously
unreliable at exact multi-digit arithmetic). A reflection pass then uses
the REAL tools to compute the ground truth and either approves the draft
or issues a corrected final answer.

    [LLM call, no tools: draft an answer from memory]
    [LLM call w/ tools: compute the actual ground-truth value]
    [LLM call: compare draft vs ground truth -> approve or correct]

This is the "self-refine" pattern: generate, critique, revise. The critic
step is what's missing from every other pattern in this set — it's the
only one that can catch and fix the agent's own mistake instead of just
reporting whatever the first pass produced.
"""
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from calc_tools import TOOLS, TOOLS_BY_NAME, as_text, new_llm

load_dotenv()

drafter = new_llm(temperature=0.7)          # no tools: reasons from memory, can be wrong
verifier = new_llm().bind_tools(TOOLS)      # has tools: produces ground truth
critic = new_llm()


def compute_ground_truth(query: str) -> str:
    ai_msg = verifier.invoke([HumanMessage(query)])
    if not ai_msg.tool_calls:
        return as_text(ai_msg.content)
    call = ai_msg.tool_calls[0]
    tool_msg = TOOLS_BY_NAME[call["name"]].invoke(call)
    return f"{call['name']}{call['args']} = {tool_msg.content}"


def run(query: str) -> str:
    draft = as_text(drafter.invoke([HumanMessage(query)]).content)
    print(f"Draft (no tools, from memory):\n{draft}\n")

    ground_truth = compute_ground_truth(query)
    print(f"Ground truth (tool-verified): {ground_truth}\n")

    verdict = critic.invoke([HumanMessage(
        f"Question: {query}\n"
        f"Draft answer: {draft}\n"
        f"Tool-verified ground truth: {ground_truth}\n\n"
        "If the draft's numbers match the ground truth, reply with exactly: APPROVED\n"
        "Otherwise, reply with the corrected final answer using the ground-truth numbers."
    )])
    verdict_text = as_text(verdict.content)

    if verdict_text.strip() == "APPROVED":
        print("Critic: APPROVED — draft was already correct.")
        return draft

    print("Critic: draft was WRONG — issuing correction.")
    return verdict_text


if __name__ == "__main__":
    answer = run("What is 393 * 12.25?")
    print("\nFinal answer:", answer)
