"""
ReAct Agent (Reason + Act)
==========================
Interleaves reasoning and acting in a loop. Unlike Reflex, the LLM sees
the *entire* conversation so far, including every previous tool result,
before deciding its next move — so step 2 can depend on step 1's answer.

    while True:
        [LLM call over full history] -> tool_calls?
            yes -> [execute tools] -> append observations -> loop
            no  -> that's the final answer, stop

This is the classic "Thought -> Action -> Observation -> Thought -> ..."
loop from the ReAct paper. It's more capable than Reflex (can chain steps)
but also more expensive (one LLM call per step) and has no separation
between "planning" and "doing" — see agent_plan_execute.py for that.
"""
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from calc_tools import TOOLS, TOOLS_BY_NAME, as_text, new_llm

load_dotenv()

llm = new_llm().bind_tools(TOOLS)

MAX_STEPS = 6


def run(query: str) -> str:
    messages = [HumanMessage(query)]

    for step in range(1, MAX_STEPS + 1):
        ai_msg = llm.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            return as_text(ai_msg.content)

        print(f"[step {step}] thought -> {len(ai_msg.tool_calls)} tool call(s)")
        for call in ai_msg.tool_calls:
            tool_msg = TOOLS_BY_NAME[call["name"]].invoke(call)
            print(f"  action: {call['name']}{call['args']} -> observation: {tool_msg.content}")
            messages.append(tool_msg)

    return "Gave up after max steps."


if __name__ == "__main__":
    answer = run("Add 4 and 6, then raise that result to the power of 2.")
    print("\nFinal answer:", answer)
