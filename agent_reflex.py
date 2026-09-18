"""
Reflex Agent
============
The simplest agentic pattern: stimulus -> action, no deliberation loop.

One LLM call maps the input directly to zero or more tool calls. Whatever
the tools return IS the final answer — there's no second pass where the
LLM looks at the result, thinks about it, or phrases it nicely.

    [LLM call] -> tool_calls -> [execute tools] -> print raw result. Done.

Good for: fast, cheap, single-turn lookups where the tool output needs no
further interpretation.
Bad for: multi-step tasks (a reflex agent can't act on a tool's result,
because it never looks at the tool's result), self-correction, or answers
that need synthesis into prose.
"""
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from calc_tools import TOOLS, TOOLS_BY_NAME, new_llm

load_dotenv()

llm = new_llm().bind_tools(TOOLS)


def run(query: str) -> None:
    ai_msg = llm.invoke([HumanMessage(query)])

    if not ai_msg.tool_calls:
        # No stimulus matched a tool -> nothing to react to.
        print(ai_msg.content)
        return

    for call in ai_msg.tool_calls:
        tool_msg = TOOLS_BY_NAME[call["name"]].invoke(call)
        print(f"{call['name']}{call['args']} = {tool_msg.content}")


if __name__ == "__main__":
    run("add 4 and 6, then raise that result to the power of 2.")
