from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages.tool import ToolMessage
from langgraph.graph import StateGraph

load_dotenv()


# -- 1) Define our single business tool
@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order that hasn't shipped."""
    # (Here you'd call your real backend API)
    return f"Order {order_id} has been cancelled."


class AgentState(TypedDict):
    order: dict
    messages: list


# -- 2) The agent "brain": invoke LLM, run tool, then invoke LLM again
def call_model(state: AgentState):
    msgs = state["messages"]
    order = state.get("order", {"order_id": "UNKNOWN"})

    # System prompt tells the model exactly what to do
    prompt = (
        f'''You are an ecommerce support agent.
ORDER ID: {order['order_id']}
If the customer asks to cancel, call cancel_order(order_id)
and then send a simple confirmation.
Otherwise, just respond normally.'''
    )
    full = [SystemMessage(prompt)] + msgs

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0).bind_tools([cancel_order])

    # 1st LLM pass: decides whether to call our tool
    first = llm.invoke(full)
    out = [first]

    if getattr(first, "tool_calls", None):
        # run the cancel_order tool
        tc = first.tool_calls[0]
        result = cancel_order.invoke(tc["args"])
        out.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        # 2nd LLM pass: generate the final confirmation text
        second = llm.invoke(full + out)
        out.append(second)

    return {"messages": out}


# -- 3) Wire it all up in a StateGraph
def construct_graph():
    g = StateGraph(AgentState)
    g.add_node("assistant", call_model)
    g.set_entry_point("assistant")
    return g.compile()


graph = construct_graph()

if __name__ == "__main__":
    example_order = {"order_id": "A12345"}
    convo = [HumanMessage(content="Please cancel my order A12345.")]
    result = graph.invoke({"order": example_order, "messages": convo})
    for msg in result["messages"]:
        print(f"{msg.type}: {msg.content}")
