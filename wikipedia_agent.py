import wikipedia
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

TOP_K_RESULTS = 1
DOC_CONTENT_CHARS_MAX = 300


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return a short summary of the top matching page."""
    titles = wikipedia.search(query, results=TOP_K_RESULTS)
    summaries = []
    for title in titles[:TOP_K_RESULTS]:
        try:
            page = wikipedia.page(title=title, auto_suggest=False)
        except (wikipedia.PageError, wikipedia.DisambiguationError):
            continue
        summaries.append(f"Page: {title}\nSummary: {page.summary}")
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)[:DOC_CONTENT_CHARS_MAX]


def as_text(content) -> str:
    """Gemini returns content as either a plain string or a list of parts."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


load_dotenv()

# Initialize the LLM with Gemini and bind the tools
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
llm_with_tools = llm.bind_tools([wikipedia_search])

messages = [HumanMessage("What was the most impressive thing about Narendra Modi?")]

ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

if ai_msg.tool_calls:
    for tool_call in ai_msg.tool_calls:
        tool_msg = wikipedia_search.invoke(tool_call)
        print(tool_msg.name)
        print(tool_call["args"])
        print(tool_msg.content)
        messages.append(tool_msg)

    print()
    final_response = llm_with_tools.invoke(messages)
    print(as_text(final_response.content))
else:
    # model answered directly without needing the tool
    print(as_text(ai_msg.content))
