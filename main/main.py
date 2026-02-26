from typing import Annotated, List
from dotenv import load_dotenv  
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, AnyMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode


from component.model import Model
from component.tool import extract_urls
from function.search.internet.internetSearch import InternetSearchAgent
from function.search.paper.paperSearch import PaperInfoSearchAgent
from function.search.web.webSearch import WebSearchAgent
from function.search.repo.repoSearch import RepoSearchAgent
class AgentState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages]

model = Model()

@tool
def internet_search(query: str) -> str:
    """Search general information on the internet."""
    agent = InternetSearchAgent()
    return agent.answer(query)


@tool
def web_search(query: str) -> str:
    """Extract and summarize content from a given URL."""
    url = extract_urls(query)
    agent = WebSearchAgent()
    return agent.answer(url,query)


@tool
def paper_search(query: str) -> str:
    """Search academic papers."""
    agent = PaperInfoSearchAgent()
    return agent.fullAnswer(query)


@tool
def repo_search(query: str) -> str:
    """Analyze GitHub repository."""
    url = extract_urls(query)
    agent = RepoSearchAgent()
    return agent.answer(url, query)
    

tools = [internet_search, web_search, paper_search, repo_search]

llm_with_tools = model.get_llm(tools)

tools = [web_search, internet_search, paper_search, repo_search]
def llm_node(state: AgentState):
    response = llm_with_tools.invoke(state.messages)
    return {"messages": [response]}
tool_node = ToolNode(tools)
def should_continue(state: AgentState):
    last_message = state.messages[-1]

    # Nếu LLM gọi tool → đi tool node
    if last_message.tool_calls:
        return "tool"

    # Nếu không → kết thúc
    return END
graph = StateGraph(AgentState)

graph.add_node("llm", llm_node)
graph.add_node("tool", tool_node)

graph.add_edge(START, "llm")

graph.add_conditional_edges(
    "llm",
    should_continue
)

# Sau khi tool chạy → quay lại LLM để tổng hợp
graph.add_edge("tool", "llm")

app = graph.compile()

result = app.invoke({
    "messages": [HumanMessage(content="Find research about RAG systems")]
})

print(result["messages"][-1].content)