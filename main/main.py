from typing import Annotated, List, Literal
from dotenv import load_dotenv  
from pydantic import BaseModel, Field
from operator import add

from langchain_core.messages import HumanMessage, AnyMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langgraph.types import Command

from component.model import Model
from component.tool import extract_urls, extract_clean_text
from function.search.internet.internetSearch import InternetSearchAgent
from function.search.paper.paperSearch import PaperInfoSearchAgent
from function.search.web.webSearch import WebSearchAgent
from function.search.repo.repoSearch import RepoSearchAgent
class AgentState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages]
    task_name: str|None = None
    tool_result: Annotated[List[str], add] = []

class RootTask(BaseModel):
    task_name: List[Literal['search','analyze']]= Field(
        default = 'search',
        description='choose task for specific application'
    )

class SearchTask(BaseModel):
    task_name: Literal['internet', 'repo', 'paper', 'web'] = Field(
        default="internet",
        description = "Task for search method in distinct way"
    )

class AnalyzeTask(BaseModel):
    task_name: Literal['code', 'paragraph'] = Field(
        default = 'paragraph',
        description="Distinct way to analyze"
    )

root_model = ChatOllama(
    model="qwen2.5:1.5b-instruct",
    temperature=0
)

model = Model().get_llm()

root_router = root_model.with_structured_output(RootTask)
search_router = root_model.with_structured_output(SearchTask)
analyze_router = root_model.with_structured_output(AnalyzeTask)


def ensure_list_str(value) -> list[str]:
    """
    LangGraph-safe normalization.
    Never return None.
    """
    if value is None:
        return ["No information found."]

    if isinstance(value, list):
        cleaned = [str(v) for v in value if v is not None and str(v).strip()]
        return cleaned if cleaned else ["No information found."]

    text = str(value).strip()
    return [text] if text else ["No information found."]

def root_node(state: AgentState):
    decision = root_router.invoke(state.messages)

    return Command(
        update={"task_name": decision.task_name},
        goto=f"{decision.task_name}_node"
    )

def search_node(state: AgentState):
    decision = search_router.invoke(state.messages)

    return Command(
        update={"task_name": decision.task_name},
        goto=f"{decision.task_name}_search_node"
    )

def analyze_node(state: AgentState):
    decision = analyze_router.invoke(state.messages)

    return Command(
        update={"task_name": decision.task_name},
        goto=f"{decision.task_name}_analyze_node"
    )

# Search
def internet_search_node(state: AgentState):
    """Search general information on the internet."""
    query = next(
        m.content for m in reversed(state.messages)
        if isinstance(m, HumanMessage)
    )
    agent = InternetSearchAgent()
    raw = agent.answer(query)
    return Command(
       # update={"tool_result": [agent.answer(query)]},
       update = {"tool_result": ensure_list_str(raw)},
        goto="final_node")

def web_search_node(state: AgentState):
    """Extract and summarize content from a given URL."""
    query = next(
        m.content for m in reversed(state.messages)
        if isinstance(m, HumanMessage)
    )
    url = extract_urls(query)
    question = extract_clean_text(query)
    agent = WebSearchAgent()
    raw = agent.answer(query)
    return Command(
        #update = {"tool_result": [agent.answer(url,question)]},
        update = {"tool_result": ensure_list_str(raw)},
        goto="final_node"
    )

def repo_search_node(state: AgentState):
    """Analyze GitHub repository."""
    query = next(
        m.content for m in reversed(state.messages)
        if isinstance(m, HumanMessage)
    )
    url = extract_urls(query)
    agent = RepoSearchAgent()
    raw = agent.answer(url, query)
    return Command(
        update = {"tool_result": ensure_list_str(raw)},
        goto="final_node")

def paper_search(state: AgentState):
    """Search academic papers."""
    query = next(
        m.content for m in reversed(state.messages)
        if isinstance(m, HumanMessage)
    )
    agent = PaperInfoSearchAgent()
    raw = agent.fullAnswer(query)
    return Command(
        update = {"tool_result": ensure_list_str(raw)},
        goto="final_node"
    )

# Analyze

def final_node(state: AgentState):

    user_query = next(
        m.content for m in reversed(state.messages)
        if isinstance(m, HumanMessage)
    )
    safe_results = ensure_list_str(state.tool_result)
    collected_info = "\n\n".join(safe_results)
    #collected_info = "\n\n".join(state.tool_result)

    prompt = f"""
You are a helpful AI assistant.

User question:
{user_query}

Information gathered:
{collected_info}

Write a clear, concise answer for the user.
If multiple sources exist, synthesize them.
Do not mention internal tools.
"""

    response = model.invoke(prompt)
    return Command(
        update={"messages": [AIMessage(content = response.content)]},
        goto=END
    )

builder = StateGraph(AgentState)
builder.add_node("root_node", root_node)
builder.add_node("search_node", search_node)
builder.add_node("analyze_node", analyze_node)
builder.add_node("internet_search_node", internet_search_node)
builder.add_node("web_search_node", web_search_node)
builder.add_node("repo_search_node", repo_search_node)
builder.add_node("paper_search_node", paper_search)
builder.add_node("final_node", final_node)    
builder.set_entry_point("root_node")

graph = builder.compile()
result = graph.invoke({
    "messages": [HumanMessage(content="Explain this repo https://github.com/unslothai/unsloth and find related papers")]
})




# router_llm = ChatOllama(
#     model="qwen2.5:1.5b-instruct",
#     base_url="http://localhost:11434",
#     temperature=0
# ).with_structured_output(Task)

# def router_node(state: AgentState):
#     decision = router_llm.invoke(state.messages)

#     return {
#         "task_name": decision.task_name
#     }

# @tool
# def internet_search(query: str) -> str:
#     """Search general information on the internet."""
#     agent = InternetSearchAgent()
#     return agent.answer(query)


# @tool
# def web_search(query: str) -> str:
#     """Extract and summarize content from a given URL."""
#     url = extract_urls(query)
#     agent = WebSearchAgent()
#     return agent.answer(url,query)


# @tool
# def paper_search(query: str) -> str:
#     """Search academic papers."""
#     agent = PaperInfoSearchAgent()
#     return agent.fullAnswer(query)


# @tool
# def repo_search(query: str) -> str:
#     """Analyze GitHub repository."""
#     url = extract_urls(query)
#     agent = RepoSearchAgent()
#     return agent.answer(url, query)
    

# tools = [internet_search, web_search, paper_search, repo_search]

# llm_with_tools = model.get_llm(tools)

# tools = [web_search, internet_search, paper_search, repo_search]
# def llm_node(state: AgentState):
#     response = llm_with_tools.invoke(state.messages)
#     return {"messages": [response]}
# tool_node = ToolNode(tools)
# def should_continue(state: AgentState):
#     last_message = state.messages[-1]

#     # Nếu LLM gọi tool → đi tool node
#     if last_message.tool_calls:
#         return "tool"

#     # Nếu không → kết thúc
#     return END
# graph = StateGraph(AgentState)

# graph.add_node("llm", llm_node)
# graph.add_node("tool", tool_node)

# graph.add_edge(START, "llm")

# graph.add_conditional_edges(
#     "llm",
#     should_continue
# )

# # Sau khi tool chạy → quay lại LLM để tổng hợp
# graph.add_edge("tool", "llm")

# app = graph.compile()

# result = app.invoke({
#     "messages": [HumanMessage(content="Find research about RAG systems")]
# })

# print(result["messages"][-1].content)