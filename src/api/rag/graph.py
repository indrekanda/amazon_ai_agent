"""LangGraph graph (entry point to invoke the backend app) and everything that relates to the graph"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient

from api.core.config import config
from api.rag.tools import get_formatted_context
from api.rag.utils.utils import get_tool_descriptions_from_node
from api.rag.agent import ToolCall, RAGUsedContext, agent_node


# State of the agent
class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    answer: str = ""
    iteration: int = Field(default=0)
    final_answer: bool = Field(default=False)
    available_tools: List[Dict[str, Any]] = []
    tool_calls: Optional[List[ToolCall]] = Field(default_factory=list)
    retrieved_context_ids: Annotated[List[RAGUsedContext], add] = []

# Tool router: 
#   - if there are tool call return "tools"
#   - otherwise return "end"
# STOP condition: to run the agent only once
def tool_router(state: State) -> str:
    """Decide whether to continue or end"""
    
    if state.final_answer:
        return "end"
    elif state.iteration > 2:
        return "end"
    elif len(state.tool_calls) > 0:
        return "tools"
    else:
        return "end"



# Graph
workflow = StateGraph(State)

tools = [get_formatted_context] # list of tools to use
tool_node = ToolNode(tools)
tool_descriptions = get_tool_descriptions_from_node(tool_node)

workflow.add_node("agent_node", agent_node)
workflow.add_node("tool_node", tool_node)

workflow.add_edge(START, "agent_node")
workflow.add_conditional_edges(
    "agent_node",
    tool_router,
    {
        "tools": "tool_node",
        "end": END
    }
)
workflow.add_edge("tool_node", "agent_node")

graph = workflow.compile()


# Run the agent, returns a dict with all keys defined in the State class
# Replacement of rag_pipeline
def run_agent(question: str):
    """Run the agent"""
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "available_tools": tool_descriptions
    }
    result = graph.invoke(initial_state)
    return result


# Replacement of rag_pipeline_wrapper to get the answer and extract additional information
def run_agent_wrapper(question: str):
    """Run the agent"""
    
    qdrant_client = QdrantClient(url=f"http://{config.QDRANT_URL}:6333")

    result = run_agent(question)
    
    image_url_list = []
    for id in result.get("retrieved_context_ids"):
        payload = qdrant_client.retrieve(
            collection_name=config.QDRANT_COLLECTION_NAME,
            ids=[id.id],
        )[0].payload
        image_url = payload.get('first_large_image')
        price = payload.get('price')
        if image_url:
            image_url_list.append({
                "image_url": image_url,
                "price": price,
                "description": id.description,
            })  
    
    return {
        "answer": result.get("answer"),
        "retrieved_images": image_url_list,
    }