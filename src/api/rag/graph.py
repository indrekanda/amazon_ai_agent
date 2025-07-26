"""LangGraph graph (entry point to invoke the backend app) and everything that relates to the graph"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
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
    # NEW: as we implement multi-turn, we need to store the context ids, we dont wnat to show all 
    # history of suggestions in the streamlit sidebasr, thus we remove the add
    retrieved_context_ids: List[RAGUsedContext] = [] 
    #retrieved_context_ids: Annotated[List[RAGUsedContext], add] = []

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


# Run the agent, returns a dict with all keys defined in the State class
# Replacement of rag_pipeline
def run_agent(question: str, thread_id: str):
    """Run the agent"""
    
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "iteration": 0, # NEW, reset the iteration counter (for eahc query)
        "available_tools": tool_descriptions
    }
    
    # NEW, add a thread id to the graph config
    # NEW: Thred id need tobe dynamic and come from fornt end
    graph_config = {"configurable": {"thread_id": thread_id}}
    
    # NEW, Context manager to save the state of the graph to the database
    # Change localhost:5433 to postgres:5432 to connect to the database inside the container
    with PostgresSaver.from_conn_string("postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db") as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
        result = graph.invoke(initial_state, config=graph_config)
    return result


# Replacement of rag_pipeline_wrapper to get the answer and extract additional information
# NEW, add thread_id to inputs (endpoint takes it as an argument; we generate it in the streamlit app)
def run_agent_wrapper(question: str, thread_id: str):
    """Run the agent"""
    
    qdrant_client = QdrantClient(url=f"http://{config.QDRANT_URL}:6333")

    result = run_agent(question, thread_id) # NEW, add thread_id to inputs
    
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