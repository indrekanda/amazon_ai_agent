"""	Agent node that does tool callin""" 
from pydantic import BaseModel, Field
from typing import List
import instructor
from openai import OpenAI
from langsmith import traceable, get_current_run_tree
from langchain_core.messages import AIMessage

from api.rag.utils.utils import lc_messages_to_regular_messages
from api.rag.utils.utils import prompt_template_config
from api.core.config import config


# Pydantic models are needed for structured output, specifically Instructor

class ToolCall(BaseModel):
    name: str
    arguments: dict

class RAGUsedContext(BaseModel):
    id: int
    description: str

class AgentResponse(BaseModel):
    answer: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    final_answer: bool = Field(default=False)
    retrieved_context_ids: List[RAGUsedContext]



# Define the agent (cpoy agent node from notebook)

@traceable(
   name="agent_node",
   run_type="llm",
   metadata={"ls_provider": config.GENERATION_MODEL_PROVIDER, "ls_model_name": config.GENERATION_MODEL},
)

def agent_node(state) -> dict:

    # yaml file path and prompt template name
    prompt_template = prompt_template_config(config.RAG_PROMPT_TEMPLATE_PATH,"rag_generation") 
    prompt = prompt_template.render(available_tools=state.available_tools)

    messages = state.messages
    conversation = []
    for msg in messages:
        conversation.append(lc_messages_to_regular_messages(msg))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model=config.GENERATION_MODEL,
        response_model=AgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
    )
    
    # Add custom metadata for tracing
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens, # is not tracked by default, we add it manually
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,  
        }

    # If there are tool calls, add them to the message
    if response.tool_calls and not response.final_answer:
        tool_calls = []
        for i, tc in enumerate(response.tool_calls):
            tool_calls.append({
                "id": f"call_{i}",
                "name": tc.name,
                "args": tc.arguments
            })

        ai_message = AIMessage(
            content=response.answer,
            tool_calls=tool_calls
            )
    else:
        ai_message = AIMessage(
            content=response.answer,
        )

    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "retrieved_context_ids": response.retrieved_context_ids
    }




