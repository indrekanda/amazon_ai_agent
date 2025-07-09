import openai
from qdrant_client import QdrantClient
from langsmith import traceable, get_current_run_tree

from chatbot_ui.core.config import config

# It is not localhost, because we are running the app from docker compose, 
# If we are running locally, we can use localhost:6333
#qdrant_client = QdrantClient(url=f"http://{config.QDRANT_URL}:6333")

# Tracing / Evals: https://smith.langchain.com/


# Super naive RAG pipeline

# 1. Embed quer (data sample is already embedded)
@traceable(
    name="embed_query", # name for the span, any name
    run_type="embedding", # type of operation, predefined by langsmith
    metadata={"ls_provider": config.EMBEDDING_MODEL_PROVIDER, "ls_model_name": config.EMBEDDING_MODEL}, # keys are predefined by langsmith
)
def get_embedding(text, model=config.EMBEDDING_MODEL):
    response = openai.embeddings.create(
        input=[text],
        model=model,
    )
    
    # Add custom metadata for tracing
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens, # is not tracked by default, we add it manually
            "total_tokens": response.usage.total_tokens,
        }
    
    
    return response.data[0].embedding


# 2. Ask a query and retrieve the context (include query embedding)
@traceable(
    name="retrieve_top_n",
    run_type="retriever",
)
def retrieve_context(query, qdrant_client, top_k=5):
    query_embedding = get_embedding(query)
    results = qdrant_client.query_points(
        collection_name=config.QDRANT_COLLECTION_NAME,
        query = query_embedding,
        limit=top_k
    )
    
    # Add metadata to the run for debugging
    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    for point in results.points:
        retrieved_context_ids.append(point.id)
        retrieved_context.append(point.payload["text"])
        similarity_scores.append(point.score)    
    
    # Return a dictionary with the retrieved context (need to modify process_context to use this)
    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
    }


# 3. Format the context (after changing retrieve_context to return a dictionary, we change process_context to use it)
@traceable(
    name="format_retrieved_context",
    run_type="prompt",
)
def process_context(context):
    """Will return all the retrieved chunks in a formatted string"""
    formatted_context = ""
    for chunk in context["retrieved_context"]:
        formatted_context += f"- {chunk}\n"
    return formatted_context


# 4. Build the prompt
@traceable(
    name="render_prompt",
    run_type="prompt",
)
def build_prompt(context, question):
    
    processed_context = process_context(context)
    
    prompt = f"""
You are a shopping assistant that can answer questions about the products in stock.
You will be given a question and a list of context.
Instructions:
- You need to answer the question based on the provided context only.
- In your answer never use word "context", istead refer to it as "available products"
Context:
{processed_context}
Question:
{question}
"""
    return prompt


# 5. Answer the question
@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={"ls_provider": config.GENERATION_MODEL_PROVIDER, "ls_model_name": config.GENERATION_MODEL},
)
def generate_answer(prompt):
    response = openai.chat.completions.create(
        model="gpt-4.1", # Good and cheap
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    
    # Add custom metadata for tracing
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens, # is not tracked by default, we add it manually
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,  
        }
        
    return response.choices[0].message.content

# Nested functions become spans of the parent function
@traceable(
    name="rag_pipeline",
)
def rag_pipeline(question, qdrant_client, top_k=5):
    retrieved_context = retrieve_context(question, qdrant_client, top_k)
    prompt = build_prompt(retrieved_context, question)
    answer = generate_answer(prompt)
    
    # Collect all the results in a single dictionary for easier validation
    # But streamlit app will need to unpack it (need to modify the app)
    final_result = {
        "answer": answer,
        "question": question,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_context": retrieved_context["retrieved_context"],
        "similarity_scores": retrieved_context["similarity_scores"],
    }
    return final_result

