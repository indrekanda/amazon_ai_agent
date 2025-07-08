import openai
from qdrant_client import QdrantClient

from core.config import config

# It is not localhost, because we are running the app from docker compose, 
# If we are running locally, we can use localhost:6333
qdrant_client = QdrantClient(url=f"http://{config.QDRANT_URL}:6333")


# Super naive RAG pipeline

# 1. Embed quer (data sample is already embedded)
def get_embedding(text, model=config.EMBEDDING_MODEL):
    response = openai.embeddings.create(
        input=[text],
        model=model,
    )
    return response.data[0].embedding


# 2. Ask a query and retrieve the context (include query embedding)
def retrieve_context(query, top_k=5):
    query_embedding = get_embedding(query)
    results = qdrant_client.query_points(
        collection_name=config.QDRANT_COLLECTION_NAME,
        query = query_embedding,
        limit=top_k
    )
    return results


# 3. Format the context
def process_context(context):
    """Will return all the retrieved chunks in a formatted string"""
    formatted_context = ""
    for chunk in context:
        formatted_context += f"- {chunk}\n"
    return formatted_context


# 4. Build the prompt
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
def generate_answer(prompt):
    response = openai.chat.completions.create(
        model="gpt-4.1", # Good and cheap
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content


def rag_pipeline(question, top_k=5):
    retrieved_context = retrieve_context(question, top_k)
    prompt = build_prompt(retrieved_context, question)
    answer = generate_answer(prompt)
    return answer

