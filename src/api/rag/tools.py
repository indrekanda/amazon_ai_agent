"""	All the tools available to the agent

Retrieval tool:
    get_formatted_context: the tool combining the functions:
    - get_embedding: embed the query
    - retrieve_context: retrieve the top k context
    - process_context: process the context    
"""

from langsmith import traceable, get_current_run_tree
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Filter, FieldCondition, MatchText, FusionQuery
import openai
from api.core.config import config


# Functions needed to retrieve context, copied from retrieval.py
@traceable(
    name="embed_query",
    run_type="embedding",
    # metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"}
    metadata={"ls_provider": config.EMBEDDING_MODEL_PROVIDER, "ls_model_name": config.EMBEDDING_MODEL},
)
# def get_embedding(text, model="text-embedding-3-small"):
def get_embedding(text, model=config.EMBEDDING_MODEL):
    response = openai.embeddings.create(
        input=[text],
        model=model,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return response.data[0].embedding


@traceable(
    name="retrieve_top_n",
    run_type="retriever"
)
def retrieve_context(query, top_k=5):
    query_embedding = get_embedding(query)

    #qdrant_client = QdrantClient(url=f"{config.QDRANT_URL}")
    qdrant_client = QdrantClient(url=f"http://{config.QDRANT_URL}:6333")

    results = qdrant_client.query_points(
        collection_name = config.QDRANT_COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=query_embedding,
                limit=20
            ),
            Prefetch(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="text",
                            match=MatchText(text=query)
                        )
                    ]
                ),
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=top_k
    )

    retrieved_context_ids = []
    retrieved_context = []
    retrieved_prices = [] # NEW: to add to conext
    similarity_scores = []

    for result in results.points:
        retrieved_context_ids.append(result.id)
        retrieved_context.append(result.payload['text'])
        retrieved_prices.append(result.payload.get('price', 'N/A'))  #NEW: to add to conext
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "retrieved_prices": retrieved_prices, # NEW: to add to conext
        "similarity_scores": similarity_scores
    }


@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_context(context):

    formatted_context = ""
    # MODIFIED: to add price to the context
    for id, chunk, price in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_prices"]):
        formatted_context += f"- {id}, price: {price}, {chunk}\n"

    return formatted_context


# The tool combining the functions
# NEW: tool to retrieve the top k context
def get_formatted_context(query: str, top_k: int = 5) -> str:

    """Get the top k context, each representing an inventory item for a given query.
    
    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more
    
    Returns:
        A string of the top k context chunks with IDs prepending each chunk, each representing an inventory item for a given query.
    """

    context = retrieve_context(query, top_k)
    formatted_context = process_context(context)

    return formatted_context