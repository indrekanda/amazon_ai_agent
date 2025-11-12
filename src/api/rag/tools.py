"""	All the tools available to the agent

Retrieval tool:
    get_formatted_context: the tool combining the functions:
    - get_embedding: embed the query
    - retrieve_context: retrieve the top k context
    - process_context: process the context    
"""

from langsmith import traceable, get_current_run_tree
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Filter, FieldCondition, MatchText, FusionQuery, MatchAny
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


### Items tool ###

@traceable(
    name="retrieve_top_n",
    run_type="retriever"
)
def retrieve_item_context(query, top_k=5):
    query_embedding = get_embedding(query)

    #qdrant_client = QdrantClient(url=f"{config.QDRANT_URL}")
    qdrant_client = QdrantClient(url=f"http://{config.QDRANT_URL}:6333")

    results = qdrant_client.query_points(
        collection_name = config.QDRANT_COLLECTION_NAME_ITEMS,
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
        retrieved_context_ids.append(result.payload['parent_asin']) # get actual product id
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
def process_item_context(context):

    formatted_context = ""
    # MODIFIED: to add price to the context
    for id, chunk, price in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_prices"]):
        formatted_context += f"- {id}, price: {price}, {chunk}\n"

    return formatted_context


# Tool to retrieve items top k context (combining retrieve_item_context and process_context)
def get_formatted_item_context(query: str, top_k: int = 5) -> str:

    """Get the top k context, each representing an inventory item for a given query.
    
    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more
    
    Returns:
        A string of the top k context chunks with IDs prepending each chunk, each representing an inventory item for a given query.
    """

    context = retrieve_item_context(query, top_k)
    formatted_context = process_item_context(context)

    return formatted_context


### Reviews tool ###

@traceable(
    name="retrieve_top_n",
    run_type="retriever"
)
def retrieve_review_context(query, item_list, top_k=20):
    query_embedding = get_embedding(query)

    qdrant_client = QdrantClient(url=config.QDRANT_URL)

    results = qdrant_client.query_points(
        collection_name=config.QDRANT_COLLECTION_NAME_REVIEWS,
        prefetch=[
            Prefetch(
                query=query_embedding,
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_asin",
                            match=MatchAny(
                                any=item_list
                            )
                        )
                    ]
                ),
                limit=top_k
            )
        ],
        query=FusionQuery(fusion="rrf"), # not really needed, as we only have one prefetch, which filters by id, but kept for consistency
        limit=top_k
    )

    retrieved_context_ids = []
    retrieved_context = []

    for result in results.points:
        retrieved_context_ids.append(result.payload['parent_asin'])
        retrieved_context.append(result.payload['text'])

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
    }


@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_review_context(context):

    formatted_context = ""

    for id, chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        formatted_context += f"- {id}: {chunk}\n"

    return formatted_context


def get_formatted_review_context(query: str, item_list: list[str], top_k: int = 20) -> str:

    """Get the top k reviews matching a query for a list of prefiltered items.
    
    Args:
        query: The query to get the top k reviews for
        item_list: The list of item IDs to prefilter for before running the query
        top_k: The number of reviews to retrieve, this should be at least 20 if multipple items are prefiltered
    
    Returns:
        A string of the top k context chunks with IDs prepending each chunk, each representing an inventory item for a given query.
    """

    context = retrieve_review_context(query, item_list, top_k)
    formatted_context = process_review_context(context)

    return formatted_context