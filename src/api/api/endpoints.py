from fastapi import APIRouter, Request
import logging

from api.api.models import RAGRequest, RAGResponse, RAGUsedImage
#from api.rag.retrieval import rag_pipeline_wrapper
from api.rag.graph import run_agent_wrapper


logger = logging.getLogger(__name__)

# Router for RAG endpoint
rag_router = APIRouter()

# Endpoint for RAG pipeline wrapper
@rag_router.post("/rag")
async def rag(
    request: Request, # is it RAGRequest from pydantic model or Request from FastAPI
    payload: RAGRequest
    ) -> RAGResponse: 
    
    #result = rag_pipeline_wrapper(payload.query)
    result = run_agent_wrapper(payload.query)
    
    used_image_urls = [RAGUsedImage(image_url=image["image_url"], price=image["price"], description=image["description"]) for image in result["retrieved_images"]]
    
    return RAGResponse(
        request_id=request.state.request_id,
        answer=result["answer"],
        used_image_urls=used_image_urls,
    )

# Main router for API endpoints
api_router = APIRouter()
api_router.include_router(rag_router, tags=["rag"])