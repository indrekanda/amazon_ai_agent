from fastapi import APIRouter, Request
import logging

from api.api.models import RAGRequest, RAGResponse
from api.rag.retrieval import rag_pipeline_wrapper


logger = logging.getLogger(__name__)


# Router for RAG endpoint
rag_router = APIRouter()

# Endpoint for RAG pipeline wrapper
@rag_router.post("/rag")
async def rag(
    request: Request, # is it RAGRequest from pydantic model or Request from FastAPI
    payload: RAGRequest
    ) -> RAGResponse: 
    
    result = rag_pipeline_wrapper(payload.query)
    
    return RAGResponse(
        request_id=request.state.request_id,
        answer=result
    )


# Main router for API endpoints
api_router = APIRouter()
api_router.include_router(rag_router, tags=["rag"])