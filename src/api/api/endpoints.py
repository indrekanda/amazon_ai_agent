from fastapi import APIRouter, Request
import logging

from api.api.models import RAGRequest, RAGResponse, RAGUsedImage, FeedbackRequest, FeedbackResponse
#from api.rag.retrieval import rag_pipeline_wrapper
from api.rag.graph import run_agent_wrapper
from api.processors.submit_feedback import submit_feedback


logger = logging.getLogger(__name__)

# Routers 
rag_router = APIRouter()
feedback_router = APIRouter()

# Endpoint for RAG pipeline wrapper
@rag_router.post("/rag")
async def rag(
    request: Request, # is it RAGRequest from pydantic model or Request from FastAPI
    payload: RAGRequest # NEW: now it has thread_id
    ) -> RAGResponse: 
    
    #result = rag_pipeline_wrapper(payload.query)
    result = run_agent_wrapper(payload.query, payload.thread_id)
    
    used_image_urls = [RAGUsedImage(image_url=image["image_url"], price=image["price"], description=image["description"]) for image in result["retrieved_images"]]
    
    return RAGResponse(
        request_id=request.state.request_id,
        answer=result["answer"],
        used_image_urls=used_image_urls,
        trace_id=result["trace_id"],
    )


# Endpoint for feedback
@feedback_router.post("/submit_feedback")
async def send_feedback(
    request: Request, 
    payload: FeedbackRequest 
    ) -> FeedbackResponse: 
    
    submit_feedback(payload.trace_id, payload.feedback_score, payload.feedback_text, payload.feedback_source_type)
    
    return FeedbackResponse(
        request_id=request.state.request_id,
        status="success",
    )

# Main router for API endpoints
api_router = APIRouter()
api_router.include_router(rag_router, tags=["rag"])
api_router.include_router(feedback_router, tags=["feedback"])