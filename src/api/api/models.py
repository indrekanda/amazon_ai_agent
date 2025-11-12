# Pydantic modles for API

from pydantic import BaseModel, Field
from typing import List, Any, Optional, Union

class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline")
    thread_id: str = Field(..., description="The id of the thread") # NEW


class RAGUsedImage(BaseModel):
    image_url: str = Field(..., description="The url of the retrieved image")
    price: Optional[float] = Field(None, description="The price of the item")
    description: str = Field(..., description="The description of the item")


class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The id of the request")
    answer: str = Field(..., description="The content of the RAR response")
    used_image_urls: List[RAGUsedImage] = Field(..., description="The list of the retrieved images")
    trace_id: str = Field(..., description="The id of the trace")


# Flieds we expect from frontend
class FeedbackRequest(BaseModel):
    feedback_score: Union[int, None] = Field(None, description="1 if positive, 0 if negative") # to translate thumbs up/down to 1/0
    feedback_text: str = Field(..., description="The text of the feedback")
    trace_id: str = Field(..., description="The id of the trace")
    thread_id: str = Field(..., description="The id of the thread")
    feedback_source_type: str = Field(..., description="The type of the feedback source: API or human")


class FeedbackResponse(BaseModel):
    request_id: str = Field(..., description="The id of the request") # from the middleware
    status: str = Field(..., description="The status of the feedback submission") # success or error
    