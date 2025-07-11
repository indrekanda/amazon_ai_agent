# Pydantic modles for API

from pydantic import BaseModel, Field
from typing import List, Any, Optional

class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline")


class RAGUsedImage(BaseModel):
    image_url: str = Field(..., description="The url of the retrieved image")
    price: Optional[float] = Field(None, description="The price of the item")
    description: str = Field(..., description="The description of the item")


class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The id of the request")
    answer: str = Field(..., description="The content of the RAR response")
    used_image_urls: List[RAGUsedImage] = Field(..., description="The list of the retrieved images")

