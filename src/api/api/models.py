from pydantic import BaseModel, Field
from typing import Optional, List, Union
class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to use for the RAG pipeline")
    thread_id: str = Field(..., description="The thread id to use for the RAG pipeline")
class RAGUsedContext(BaseModel):
    image_url: str = Field(..., description="The image url of the item used to answer the question")
    price: Optional[float] = Field(..., description="The price of the item used to answer the question")
    description: str = Field(..., description="The description of the item used to answer the question")

class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The request id")
    answer: str = Field(..., description="The answer to the query")
    used_context: List[RAGUsedContext] = Field(..., description="information about the items used to answer the question")

class RAGGenerationResponseWithReferencesUsedContext(BaseModel):
    id: str = Field(description="The id of the item used to answer the question")
    description: str = Field(description="Short description of the item used to answer the question")
    rating: float = Field(description="The rating of the item used to answer the question")


class RAGGenerationResponseWithReferences(BaseModel):
    answer: str = Field(description="The answer to the question")
    references: list[RAGGenerationResponseWithReferencesUsedContext] = Field(description="List of items used to answer the question")
class FeedbackRequest(BaseModel):
    feedback_score: Union[int, None] = Field(..., description="1 if the feedback is positive, 0 if the feedback is negative")
    feedback_text: str = Field(default="", description="The feedback text")
    trace_id: Optional[str] = Field(default=None, description="The trace ID")
    thread_id: str = Field(..., description="The thread ID")
    feedback_source_type: str = Field(..., description="The type of feedback. Human or API")

class FeedbackResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    status: str = Field(..., description="The status of the feedback submission")