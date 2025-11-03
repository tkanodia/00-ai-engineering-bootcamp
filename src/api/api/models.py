from pydantic import BaseModel, Field
from typing import Optional, List
class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to use for the RAG pipeline")

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

