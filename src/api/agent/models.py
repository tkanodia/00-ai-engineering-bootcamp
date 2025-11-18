from pydantic import BaseModel, Field
from typing import List, Dict, Any, Annotated
from operator import add

class ToolCall(BaseModel):
    name: str
    arguments: dict

    
class RAGUsedContext(BaseModel):
    id: str = Field(description="ID of the item used to answer the question.")
    description: str = Field(description="Short description of the item used to answer the question.")

class ProductQAAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question.")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []

class AgentProperties(BaseModel):
    iteration: int = 0
    final_answer: bool = False
    available_tools: List[Dict[str, Any]] = []
    tool_calls: List[ToolCall] = []

class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    user_intent: str = ""
    product_qa_agent: AgentProperties = Field(default_factory=AgentProperties)
    shopping_cart_agent: AgentProperties = Field(default_factory=AgentProperties)
    answer: str = ""
    references: Annotated[List[RAGUsedContext], add] = []
    user_id: str = ""
    cart_id: str = ""
    trace_id: str = ""

class IntentRouterResponse(BaseModel):
    user_intent: str
    answer: str

class ShoppingCartAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []