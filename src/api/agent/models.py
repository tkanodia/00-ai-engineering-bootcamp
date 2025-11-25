from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Annotated, Literal
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

class Delegation(BaseModel):
    agent: str
    task: str

class CoordinatorAgentProperties(BaseModel):
    iteration: int = 0
    final_answer: bool = False
    plan: List[Delegation] = []
    next_agent: str = ""

class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    user_intent: str = ""
    product_qa_agent: AgentProperties = Field(default_factory=AgentProperties)
    shopping_cart_agent: AgentProperties = Field(default_factory=AgentProperties)
    warehouse_manager_agent: AgentProperties = Field(default_factory=AgentProperties)
    coordinator_agent: CoordinatorAgentProperties = Field(default_factory=CoordinatorAgentProperties)
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

class CoordinatorAgentResponse(BaseModel):
    next_agent: Literal["product_qa_agent", "shopping_cart_agent", "warehouse_manager_agent", ""] = Field(
        description="The name of the next agent to invoke. Must be one of: product_qa_agent, shopping_cart_agent, warehouse_manager_agent, or empty string '' when final_answer is true."
    )
    plan: List[Delegation]
    final_answer: bool
    answer: str
    
    @field_validator('next_agent')
    @classmethod
    def validate_next_agent(cls, v):
        """Ensure next_agent is a valid agent name, not OpenAI's internal tool names"""
        valid_agents = ["product_qa_agent", "shopping_cart_agent", "warehouse_manager_agent", ""]
        
        # If we get something like 'multi_tool_use.parallel', default to empty string
        if v not in valid_agents:
            return ""
        
        return v

class WarehouseManagerAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []