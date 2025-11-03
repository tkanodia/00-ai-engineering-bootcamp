import logging
from fastapi import APIRouter, Request
from api.api.models import RAGRequest, RAGResponse, RAGUsedContext
from api.api.agent.graph import run_agent_wrapper


logger = logging.getLogger(__name__)

rag_router = APIRouter()

@rag_router.post("/")
def rag(request: Request, payload: RAGRequest) -> RAGResponse:

    answer = run_agent_wrapper(payload.query)
    return RAGResponse(
        request_id=request.state.request_id, 
        answer=answer["answer"],
        used_context=[RAGUsedContext(**item) for item in answer["used_context"]]
    )

api_router = APIRouter()

api_router.include_router(rag_router, prefix="/rag", tags=["rag"])