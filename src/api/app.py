from fastapi import FastAPI, Request
from pydantic import BaseModel

from openai import OpenAI
from groq import Groq
from google import genai

from src.api.core.config import config

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_llm(provider, model_name, messages, max_tokens=500):

    if provider == "OpenAI":
        client = OpenAI(api_key=config.OPENAI_API_KEY)
    elif provider == "Groq":
        client = Groq(api_key=config.GROQ_API_KEY)
    else:
        client = genai.Client(api_key=config.GOOGLE_API_KEY)

    if provider == "Google":
        return client.models.generate_content(
            model=model_name,
            contents=[message["content"] for message in messages],
        ).text
    else:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens
        ).choices[0].message.content


class ChatRequest(BaseModel):
    provider: str
    model_name: str
    messages: list[dict]

class ChatResponse(BaseModel):
    message: str


app = FastAPI()

@app.post("/chat")
def chat(
    request: Request,
    payload: ChatRequest
) -> ChatResponse:
    

    result = run_llm(payload.provider, payload.model_name, payload.messages)

    return ChatResponse(message=result)

@app.get("/")
async def root():
    """Root endpoint that returns a welcome message"""
    return {"message": "Hello API"}