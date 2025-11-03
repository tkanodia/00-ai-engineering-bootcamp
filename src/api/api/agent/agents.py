from langsmith import traceable, get_current_run_tree
from langchain_core.messages import convert_to_openai_messages
from openai import OpenAI
from jinja2 import Template 
from api.api.agent.utils.utils import format_ai_message
from api.api.agent.models import State, AgentResponse, IntentRouterResponse   
import instructor
from api.api.agent.utils.prompt_management import prompt_template_config

@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def agent_node(state: State) -> dict:

    prompt_template =  prompt_template_config("src/api/api/agent/prompts/qa_agent.yaml", "qa_agent")

    prompt = prompt_template.render(
        available_tools=state.available_tools
    )

    messages = state.messages

    conversation = []

    for message in messages:
            conversation.append(convert_to_openai_messages(message))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
            model="gpt-4.1-mini",
            response_model=AgentResponse,
            messages=[{"role": "system", "content": prompt}, *conversation],
            temperature=0.5,
    )

    ai_message = format_ai_message(response)

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }

    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "references": response.references
    }

@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def intent_router_node(state: State):

    prompt_template = prompt_template_config("src/api/api/agent/prompts/intent_router_agent.yaml", "intent_router_agent")
    
    prompt = prompt_template.render()

    messages = state.messages

    conversation = []

    for message in messages:
            conversation.append(convert_to_openai_messages(message))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
            model="gpt-4.1-mini",
            response_model=IntentRouterResponse,
            messages=[{"role": "system", "content": prompt}, *conversation],
            temperature=0.5,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }

    return {
        "question_relevant": response.question_relevant,
        "answer": response.answer
    }


