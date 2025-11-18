from langsmith import traceable, get_current_run_tree
from langchain_core.messages import convert_to_openai_messages, AIMessage
from openai import OpenAI
from jinja2 import Template 
from api.agent.utils.utils import format_ai_message
from api.agent.models import State, ProductQAAgentResponse, IntentRouterResponse, ShoppingCartAgentResponse   
import instructor
from api.agent.utils.prompt_management import prompt_template_config

@traceable(
    name="product_qa_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def product_qa_agent(state: State) -> dict:

    prompt_template =  prompt_template_config("src/api/agent/prompts/qa_agent.yaml", "qa_agent")

    prompt = prompt_template.render(
        available_tools=state.product_qa_agent.available_tools
    )

    messages = state.messages

    conversation = []

    for message in messages:
            conversation.append(convert_to_openai_messages(message))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
            model="gpt-4.1-mini",
            response_model=ProductQAAgentResponse,
            messages=[{"role": "system", "content": prompt}, *conversation],
            temperature=0,
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
      "product_qa_agent": {
        "iteration": state.product_qa_agent.iteration + 1,
        "final_answer": response.final_answer,
        "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
        "available_tools": state.product_qa_agent.available_tools
      },
      "answer": response.answer,
      "references": response.references
   }

@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def intent_router_node(state: State):

    prompt_template = prompt_template_config("src/api/agent/prompts/intent_router_agent.yaml", "intent_router_agent")
    
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
            temperature=0,
    )

    if response.user_intent == "product_qa":
      ai_message = []
    else:
        ai_message = [AIMessage(
            content=response.answer,
        )]

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }

        # get the trace_id from the current run - in intent router as this function always executes for our graph
        trace_id = str(getattr(current_run, "trace_id", current_run.id))

    return {
      "messages": ai_message,
      "user_intent": response.user_intent,
      "answer": response.answer
      }


@traceable(
    name="shopping_cart_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def shopping_cart_agent(state) -> dict:
    prompt_template = prompt_template_config("src/api/agent/prompts/shopping_agent.yaml", "shopping_agent_prompt")
    prompt = prompt_template.render(
        available_tools=state.shopping_cart_agent.available_tools,
        user_id=state.user_id,
        cart_id=state.cart_id
    )
    messages = state.messages
    
    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=ShoppingCartAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0,
    )


    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }

    ai_message = format_ai_message(response)

    return {
      "messages": [ai_message],
      "shopping_cart_agent": {
        "iteration": state.shopping_cart_agent.iteration + 1,
        "final_answer": response.final_answer,
        "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
        "available_tools": state.shopping_cart_agent.available_tools
      },
      "answer": response.answer
   }