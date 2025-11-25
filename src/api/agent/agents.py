from typing import Type, Any, List, Dict, Optional
from langsmith import traceable, get_current_run_tree
from langchain_core.messages import convert_to_openai_messages, AIMessage
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from api.agent.utils.utils import format_ai_message, filter_messages_for_coordinator
from api.agent.models import State, ProductQAAgentResponse, ShoppingCartAgentResponse, CoordinatorAgentResponse, WarehouseManagerAgentResponse
import instructor
from api.agent.utils.prompt_management import prompt_template_config
from litellm import completion

# same prompt may not work well for every llm. We can tie prompts to models and llms for litellm router. Variation of prompts for different models.
# This function is used to create chat completions with instructor and litellm


def chat_completions(
    conversation: list, 
    prompt_path: str,
    response_model: Type[Any],
    prompt_params: Optional[Dict[str, Any]] = {},
    agent_tools: Optional[List[Dict[str, Any]]] = [],
    models: List[str] = ["gpt-4.1", "groq/llama-3.3-70b-versatile"] 
) -> tuple[Any, Any]:
    """
    Create chat completions with instructor and litellm
    
    Args:
        model: List of model names for litellm router
        prompt_path: Path to the prompt template
        response_model: Pydantic model for structured output
        messages: List of chat messages
        response_model: Pydantic model for structured output
        
    Returns:
        Tuple of (parsed_response, raw_response)
    """
    # Create instructor client from litellm
    client = instructor.from_litellm(completion)
    
    response = None
    raw_response = None
    last_error = None
    
    for model in models:
        try:
            # Load model-specific prompt from YAML (uses model name as key)
            prompt_template = prompt_template_config(prompt_path, model, model=model)
            # Render the template to get the final prompt string
            prompt = prompt_template.render(
                available_tools=agent_tools,
                **prompt_params
            )
            response, raw_response = client.chat.completions.create_with_completion(
                model=model,
                messages=[{"role": "system", "content": prompt}, *conversation],
                response_model=response_model,
                temperature=0
            )
            break  # Success, exit loop
        except Exception as e:
            print(f"Error with model {model}: {e}")
            last_error = e
            continue
    
    # If all models failed, raise error
    if response is None or raw_response is None:
        raise RuntimeError(f"All models failed. Last error: {last_error}")
    
    return response, raw_response

@traceable(
    name="product_qa_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def product_qa_agent(state: State) -> dict:

    messages = state.messages

    conversation = []

    for message in messages:
            conversation.append(convert_to_openai_messages(message))

    response, raw_response = chat_completions(
        conversation=conversation,
        prompt_path="src/api/agent/prompts/qa_agent.yaml",
        response_model=ProductQAAgentResponse,
        agent_tools=state.product_qa_agent.available_tools
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
    name="coordinator_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def coordinator_agent(state):

    messages = state.messages
    
    # DEBUG: Log coordinator iteration
    print(f"🔄 Coordinator iteration {state.coordinator_agent.iteration + 1}")
    print(f"📊 Current state - Messages count: {len(messages)}")
    
    # Filter messages to remove tool-related messages that would confuse the coordinator
    # The coordinator doesn't use tools, only delegates to agents
    # TEMPORARILY DISABLED to debug - using raw messages
    # filtered_messages = filter_messages_for_coordinator(messages)

    conversation = []

    for message in messages:
            conversation.append(convert_to_openai_messages(message))

    response, raw_response = chat_completions(
        conversation=conversation,
        prompt_path="src/api/agent/prompts/coordinator_agent.yaml",
        response_model=CoordinatorAgentResponse,
        agent_tools=[]
    )
    
    # DEBUG: Log coordinator decision
    print(f"🎯 Coordinator decision: next_agent='{response.next_agent}', final_answer={response.final_answer}")
    print(f"📋 Plan: {response.plan}")

    if response.final_answer:
        ai_message = [AIMessage(
            content=response.answer,
        )]
    else:
        ai_message = []

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
        "answer": response.answer,
        "coordinator_agent": {
            "iteration": state.coordinator_agent.iteration + 1,
            "final_answer": response.final_answer,
            "next_agent": response.next_agent,
            "plan": [data.model_dump() for data in response.plan]
        },
        "trace_id": trace_id
   }

# @traceable(
#     name="agent_node",
#     run_type="llm",
#     metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
# )
# def intent_router_node(state: State):

#     prompt_template = prompt_template_config("src/api/agent/prompts/intent_router_agent.yaml", "intent_router_agent")
    
#     prompt = prompt_template.render()

#     messages = state.messages

#     conversation = []

#     for message in messages:
#             conversation.append(convert_to_openai_messages(message))

#     client = instructor.from_openai(OpenAI())

#     response, raw_response = client.chat.completions.create_with_completion(
#             model="gpt-4.1",
#             response_model=IntentRouterResponse,
#             messages=[{"role": "system", "content": prompt}, *conversation],
#             temperature=0,
#     )

#     if response.user_intent == "product_qa":
#       ai_message = []
#     else:
#         ai_message = [AIMessage(
#             content=response.answer,
#         )]

#     current_run = get_current_run_tree()
#     if current_run:
#         current_run.metadata["usage_metadata"] = {
#             "input_tokens": raw_response.usage.prompt_tokens,
#             "output_tokens": raw_response.usage.completion_tokens,
#             "total_tokens": raw_response.usage.total_tokens
#         }

#         # get the trace_id from the current run - in intent router as this function always executes for our graph
#         trace_id = str(getattr(current_run, "trace_id", current_run.id))

#     return {
#       "messages": ai_message,
#       "user_intent": response.user_intent,
#       "answer": response.answer
#       }


@traceable(
    name="shopping_cart_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def shopping_cart_agent(state) -> dict:
    messages = state.messages
    
    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    response, raw_response = chat_completions(
        conversation=conversation,
        prompt_path="src/api/agent/prompts/shopping_agent.yaml",
        response_model=ShoppingCartAgentResponse,
        prompt_params=dict(
            user_id=state.user_id,
            cart_id=state.cart_id
        ),
        agent_tools=state.shopping_cart_agent.available_tools
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


@traceable(
    name="warehouse_manager_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def warehouse_manager_agent(state) -> dict:

    messages = state.messages

    conversation = []

    for message in messages:
            conversation.append(convert_to_openai_messages(message))

    response, raw_response = chat_completions(
        conversation=conversation,
        prompt_path="src/api/agent/prompts/warehouse_agent.yaml",
        response_model=WarehouseManagerAgentResponse,
        agent_tools=state.warehouse_manager_agent.available_tools
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
        "warehouse_manager_agent": {
            "iteration": state.warehouse_manager_agent.iteration + 1,
            "final_answer": response.final_answer,
            "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
            "available_tools": state.warehouse_manager_agent.available_tools
        },
        "answer": response.answer
    }