from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from api.agent.tools import get_formatted_item_context, get_formatted_reviews_context
from api.agent.agents import product_qa_agent, shopping_cart_agent, intent_router_node
from api.agent.tools import add_to_shopping_cart, remove_from_cart, get_shopping_cart
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langgraph.checkpoint.postgres import PostgresSaver
import json
import numpy as np
from api.agent.agents import product_qa_agent, shopping_cart_agent, intent_router_node
from api.agent.tools import add_to_shopping_cart, remove_from_cart, get_shopping_cart
from api.agent.models import State
from api.agent.utils.utils import get_tool_descriptions


### Routers for the workflow
def product_qa_agent_tool_router(state) -> str:
    """Decide whether to continue or end"""
    
    if state.product_qa_agent.final_answer:
        return "end"
    elif state.product_qa_agent.iteration > 4:
        return "end"
    elif len(state.product_qa_agent.tool_calls) > 0:
        return "tools"
    else:
        return "end"

def shopping_cart_agent_tool_router(state) -> str:
    """Decide whether to continue or end"""
    
    if state.shopping_cart_agent.final_answer:
        return "end"
    elif state.shopping_cart_agent.iteration > 2:
        return "end"
    elif len(state.shopping_cart_agent.tool_calls) > 0:
        return "tools"
    else:
        return "end"

def user_intent_router(state) -> str:
    """Decide whether to continue or end"""
    
    if state.user_intent == "product_qa":
        return "product_qa_agent"
    elif state.user_intent == "shopping_cart":
        return "shopping_cart_agent"
    else:
        return "end"

### Workflow


workflow = StateGraph(State)

product_qa_agent_tools = [get_formatted_item_context, get_formatted_reviews_context]
product_qa_agent_tool_node = ToolNode(product_qa_agent_tools)
product_qa_agent_tool_descriptions = get_tool_descriptions(product_qa_agent_tools)

shopping_cart_agent_tools = [add_to_shopping_cart, remove_from_cart, get_shopping_cart]
shopping_cart_agent_tool_node = ToolNode(shopping_cart_agent_tools)
shopping_cart_agent_tool_descriptions = get_tool_descriptions(shopping_cart_agent_tools)

workflow.add_node("product_qa_agent", product_qa_agent)
workflow.add_node("shopping_cart_agent", shopping_cart_agent)
workflow.add_node("intent_router", intent_router_node)

workflow.add_node("product_qa_agent_tool_node", product_qa_agent_tool_node)
workflow.add_node("shopping_cart_agent_tool_node", shopping_cart_agent_tool_node)

workflow.add_edge(START, "intent_router")

workflow.add_conditional_edges(
    "intent_router",
    user_intent_router,
    {
        "product_qa_agent": "product_qa_agent",
        "shopping_cart_agent": "shopping_cart_agent",
        "end": END
    }
)

workflow.add_conditional_edges(
    "product_qa_agent",
    product_qa_agent_tool_router,
    {
        "tools": "product_qa_agent_tool_node",
        "end": END
    }
)

workflow.add_conditional_edges(
    "shopping_cart_agent",
    shopping_cart_agent_tool_router,
    {
        "tools": "shopping_cart_agent_tool_node",
        "end": END
    }
)

workflow.add_edge("product_qa_agent_tool_node", "product_qa_agent")
workflow.add_edge("shopping_cart_agent_tool_node", "shopping_cart_agent")


# Why do we need this edge?
# This edge routes the flow from the 'tool_node' (where the tool output is computed) back to the 'agent_node',
# enabling the agent to use the results of tool calls to produce a final answer or potentially call more tools.
# It's essential in agent-tooling workflows where the agent needs to reason over new information or iterate on next actions.
# workflow.add_edge("tool_node", "agent_node")

## Agent execution function to invoke the graph


# def run_agent(query: str, thread_id: str):
#     initial_state = {
#         "messages": [{"role": "user", "content": query}],
#         "iteration": 0,
#         "available_tools": tool_descriptions
#     }

#     # thread id should come from frontend as user could have multiple conversations with the agent

#     config = {"configurable": {"thread_id": thread_id}}
#     with PostgresSaver.from_conn_string("postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db") as checkpointer:
#         graph = workflow.compile(checkpointer=checkpointer)
#         result = graph.invoke(initial_state, config=config)

#     return result



# # Sample data chunks from graph.stream to understand how chunks are structured
# chunk[0] -> debug or updates
# chunk[1] -> checkpoint / task / task_result 
# run cell 50 in 03-Streaming-State.ipynb to understand the data structure 


def run_agent_stream_wrapper(question: str, thread_id: str):
    """
    This function is a streaming execution wrapper for invoking an agent workflow.
    It processes a question and a thread_id, builds the initial execution state,
    manages graph workflow execution with a database checkpointer, and yields streaming events 
    that describe the agent's reasoning and tool use in real time.

    At a high level, here's what happens in this function:

    Pseudo-algorithm:
    1. Define helper functions for processing messages and workflow events for streaming.
       - Format messages for Server-Sent Events (SSE).
       - Interpret the meaning of workflow 'chunks' (events emitted during graph execution).
    2. Create a client for the vector database (Qdrant) (for use in tools; side effect).
    3. Build the initial workflow state with the question and relevant tool info.
    4. Define workflow configuration parameters, including the thread ID for context.
    5. Open a connection to a Postgres-backed checkpointer to manage graph state.
    6. Compile the workflow with the checkpointer (so state is persisted).
    7. Stream execution of the workflow graph:
       a. For each chunk of streamed workflow output:
           i.   Optionally process and interpret the chunk to a user-facing message.
           ii.  (To "yield" a message means to immediately send each generated message/event to the 
            caller as soon as it is available, allowing for real-time streaming instead of waiting for
            all results before returning. In Python, this is commonly done using the `yield` keyword 
            inside a generator function.)
    """


    def _string_for_sse(message: str):
        return f"data: {message}\n\n"

    def _process_graph_event(chunk):

        def _is_node_start(chunk):
            return chunk[1].get("type") == "task"

        def _is_node_end(chunk):
            return chunk[0] == "updates"

        def _tool_to_text(tool_call):
            if tool_call.name == "get_formatted_item_context":
                return f"Looking for items: {tool_call.arguments.get('query', '')}."
            elif tool_call.name == "get_formatted_reviews_context":
                return f"Fetching user reviews..."
            else:
                return "Unknown tool call..."

        if _is_node_start(chunk):
            if chunk[1].get("payload", {}).get("name") == "intent_router_node":
                return "Analysing the question..."
            if chunk[1].get("payload", {}).get("name") == "agent_node":
                return "Planning..."
            if chunk[1].get("payload", {}).get("name") == "tool_node":
                message = " ".join([_tool_to_text(tool_call) for tool_call in chunk[1].get('payload', {}).get('input', {}).tool_calls])
                return message
        else:
            return False

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "user_intent": "",
        "product_qa_agent": {
            "iteration": 0,
            "final_answer": False,
            "available_tools": product_qa_agent_tool_descriptions,
            "tool_calls": []
        },
        "shopping_cart_agent": {
            "iteration": 0,
            "final_answer": False,
            "available_tools": shopping_cart_agent_tool_descriptions,
            "tool_calls": []
        },
        "coordinator_agent": {
            "iteration": 0,
            "final_answer": False,
            "plan": [],
            "next_agent": ""
        },
        "user_id": thread_id,
        "cart_id": thread_id
    }

    # thread id should come from frontend as user could have multiple conversations with the agent
    # we are using thread id as user id and cart id - just for the sake of simplicity

    config = {"configurable": {"thread_id": thread_id}}

    with PostgresSaver.from_conn_string("postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db") as checkpointer:

        graph = workflow.compile(checkpointer=checkpointer)

        for chunk in graph.stream(
            initial_state,
            config=config,
            # The "debug" stage yields detailed intermediate steps of the graph's execution,
            # while the "values" stage yields the output values computed at each step.
            # watch video 5 in week 5 - sprint 4 to understand this better. 
            # stream mode can be of type debug, updates, values, checkpoints
            stream_mode=["debug", "values"]
        ):
            processed_chunk = _process_graph_event(chunk)

            if processed_chunk:
                yield _string_for_sse(processed_chunk)

            if chunk[0] == "values":
                result = chunk[1]

    used_context = []
    dummy_vector = np.zeros(1536).tolist()

    for item in result.get("references", []):
        payload = qdrant_client.query_points(
            collection_name="Amazon-items-collection-hybrid-search",
            query=dummy_vector,
            using="text-embedding-3-small",
            limit=1,
            with_payload=True,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin", 
                        match=MatchValue(value=item.id))
                ]
            )
        ).points[0].payload
        image_url = payload.get("image")
        price = payload.get("price")
        if image_url:
            used_context.append({"image_url": image_url, "price": price, "description": item.description})

    shopping_cart = get_shopping_cart(thread_id, thread_id)
    shopping_cart_items = [
        {
            "price": item.get("price"),
            "quantity": item.get("quantity"),
            "currency": item.get("currency"),
            "product_image_url": item.get("product_image_url"),
            "total_price": item.get("total_price")
        }
        for item in shopping_cart
    ]

    yield  _string_for_sse(json.dumps(
        {
            "type": "final_result",
            "data": {
                "answer": result.get("answer"),
                "used_context": used_context,
                "trace_id": result.get("trace_id"),
                "shopping_cart_items": shopping_cart_items
            }
        },
        default=float
    ))
