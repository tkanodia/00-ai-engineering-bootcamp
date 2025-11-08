from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from api.api.agent.models import State
from api.api.agent.tools import get_formatted_item_context, get_formatted_reviews_context
from api.api.agent.agents import agent_node, intent_router_node
from api.api.agent.utils.utils import get_tool_descriptions
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langgraph.checkpoint.postgres import PostgresSaver

import numpy as np

### Routers for the workflow
def tool_router(state: State) -> str:
    """Decide whether to continue or end"""
    
    if state.final_answer:
        return "end"
    elif state.iteration > 2:
        return "end"
    elif len(state.tool_calls) > 0:
        return "tools"
    else:
        return "end"

def intent_router_conditional_edges(state: State):

    if state.question_relevant:
        return "agent_node"
    else:
        return "end"

### Workflow


workflow = StateGraph(State)

tools = [get_formatted_item_context, get_formatted_reviews_context]
tool_node = ToolNode(tools)
tool_descriptions = get_tool_descriptions(tools)

workflow.add_node("agent_node", agent_node)
workflow.add_node("tool_node", tool_node)
workflow.add_node("intent_router_node", intent_router_node)

workflow.add_edge(START, "intent_router_node")

workflow.add_conditional_edges(
    "intent_router_node",
    intent_router_conditional_edges,
    {
        "agent_node": "agent_node",
        "end": END
    }
)

workflow.add_conditional_edges(
    "agent_node",
    tool_router,
    {
        "tools": "tool_node",
        "end": END
    }
)

# Why do we need this edge?
# This edge routes the flow from the 'tool_node' (where the tool output is computed) back to the 'agent_node',
# enabling the agent to use the results of tool calls to produce a final answer or potentially call more tools.
# It's essential in agent-tooling workflows where the agent needs to reason over new information or iterate on next actions.
workflow.add_edge("tool_node", "agent_node")

## Agent execution function to invoke the graph


def run_agent(query: str, thread_id: str):
    initial_state = {
        "messages": [{"role": "user", "content": query}],
        "iteration": 0,
        "available_tools": tool_descriptions
    }

    # thread id should come from frontend as user could have multiple conversations with the agent

    config = {"configurable": {"thread_id": thread_id}}
    with PostgresSaver.from_conn_string("postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db") as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
        result = graph.invoke(initial_state, config=config)

    return result



def run_agent_wrapper(question: str, thread_id: str):
    qdrant_client = QdrantClient(url="http://qdrant:6333")
    result = run_agent(question, thread_id)

    used_context = []
    dummy_vector = np.zeros(1536).tolist()

    for item in result.get("references", []):
        payload = qdrant_client.query_points(
            collection_name="Amazon-items-collection-hybrid-search",
            query=dummy_vector,
            using="text-embedding-3-small",
            limit=1,
            with_payload=True,
            with_vectors=False,
            query_filter=Filter(
                must=[
                    FieldCondition(key="parent_asin", match=MatchValue(value=item.id))
                ]
            )
        ).points[0].payload
        image_url = payload.get("image", "")
        price = payload.get("price", "")
        if image_url not in [None, ""]:
            used_context.append({
                "image_url": image_url,
                "price": price,
                "description": item.description
            })
    return {
        "answer": result.get("answer"),
        "used_context": used_context
    }