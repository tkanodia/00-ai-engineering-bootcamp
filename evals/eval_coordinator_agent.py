from langsmith import Client

from api.agent.agents import coordinator_agent
from api.agent.graph import State

client = Client()


def next_agent_evaluator(run, example):

    next_agent_match = run.outputs["coordinator_agent"]["next_agent"] == example.outputs["next_agent"] 
    final_answer_match = run.outputs["coordinator_agent"]["final_answer"] == example.outputs["coordinator_final_answer"]
    
    return next_agent_match and final_answer_match



results = client.evaluate(
    lambda x: coordinator_agent(State(messages=x["messages"])),
    data="coordinator-eval-dataset",
    evaluators=[
        next_agent_evaluator
    ],
    experiment_prefix="coordinator-eval-dataset"
)