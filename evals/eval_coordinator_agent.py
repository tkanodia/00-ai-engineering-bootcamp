from langsmith import Client

from api.agent.agents import coordinator_agent
from api.agent.graph import State
from time import sleep

SLEEP_TIME = 5
ACCURACY_THRESHOLD = 0.6

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


print(f"Sleeping for {SLEEP_TIME} seconds to allow the evaluation to complete")
sleep(SLEEP_TIME)

results_resp = client.read_project(
    project_name = results.experiment_name,
    include_stats = True
)

avg_metric = results_resp.feedback_stats.get("next_agent_evaluator").get("avg")
errors = results_resp.feedback_stats.get("next_agent_evaluator").get("errors")

if avg_metric >= ACCURACY_THRESHOLD:
    output_message = f"✅ :Evaluator passed with accuracy {avg_metric}"
else:
    output_message = f"❌ :Evaluator failed with accuracy {avg_metric}"

if errors > 0:
    raise AssertionError(f"Evaluator failed with {errors} errors")
elif avg_metric >= ACCURACY_THRESHOLD:
    print(output_message, flush=True)
else:
    raise AssertionError(output_message)