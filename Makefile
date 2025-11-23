run-docker-compose:
	uv sync
	docker compose up --build

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*/*.ipynb

run-evals:
	uv sync
	PYTHONPATH=${PWD}/src:$$PYTHONPATH:${PWD} uv run --env-file .env python -m evals.eval_retrievers

run-evals-coordinator-agent:
	uv sync
	PYTHONPATH=${PWD}/src:$$PYTHONPATH:${PWD} uv run --env-file .env python -m evals.eval_coordinator_agent