# alias act="source .venv/bin/activate"
# alias run-tests="pytest tests"

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*.ipynb

run-streamlit:
	streamlit run src/chatbot_ui/streamlit_app.py

build-docker-streamlit:
	docker build -t streamlit-app:latest .

run-docker-streamlit:
	docker run -v ${PWD}/.env:/app/.env -p 8501:8501 streamlit-app:latest

# -- build - build the app each time we run it
# -d - launche and it runs as a detached process in background (no need to keep terminal open, no logs in the terminal, harder to stop)
run-docker-compose:
	uv sync
	docker compose up --build

run-evals:
	uv sync
	PYTHONPATH=${PWD}/src:$$PYTHONPATH:${PWD} uv run --env-file .env python -m evals.eval_retriever
