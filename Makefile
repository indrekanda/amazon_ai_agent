activate-venv:
	source .venv/bin/activate

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*.ipynb

run-streamlit:
	streamlit run src/chatbot-ui/streamlit_app.py