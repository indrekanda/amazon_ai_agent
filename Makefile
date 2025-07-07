activate-venv:
	source .venv/bin/activate

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*.ipynb