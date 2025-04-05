install:
	python3 -m venv .venv && pip3 install -r requirements.txt

activate:
	echo "source .venv/bin/activate" && echo ". Please copy the line to active the env"

install:
	pip3 install -r requirements.txt

app-run:
	streamlit run app.py

run:
	python3 main.py

