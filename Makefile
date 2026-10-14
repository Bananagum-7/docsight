.PHONY: install data index test api ui eval-retrieval eval-answers docker

install:
	pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
	pip install -r requirements.txt

data:
	bash scripts/fetch_docs.sh

index:
	python scripts/build_index.py

test:
	pytest -q

api:
	uvicorn docsight.api:app --reload

ui:
	streamlit run app.py

eval-retrieval:
	python eval/run_retrieval.py

eval-answers:
	python eval/run_answers.py

docker:
	docker compose up --build
