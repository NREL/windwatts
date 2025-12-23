# gets the virtual environment ready for the backend
setup:
	python3 -m venv .venv
	. .venv/bin/activate; pip install -r requirements.txt
	. .venv/bin/activate; pip install -r requirements-dev.txt

run:
	. .venv/bin/activate; gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app

.PHONY: openapi
openapi:
	. .venv/bin/activate; PYTHONPATH=. SKIP_DATA_INIT=1 python scripts/generate_openapi.py

.PHONY: lint
lint:
	. .venv/bin/activate; ruff check .

.PHONY: format
format:
	. .venv/bin/activate; ruff check --fix .; ruff format .

.PHONY: test
test:
	. .venv/bin/activate; pytest
