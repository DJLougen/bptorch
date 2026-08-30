.PHONY: all setup dev server web test test-backend test-frontend parity lint build contracts clean examples train-samples infer-samples

PYTHON ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest
RUFF ?= .venv/bin/ruff
MYPY ?= .venv/bin/mypy
UVICORN ?= .venv/bin/uvicorn
NPM ?= npm

all: test lint

setup:
	uv venv .venv --python 3.12 || python3 -m venv .venv
	$(PYTHON) -m pip install -e ".[dev]" || $(PYTHON) -m pip install torch fastapi pydantic uvicorn websockets httpx pytest hypothesis ruff mypy
	cd web && $(NPM) install

dev:
	@echo "Starting bpTorch backend (:8000) and frontend (:5173)..."
	@trap 'kill 0' EXIT; \
	(PYTHONPATH=server $(UVICORN) neural_blueprint.api.main:app --host 127.0.0.1 --port 8000 --reload) & \
	(cd web && $(NPM) run dev) & \
	wait

server:
	PYTHONPATH=server $(UVICORN) neural_blueprint.api.main:app --host 127.0.0.1 --port 8000 --reload

web:
	cd web && $(NPM) run dev

test: test-backend test-frontend

test-backend:
	PYTHONPATH=server $(PYTEST) server/tests -v

test-frontend:
	cd web && $(NPM) run test:run

parity:
	PYTHONPATH=server $(PYTEST) server/tests/parity -v

lint:
	PYTHONPATH=server $(RUFF) check server
	cd web && $(NPM) run typecheck

examples:
	PYTHONPATH=server $(PYTHON) scripts/save_examples.py

train-samples:
	PYTHONPATH=server $(PYTHON) scripts/train_all_samples.py

infer-samples:
	PYTHONPATH=server $(PYTHON) scripts/infer_all_samples.py

build: contracts examples
	cd web && $(NPM) run build

contracts:
	PYTHONPATH=server $(PYTHON) scripts/generate_contracts.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist web/dist server/__pycache__ server/*/__pycache__
