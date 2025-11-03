install:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

run:
	. .venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port $${GATEWAY_PORT:-8000}

dev:
	. .venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port $${GATEWAY_PORT:-8000}


