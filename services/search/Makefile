OPENAPI_FILE=python_service/openapi.json

.PHONY: openapi
openapi:
	@cd python_service && python - <<'PY'
import json
from app import app
with open("openapi.json", "w") as f:
    json.dump(app.openapi(), f, indent=2)
print("OpenAPI schema written to openapi.json")
PY
