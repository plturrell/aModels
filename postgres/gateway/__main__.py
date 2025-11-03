from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("FASTAPI_HOST", "0.0.0.0")
    port = int(os.getenv("FASTAPI_PORT", "8000"))
    reload_flag = os.getenv("FASTAPI_RELOAD", "false").lower() in {"1", "true", "yes"}
    uvicorn.run("gateway.app:app", host=host, port=port, reload=reload_flag)


if __name__ == "__main__":
    main()

