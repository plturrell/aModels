import base64
import importlib.util
import os
import pathlib
from types import SimpleNamespace

import pytest


# Configure test-friendly limits before loading the server module.
os.environ.setdefault("MARKITDOWN_MAX_UPLOAD_BYTES", "1024")
os.environ.setdefault("MARKITDOWN_STREAM_CHUNK_BYTES", "256")


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "server.py"
SPEC = importlib.util.spec_from_file_location("markitdown_service_server", MODULE_PATH)
server = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(server)


@pytest.fixture
def client():
    with server.app.test_client() as test_client:
        yield test_client


@pytest.fixture
def markitdown_stub(monkeypatch):
    stub = SimpleNamespace(calls=[])

    class DummyResult:
        def __init__(self, suffix: str):
            self.metadata = {"suffix": suffix}
            self._text = f"converted{suffix}"

        def __str__(self) -> str:
            return self._text

    def fake_convert(path: str):
        suffix = pathlib.Path(path).suffix or ""
        stub.calls.append(pathlib.Path(path))
        return DummyResult(suffix)

    stub.convert = fake_convert
    monkeypatch.setattr(server, "md", stub)
    return stub


def test_json_file_path_rejected(client):
    response = client.post("/convert", json={"file_path": "/etc/passwd"})
    assert response.status_code == 400
    assert "file_content is required" in response.get_json()["error"]


def test_json_payload_size_enforced(client):
    oversized_bytes = b"a" * (server.MAX_UPLOAD_BYTES + 1)
    encoded = base64.b64encode(oversized_bytes).decode("ascii")

    response = client.post(
        "/convert",
        json={"file_content": encoded, "file_extension": ".txt"},
    )

    assert response.status_code == 400
    assert "exceeds" in response.get_json()["error"]


def test_json_conversion_uses_tempfile_cleanup(client, markitdown_stub):
    payload = base64.b64encode(b"hello world").decode("ascii")

    response = client.post(
        "/convert",
        json={"file_content": payload, "file_extension": ".txt"},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["text_content"] == "converted.txt"
    assert body["metadata"]["suffix"] == ".txt"

    assert len(markitdown_stub.calls) == 1
    converted_path = markitdown_stub.calls[0]
    assert not converted_path.exists()


def test_stream_conversion_success(client, markitdown_stub):
    response = client.post(
        "/convert-stream?file_extension=.txt",
        data=b"streaming content",
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["text_content"] == "converted.txt"
    assert body["metadata"]["suffix"] == ".txt"

    assert len(markitdown_stub.calls) == 1
    converted_path = markitdown_stub.calls[0]
    assert not converted_path.exists()


def test_stream_payload_size_enforced(client, monkeypatch):
    monkeypatch.setattr(server, "_validate_payload_size", lambda _content_length: None)
    oversized_stream = b"b" * (server.MAX_UPLOAD_BYTES + 1)

    response = client.post(
        "/convert-stream",
        data=oversized_stream,
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 400
    assert "stream payload exceeds" in response.get_json()["error"]
