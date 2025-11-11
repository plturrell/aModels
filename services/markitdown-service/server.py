#!/usr/bin/env python3
"""
MarkItDown HTTP Service
Exposes markitdown as an HTTP microservice for document conversion.
"""

import base64
import binascii
import logging
import os
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest

# Add markitdown to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party" / "markitdown" / "packages" / "markitdown" / "src"))

try:
    from markitdown import MarkItDown
except ImportError as exc:
    raise RuntimeError(
        "markitdown package is not installed. Ensure dependencies are baked into the image"
    ) from exc

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = int(os.getenv("MARKITDOWN_MAX_UPLOAD_BYTES", 25 * 1024 * 1024))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

STREAM_CHUNK_BYTES = int(os.getenv("MARKITDOWN_STREAM_CHUNK_BYTES", 1024 * 1024))

# Initialize MarkItDown
md = MarkItDown()


def _validate_payload_size(content_length: int | None) -> None:
    if content_length is not None and content_length > MAX_UPLOAD_BYTES:
        raise BadRequest("payload exceeds configured size limit")


def _decode_file_content(encoded: str, file_ext: str) -> tuple[str, str]:
    if not encoded:
        raise BadRequest("file_content must be provided")

    # Roughly validate size before decoding (base64 expands by ~4/3)
    if len(encoded) * 3 // 4 > MAX_UPLOAD_BYTES:
        raise BadRequest("encoded payload exceeds configured size limit")

    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise BadRequest("file_content is not valid base64") from exc

    if len(decoded) > MAX_UPLOAD_BYTES:
        raise BadRequest("decoded payload exceeds configured size limit")

    suffix = file_ext if file_ext.startswith(".") else f".{file_ext}" if file_ext else ".txt"
    return decoded, suffix


def _normalize_suffix(suffix: str) -> str:
    if not suffix:
        return ".bin"
    return suffix if suffix.startswith(".") else f".{suffix}"


def _stream_to_tempfile(stream, suffix: str) -> str:
    normalized_suffix = _normalize_suffix(suffix)
    total = 0
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=normalized_suffix)
    try:
        while True:
            chunk = stream.read(STREAM_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise BadRequest("stream payload exceeds configured size limit")
            tmp.write(chunk)
    except Exception:
        tmp.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise
    finally:
        tmp.close()

    if total == 0:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise BadRequest("stream payload must not be empty")

    return tmp.name


def _write_bytes_to_tempfile(data: bytes, suffix: str) -> str:
    normalized_suffix = _normalize_suffix(suffix)
    with tempfile.NamedTemporaryFile(delete=False, suffix=normalized_suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    return tmp_path

@app.route("/healthz", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "markitdown"}), 200

@app.route("/convert", methods=["POST"])
def convert():
    """
    Convert a document to markdown.
    
    Accepts:
    - multipart/form-data with 'file' field
    - JSON with 'file_content' (base64)
    - Optional 'format' parameter (default: markdown)
    
    Returns:
    - JSON with 'text_content', 'metadata', 'format'
    """
    try:
        _validate_payload_size(request.content_length)

        # Handle multipart form data
        if request.content_type and "multipart/form-data" in request.content_type:
            if "file" not in request.files:
                raise BadRequest("file field is required")
            
            file = request.files["file"]
            if file.filename == "":
                raise BadRequest("file is required")
            
            tmp_path = _stream_to_tempfile(file.stream, Path(file.filename).suffix)

            try:
                # Convert using markitdown
                result = md.convert(tmp_path)
                
                # Extract metadata
                metadata = {}
                if hasattr(result, "metadata"):
                    metadata = result.metadata
                
                return jsonify({
                    "text_content": str(result),
                    "metadata": metadata,
                    "format": "markdown",
                }), 200
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # Handle JSON request
        elif request.is_json:
            data = request.get_json()
            if not isinstance(data, dict):
                raise BadRequest("JSON payload must be an object")

            if "file_content" not in data:
                raise BadRequest("file_content is required in JSON payload")

            decoded_bytes, suffix = _decode_file_content(
                data.get("file_content", ""), data.get("file_extension", ".txt")
            )

            tmp_path = _write_bytes_to_tempfile(decoded_bytes, suffix)

            try:
                result = md.convert(tmp_path)
                metadata = {}
                if hasattr(result, "metadata"):
                    metadata = result.metadata

                return jsonify({
                    "text_content": str(result),
                    "metadata": metadata,
                    "format": "markdown",
                }), 200
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        else:
            raise BadRequest("Content-Type must be multipart/form-data or application/json")
    
    except BadRequest as e:
        logger.warning(f"Bad request: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Conversion error: {e}", exc_info=True)
        return jsonify({"error": f"Conversion failed: {str(e)}"}), 500

@app.route("/convert-stream", methods=["POST"])
def convert_stream():
    """
    Convert a streamed binary payload to markdown.

    Query parameters:
    - file_extension: optional extension to hint converter (default: .bin)
    """
    _validate_payload_size(request.content_length)

    suffix = request.args.get("file_extension", ".bin")

    stream = request.stream
    if stream is None:
        raise BadRequest("request stream is unavailable")

    tmp_path = _stream_to_tempfile(stream, suffix)

    try:
        result = md.convert(tmp_path)
        metadata = {}
        if hasattr(result, "metadata"):
            metadata = result.metadata

        return jsonify({
            "text_content": str(result),
            "metadata": metadata,
            "format": "markdown",
        }), 200
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=os.getenv("DEBUG", "false").lower() == "true")

