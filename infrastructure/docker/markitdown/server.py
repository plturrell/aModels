#!/usr/bin/env python3
"""
MarkItDown HTTP Service
Exposes markitdown as an HTTP microservice for document conversion.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest, InternalServerError
import tempfile

# Add markitdown to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party" / "markitdown" / "packages" / "markitdown" / "src"))

try:
    from markitdown import MarkItDown
except ImportError:
    # Fallback: try installing markitdown
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "markitdown[all]"])
    from markitdown import MarkItDown

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MarkItDown
md = MarkItDown()

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
    - JSON with 'file_path' or 'file_content' (base64)
    - Optional 'format' parameter (default: markdown)
    
    Returns:
    - JSON with 'text_content', 'metadata', 'format'
    """
    try:
        # Handle multipart form data
        if request.content_type and "multipart/form-data" in request.content_type:
            if "file" not in request.files:
                raise BadRequest("file field is required")
            
            file = request.files["file"]
            if file.filename == "":
                raise BadRequest("file is required")
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name
            
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
            
            # Check for file_path
            if "file_path" in data:
                file_path = data["file_path"]
                if not os.path.exists(file_path):
                    raise BadRequest(f"File not found: {file_path}")
                
                result = md.convert(file_path)
                metadata = {}
                if hasattr(result, "metadata"):
                    metadata = result.metadata
                
                return jsonify({
                    "text_content": str(result),
                    "metadata": metadata,
                    "format": "markdown",
                }), 200
            
            # Check for file_content (base64)
            elif "file_content" in data:
                import base64
                file_content = base64.b64decode(data["file_content"])
                file_ext = data.get("file_extension", ".txt")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(file_content)
                    tmp_path = tmp.name
                
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
                raise BadRequest("file_path or file_content is required")
        
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
    Stream conversion for large files (not implemented yet).
    """
    return jsonify({"error": "Streaming conversion not yet implemented"}), 501

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=os.getenv("DEBUG", "false").lower() == "true")

