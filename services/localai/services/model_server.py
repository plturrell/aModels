#!/usr/bin/env python3
"""
Model Serving Service

Lightweight FastAPI service that serves model files over HTTP/network.
This eliminates the need for volume mounts by serving models via network.

Features:
- List available models
- Download model files/directories
- Stream large files efficiently
- Cache model metadata
- Health endpoint
"""

from __future__ import annotations

import os
import json
import tarfile
import io
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configuration - try multiple possible paths
def find_models_directory():
    """Find the models directory by checking multiple possible locations"""
    possible_paths = [
        os.getenv("MODELS_BASE", "/home/aModels/models"),
        "/home/aModels/models",
        "/models",
        "/tmp/models",
        "/app/models"
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # Check if it has model directories
            try:
                items = os.listdir(path)
                if len(items) > 0:
                    # Check if any item looks like a model directory
                    for item in items[:5]:
                        item_path = os.path.join(path, item)
                        if os.path.isdir(item_path):
                            # Check for model files
                            model_files = os.listdir(item_path)
                            if any(f.endswith(('.json', '.safetensors', '.bin', '.pt')) for f in model_files):
                                print(f"[MODEL_SERVER] Found models at: {path}")
                                return path
            except (OSError, PermissionError):
                continue
    
    # Return the first path that exists, even if empty
    for path in possible_paths:
        if os.path.exists(path):
            print(f"[MODEL_SERVER] Using models path (may be empty): {path}")
            return path
    
    # Default fallback
    default_path = "/home/aModels/models"
    print(f"[MODEL_SERVER] Using default path: {default_path}")
    return default_path

MODELS_BASE = find_models_directory()
MODEL_SERVER_PORT = int(os.getenv("MODEL_SERVER_PORT", "8088"))

app = FastAPI(
    title="Model Serving Service",
    description="HTTP service for serving AI model files",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    path: str
    size: int
    files: List[str]
    has_config: bool
    has_tokenizer: bool


class ModelListResponse(BaseModel):
    """Response for model list"""
    models: List[ModelInfo]
    total: int
    base_path: str


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "model-server",
        "models_base": MODELS_BASE,
        "models_accessible": os.path.exists(MODELS_BASE)
    }


@app.get("/models", response_model=ModelListResponse)
async def list_models():
    """List all available models"""
    if not os.path.exists(MODELS_BASE):
        raise HTTPException(status_code=404, detail=f"Models directory not found: {MODELS_BASE}")
    
    models = []
    try:
        entries = os.listdir(MODELS_BASE)
        for entry_name in entries:
            entry_path = os.path.join(MODELS_BASE, entry_name)
            if os.path.isdir(entry_path):
                # Calculate directory size
                total_size = 0
                files = []
                has_config = False
                has_tokenizer = False
                
                for root, dirs, filenames in os.walk(entry_path):
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        try:
                            total_size += os.path.getsize(file_path)
                            rel_path = os.path.relpath(file_path, entry_path)
                            files.append(rel_path)
                            
                            if filename == "config.json":
                                has_config = True
                            if "tokenizer" in filename.lower():
                                has_tokenizer = True
                        except (OSError, IOError):
                            pass
                
                models.append(ModelInfo(
                    name=entry_name,
                    path=entry_path,
                    size=total_size,
                    files=sorted(files),
                    has_config=has_config,
                    has_tokenizer=has_tokenizer
                ))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")
    
    return ModelListResponse(
        models=sorted(models, key=lambda x: x.name),
        total=len(models),
        base_path=MODELS_BASE
    )


@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get information about a specific model"""
    model_path = os.path.join(MODELS_BASE, model_name)
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
    
    if not os.path.isdir(model_path):
        raise HTTPException(status_code=400, detail=f"Not a model directory: {model_name}")
    
    # Get model details
    files = []
    total_size = 0
    has_config = False
    has_tokenizer = False
    
    for root, dirs, filenames in os.walk(model_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            try:
                file_size = os.path.getsize(file_path)
                total_size += file_size
                rel_path = os.path.relpath(file_path, model_path)
                files.append({
                    "path": rel_path,
                    "size": file_size
                })
                
                if filename == "config.json":
                    has_config = True
                if "tokenizer" in filename.lower():
                    has_tokenizer = True
            except (OSError, IOError):
                pass
    
    return {
        "name": model_name,
        "path": model_path,
        "size": total_size,
        "files": sorted(files, key=lambda x: x["path"]),
        "has_config": has_config,
        "has_tokenizer": has_tokenizer,
        "file_count": len(files)
    }


@app.get("/models/{model_name}/download")
async def download_model(
    model_name: str,
    format: str = Query("tar", regex="^(tar|zip)$")
):
    """Download entire model as archive"""
    model_path = os.path.join(MODELS_BASE, model_name)
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
    
    if not os.path.isdir(model_path):
        raise HTTPException(status_code=400, detail=f"Not a model directory: {model_name}")
    
    if format == "tar":
        # Create tar archive in memory
        tar_buffer = io.BytesIO()
        
        with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
            tar.add(model_path, arcname=model_name)
        
        tar_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(tar_buffer.read()),
            media_type="application/gzip",
            headers={
                "Content-Disposition": f'attachment; filename="{model_name}.tar.gz"'
            }
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")


@app.get("/models/{model_name}/file/{file_path:path}")
async def get_model_file(model_name: str, file_path: str):
    """Get a specific file from a model"""
    model_path = os.path.join(MODELS_BASE, model_name)
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
    
    # Security: prevent path traversal
    if ".." in file_path or file_path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    full_path = os.path.join(model_path, file_path)
    
    # Ensure the file is within the model directory
    if not os.path.abspath(full_path).startswith(os.path.abspath(model_path)):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    if os.path.isdir(full_path):
        raise HTTPException(status_code=400, detail="Path is a directory, not a file")
    
    return FileResponse(
        full_path,
        media_type="application/octet-stream",
        filename=os.path.basename(file_path)
    )


@app.get("/models/{model_name}/sync")
async def sync_model(model_name: str, target_path: Optional[str] = None):
    """
    Sync model to a target path (for transformers-service to cache locally)
    Returns instructions for downloading model files
    """
    model_path = os.path.join(MODELS_BASE, model_name)
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
    
    # Get all files in model
    files = []
    for root, dirs, filenames in os.walk(model_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, model_path)
            file_size = os.path.getsize(file_path)
            files.append({
                "path": rel_path,
                "size": file_size,
                "url": f"/models/{model_name}/file/{rel_path}"
            })
    
    return {
        "model": model_name,
        "base_url": f"http://model-server:{MODEL_SERVER_PORT}",
        "files": sorted(files, key=lambda x: x["path"]),
        "total_size": sum(f["size"] for f in files),
        "file_count": len(files),
        "download_url": f"/models/{model_name}/download"
    }


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Model Serving Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "list_models": "/models",
            "model_info": "/models/{model_name}",
            "download_model": "/models/{model_name}/download",
            "get_file": "/models/{model_name}/file/{file_path}",
            "sync_model": "/models/{model_name}/sync"
        },
        "models_base": MODELS_BASE,
        "models_accessible": os.path.exists(MODELS_BASE)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=MODEL_SERVER_PORT)

