"""
Model Client - Fetches models from model-server with local caching
"""

import os
import shutil
import tarfile
import requests
from pathlib import Path
from typing import Optional, Dict
import io

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://model-server:8088")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/models-cache")
CACHE_ENABLED = os.getenv("MODEL_CACHE_ENABLED", "true").lower() == "true"


def ensure_cache_dir():
    """Ensure cache directory exists"""
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    return MODEL_CACHE_DIR


def get_model_from_server(model_name: str, target_path: Optional[str] = None) -> Optional[str]:
    """
    Fetch model from model-server and cache it locally.
    Returns the local path to the cached model, or None if failed.
    """
    if not CACHE_ENABLED:
        return None
    
    try:
        # Check if model-server is available
        health_url = f"{MODEL_SERVER_URL}/health"
        response = requests.get(health_url, timeout=5)
        if response.status_code != 200:
            print(f"[MODEL_CLIENT] Model-server not available: {response.status_code}")
            return None
        
        # Check if model exists on server
        model_info_url = f"{MODEL_SERVER_URL}/models/{model_name}"
        response = requests.get(model_info_url, timeout=10)
        if response.status_code != 200:
            print(f"[MODEL_CLIENT] Model {model_name} not found on server")
            return None
        
        # Determine cache path
        cache_dir = ensure_cache_dir()
        cached_model_path = os.path.join(cache_dir, model_name)
        
        # Check if already cached
        if os.path.exists(cached_model_path) and os.path.isdir(cached_model_path):
            # Verify cache is valid (has config.json or similar)
            config_file = os.path.join(cached_model_path, "config.json")
            if os.path.exists(config_file):
                print(f"[MODEL_CLIENT] Using cached model: {cached_model_path}")
                return cached_model_path
        
        # Download model
        print(f"[MODEL_CLIENT] Downloading model {model_name} from model-server...")
        download_url = f"{MODEL_SERVER_URL}/models/{model_name}/download"
        response = requests.get(download_url, timeout=300, stream=True)  # 5 min timeout for large models
        
        if response.status_code != 200:
            print(f"[MODEL_CLIENT] Failed to download model: {response.status_code}")
            return None
        
        # Extract to cache
        print(f"[MODEL_CLIENT] Extracting model to cache...")
        os.makedirs(cached_model_path, exist_ok=True)
        
        tar_buffer = io.BytesIO(response.content)
        with tarfile.open(fileobj=tar_buffer, mode='r:gz') as tar:
            tar.extractall(cache_dir)
        
        # Verify extraction
        if os.path.exists(cached_model_path) and os.path.isdir(cached_model_path):
            config_file = os.path.join(cached_model_path, "config.json")
            if os.path.exists(config_file):
                print(f"[MODEL_CLIENT] ✅ Model cached successfully: {cached_model_path}")
                return cached_model_path
            else:
                print(f"[MODEL_CLIENT] ⚠️ Cached model missing config.json")
        
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"[MODEL_CLIENT] Network error fetching model: {e}")
        return None
    except Exception as e:
        print(f"[MODEL_CLIENT] Error fetching model: {e}")
        return None


def get_model_path(model_name: str, registry_path: str) -> str:
    """
    Get model path, trying model-server first, then falling back to registry path.
    Returns the path to use for loading the model.
    """
    # Try model-server first
    cached_path = get_model_from_server(model_name)
    if cached_path:
        return cached_path
    
    # Fallback to registry path (direct mount or local)
    print(f"[MODEL_CLIENT] Using registry path: {registry_path}")
    return registry_path


def sync_model_files(model_name: str, base_url: str, files: list, target_dir: str):
    """
    Sync individual model files from model-server.
    More efficient than downloading entire archive for small updates.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    for file_info in files:
        file_path = file_info.get("path", "")
        file_url = file_info.get("url", "")
        
        if not file_path or not file_url:
            continue
        
        # Construct full URL
        if not file_url.startswith("http"):
            file_url = f"{base_url}{file_url}"
        
        # Create directory structure
        full_path = os.path.join(target_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Download file
        try:
            response = requests.get(file_url, timeout=60, stream=True)
            if response.status_code == 200:
                with open(full_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            print(f"[MODEL_CLIENT] Error downloading {file_path}: {e}")

