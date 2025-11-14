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
        
        # Get list of all models and find matching one
        # Model names on server are directory names, not registry keys
        models_list_url = f"{MODEL_SERVER_URL}/models"
        response = requests.get(models_list_url, timeout=10)
        if response.status_code != 200:
            print(f"[MODEL_CLIENT] Cannot list models from server: {response.status_code}")
            return None
        
        response_data = response.json()
        
        # Handle different response formats
        # Could be: list of dicts, list of strings, or dict with 'models' key
        if isinstance(response_data, dict):
            # If it's a dict, look for 'models' key
            if 'models' in response_data:
                all_models = response_data['models']
            else:
                # Try to use the dict itself as a single model entry
                all_models = [response_data]
        elif isinstance(response_data, list):
            all_models = response_data
        else:
            print(f"[MODEL_CLIENT] Unexpected response format: {type(response_data)}")
            return None
        
        if not all_models:
            print(f"[MODEL_CLIENT] No models found on server")
            return None
        
        # Normalize to list of dicts
        normalized_models = []
        for model in all_models:
            if isinstance(model, dict):
                normalized_models.append(model)
            elif isinstance(model, str):
                # If it's a string, treat it as the model name
                normalized_models.append({'name': model})
            else:
                print(f"[MODEL_CLIENT] Unexpected model format: {type(model)}")
                continue
        
        # Try to find matching model by name patterns
        model_name_variants = [
            model_name,  # Try as-is (e.g., "phi-3.5-mini")
            f"{model_name}-instruct-pytorch",  # Try with suffix
            f"{model_name}-transformers",  # Try with transformers suffix
            f"{model_name}-instruct",  # Try with instruct suffix
        ]
        
        model_info = None
        server_model_name = None
        
        # First try exact matches
        for model in normalized_models:
            model_dir_name = model.get('name', '')
            if model_dir_name in model_name_variants:
                model_info = model
                server_model_name = model_dir_name
                print(f"[MODEL_CLIENT] Found exact match: {server_model_name}")
                break
        
        # If no exact match, try partial matches
        if not model_info:
            for model in normalized_models:
                model_dir_name = model.get('name', '').lower()
                for variant in model_name_variants:
                    if variant.lower() in model_dir_name or model_dir_name in variant.lower():
                        model_info = model
                        server_model_name = model.get('name')
                        print(f"[MODEL_CLIENT] Found partial match: {server_model_name} (for {model_name})")
                        break
                if model_info:
                    break
        
        if not model_info:
            print(f"[MODEL_CLIENT] Model {model_name} not found on server")
            available_names = [m.get('name', str(m)) for m in normalized_models[:5]]
            print(f"[MODEL_CLIENT] Available models: {available_names}")
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
        
        # Download model using the server model name we found
        print(f"[MODEL_CLIENT] Downloading model {server_model_name} from model-server...")
        download_url = f"{MODEL_SERVER_URL}/models/{server_model_name}/download"
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
        
        # Verify extraction - check if model was extracted to cache_dir or a subdirectory
        # The tar might extract to cache_dir/model_name or just cache_dir
        possible_paths = [
            cached_model_path,
            os.path.join(cache_dir, server_model_name),
        ]
        
        extracted_path = None
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                config_file = os.path.join(path, "config.json")
                if os.path.exists(config_file):
                    extracted_path = path
                    break
        
        if extracted_path:
            print(f"[MODEL_CLIENT] ✅ Model cached successfully: {extracted_path}")
            return extracted_path
        else:
            print(f"[MODEL_CLIENT] ⚠️ Cached model missing config.json")
            # List what was extracted for debugging
            if os.path.exists(cache_dir):
                extracted_items = os.listdir(cache_dir)
                print(f"[MODEL_CLIENT] Extracted items: {extracted_items[:5]}")
        
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
    import os
    
    # First check if registry_path exists locally (direct mount)
    if os.path.exists(registry_path):
        print(f"[MODEL_CLIENT] Using direct mount path: {registry_path}")
        return registry_path
    
    # Try model-server cache
    cached_path = get_model_from_server(model_name)
    if cached_path and os.path.exists(cached_path):
        print(f"[MODEL_CLIENT] Using model-server cache: {cached_path}")
        return cached_path
    
    # Fallback to registry path even if it doesn't exist (let transformers handle the error)
    print(f"[MODEL_CLIENT] Using registry path (may not exist): {registry_path}")
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

