#!/usr/bin/env python3
"""
Batch embedding script for sap-rpt-1-oss semantic embeddings.
Processes multiple items in a single call to improve efficiency.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add sap-rpt-1-oss to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
SAP_RPT_PATH = REPO_ROOT / "models" / "sap-rpt-1-oss-main"

if SAP_RPT_PATH.exists():
    sys.path.insert(0, str(SAP_RPT_PATH))
else:
    print("[]")
    print(f"sap-rpt-1-oss not found at {SAP_RPT_PATH}", file=sys.stderr)
    sys.exit(1)

try:
    from sap_rpt_oss.data.tokenizer import Tokenizer
    from sap_rpt_oss.constants import ZMQ_PORT_DEFAULT
    from sap_rpt_oss.scripts.start_embedding_server import start_embedding_server, wait_until_done
except ImportError as exc:
    print("[]")
    print(f"Failed to import sap-rpt-1-oss modules: {exc}", file=sys.stderr)
    sys.exit(1)


# Global tokenizer instance for connection pooling
_tokenizer = None
_tokenizer_lock = None

def get_tokenizer(zmq_port: int = None) -> Tokenizer:
    """Get or create tokenizer instance (connection pooling)."""
    global _tokenizer, _tokenizer_lock
    
    if _tokenizer_lock is None:
        import threading
        _tokenizer_lock = threading.Lock()
    
    if _tokenizer is None:
        if zmq_port is None:
            zmq_port = int(os.environ.get("SAP_RPT_ZMQ_PORT", ZMQ_PORT_DEFAULT))
        
        try:
            _tokenizer = Tokenizer(zmq_port=zmq_port)
            _tokenizer.socket_init()
        except Exception as e:
            # Try to start server if not running
            try:
                start_embedding_server(Tokenizer.sentence_embedding_model_name)
                wait_until_done(timeout=30)
                _tokenizer = Tokenizer(zmq_port=zmq_port)
                _tokenizer.socket_init()
            except Exception as retry_error:
                raise RuntimeError(f"Failed to initialize tokenizer: {retry_error}")
    
    return _tokenizer


def generate_batch_table_embeddings(
    items: List[Dict[str, Any]],
    zmq_port: int = None
) -> List[Dict[str, Any]]:
    """Generate semantic embeddings for multiple tables in batch.
    
    Args:
        items: List of dicts with 'id', 'table_name', 'columns', 'metadata'
        zmq_port: Optional ZMQ port
    
    Returns:
        List of dicts with 'id', 'embedding', 'error'
    """
    if zmq_port is None:
        zmq_port = int(os.environ.get("SAP_RPT_ZMQ_PORT", ZMQ_PORT_DEFAULT))
    
    tokenizer = get_tokenizer(zmq_port)
    results = []
    
    # Prepare all texts for batch processing
    texts = []
    for item in items:
        table_name = item.get("table_name", "")
        columns = item.get("columns", [])
        
        # Create text representation: table name + column names
        item_texts = [table_name]
        if isinstance(columns, list):
            for col in columns:
                if isinstance(col, dict):
                    col_name = col.get("name", "")
                    if col_name:
                        item_texts.append(f"{table_name}.{col_name}")
                elif isinstance(col, str):
                    item_texts.append(f"{table_name}.{col}")
        
        texts.extend(item_texts)
    
    try:
        # Generate embeddings for all texts at once
        embeddings_tensor = tokenizer.texts_to_tensor(texts)
        embeddings_list = embeddings_tensor.cpu().numpy().tolist()
        
        # Map embeddings back to items
        embedding_idx = 0
        for item in items:
            table_name = item.get("table_name", "")
            columns = item.get("columns", [])
            
            # Count how many embeddings this item needs
            num_embeddings = 1  # table name
            if isinstance(columns, list):
                for col in columns:
                    if isinstance(col, dict) and col.get("name"):
                        num_embeddings += 1
                    elif isinstance(col, str):
                        num_embeddings += 1
            
            # Extract embeddings for this item
            item_embeddings = embeddings_list[embedding_idx:embedding_idx + num_embeddings]
            embedding_idx += num_embeddings
            
            # Mean pooling to get single table embedding
            import numpy as np
            if len(item_embeddings) > 1:
                table_embedding = np.mean(item_embeddings, axis=0).tolist()
            else:
                table_embedding = item_embeddings[0] if item_embeddings else []
            
            results.append({
                "id": item.get("id", ""),
                "embedding": table_embedding,
                "error": None
            })
    
    except Exception as e:
        # Return errors for all items
        for item in items:
            results.append({
                "id": item.get("id", ""),
                "embedding": [],
                "error": str(e)
            })
    
    return results


def generate_batch_column_embeddings(
    items: List[Dict[str, Any]],
    zmq_port: int = None
) -> List[Dict[str, Any]]:
    """Generate semantic embeddings for multiple columns in batch.
    
    Args:
        items: List of dicts with 'id', 'column_name', 'column_type', 'table_name'
        zmq_port: Optional ZMQ port
    
    Returns:
        List of dicts with 'id', 'embedding', 'error'
    """
    if zmq_port is None:
        zmq_port = int(os.environ.get("SAP_RPT_ZMQ_PORT", ZMQ_PORT_DEFAULT))
    
    tokenizer = get_tokenizer(zmq_port)
    results = []
    
    # Prepare all texts for batch processing
    texts = []
    for item in items:
        column_name = item.get("column_name", "")
        column_type = item.get("column_type", "")
        table_name = item.get("table_name", "")
        
        if table_name:
            text = f"{table_name}.{column_name} ({column_type})"
        else:
            text = f"{column_name} ({column_type})"
        
        texts.append(text)
    
    try:
        # Generate embeddings for all texts at once
        embeddings_tensor = tokenizer.texts_to_tensor(texts)
        embeddings_list = embeddings_tensor.cpu().numpy().tolist()
        
        # Map embeddings to results
        for i, item in enumerate(items):
            results.append({
                "id": item.get("id", ""),
                "embedding": embeddings_list[i] if i < len(embeddings_list) else [],
                "error": None
            })
    
    except Exception as e:
        # Return errors for all items
        for item in items:
            results.append({
                "id": item.get("id", ""),
                "embedding": [],
                "error": str(e)
            })
    
    return results


def main() -> None:
    """Main entry point for batch embedding generation."""
    parser = argparse.ArgumentParser(description="Generate batch semantic embeddings using sap-rpt-1-oss")
    parser.add_argument("--artifact-type", required=True, choices=["table", "column"])
    parser.add_argument("--items", required=True, help="JSON array of items to embed")
    parser.add_argument("--zmq-port", type=int, help="ZMQ port for embedding server")
    
    args = parser.parse_args()
    
    try:
        items = json.loads(args.items)
    except json.JSONDecodeError as e:
        print("[]")
        print(f"Invalid JSON for items: {e}", file=sys.stderr)
        sys.exit(1)
    
    zmq_port = args.zmq_port
    if zmq_port is None:
        zmq_port = int(os.environ.get("SAP_RPT_ZMQ_PORT", ZMQ_PORT_DEFAULT))
    
    results = []
    try:
        if args.artifact_type == "table":
            results = generate_batch_table_embeddings(items, zmq_port)
        elif args.artifact_type == "column":
            results = generate_batch_column_embeddings(items, zmq_port)
    except Exception as e:
        print("[]")
        print(f"Error generating batch embeddings: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(json.dumps(results))


if __name__ == "__main__":
    main()

