#!/usr/bin/env python3
"""
Bridge script for sap-rpt-1-oss semantic embeddings.
Provides interface to sap-rpt-1-oss ZMQ embedding server for semantic table/column embeddings.
"""
import argparse
import json
import os
import sys
from pathlib import Path

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
    print("Install with: pip install git+https://github.com/SAP-samples/sap-rpt-1-oss", file=sys.stderr)
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
    
    with _tokenizer_lock:
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

def generate_semantic_embedding(text: str, zmq_port: int = None) -> list[float]:
    """Generate semantic embedding using sap-rpt-1-oss sentence embedder."""
    if zmq_port is None:
        zmq_port = int(os.environ.get("SAP_RPT_ZMQ_PORT", ZMQ_PORT_DEFAULT))
    
    try:
        tokenizer = get_tokenizer(zmq_port)
        
        # Generate embedding
        embedding_tensor = tokenizer.texts_to_tensor([text])
        
        # Convert to list of floats
        embedding = embedding_tensor.squeeze(0).cpu().numpy().astype(float).tolist()
        
        return embedding
    except Exception as e:
        print(f"[]", file=sys.stderr)
        print(f"Failed to generate semantic embedding: {e}", file=sys.stderr)
        return []


def generate_table_semantic_embedding(table_name: str, columns: list, metadata: dict = None) -> list[float]:
    """Generate semantic embedding for a table using column names and table name."""
    zmq_port = int(os.environ.get("SAP_RPT_ZMQ_PORT", ZMQ_PORT_DEFAULT))
    
    try:
        tokenizer = get_tokenizer(zmq_port)
        
        # Prepare texts: table name + column names
        texts = [table_name]
        if isinstance(columns, list):
            for col in columns:
                if isinstance(col, dict):
                    col_name = col.get("name", "")
                    if col_name:
                        texts.append(f"{table_name}.{col_name}")
                elif isinstance(col, str):
                    texts.append(f"{table_name}.{col}")
        
        # Generate embeddings
        embeddings = tokenizer.texts_to_tensor(texts)
        
        # Mean pooling to get single table embedding
        table_embedding = embeddings.mean(dim=0).cpu().numpy().astype(float).tolist()
        
        return table_embedding
    except Exception as e:
        print(f"[]", file=sys.stderr)
        print(f"Failed to generate table semantic embedding: {e}", file=sys.stderr)
        return []


def generate_column_semantic_embedding(column_name: str, column_type: str, table_name: str = "") -> list[float]:
    """Generate semantic embedding for a column."""
    zmq_port = int(os.environ.get("SAP_RPT_ZMQ_PORT", ZMQ_PORT_DEFAULT))
    
    # Create semantic text: "table_name.column_name (type)"
    if table_name:
        text = f"{table_name}.{column_name} ({column_type})"
    else:
        text = f"{column_name} ({column_type})"
    
    return generate_semantic_embedding(text, zmq_port)


def main() -> None:
    """Main entry point for sap-rpt-1-oss embedding generation."""
    parser = argparse.ArgumentParser(description="Generate semantic embeddings using sap-rpt-1-oss")
    parser.add_argument("--artifact-type", required=True, choices=["text", "table", "column"])
    parser.add_argument("--text", help="Text to embed (for text type)")
    parser.add_argument("--table-name", help="Table name (for table/column type)")
    parser.add_argument("--columns", help="JSON array of column definitions (for table type)")
    parser.add_argument("--column-name", help="Column name (for column type)")
    parser.add_argument("--column-type", help="Column type (for column type)")
    parser.add_argument("--zmq-port", type=int, help="ZMQ port for embedding server")
    
    args = parser.parse_args()
    
    zmq_port = args.zmq_port
    if zmq_port is None:
        zmq_port = int(os.environ.get("SAP_RPT_ZMQ_PORT", ZMQ_PORT_DEFAULT))
    
    embedding: list[float] = []
    
    try:
        if args.artifact_type == "text":
            if not args.text:
                print("[]")
                print("--text required for text artifact type", file=sys.stderr)
                sys.exit(1)
            embedding = generate_semantic_embedding(args.text, zmq_port)
            
        elif args.artifact_type == "table":
            if not args.table_name:
                print("[]")
                print("--table-name required for table artifact type", file=sys.stderr)
                sys.exit(1)
            
            columns = []
            if args.columns:
                columns = json.loads(args.columns)
            
            embedding = generate_table_semantic_embedding(args.table_name, columns)
            
        elif args.artifact_type == "column":
            if not args.column_name or not args.column_type:
                print("[]")
                print("--column-name and --column-type required for column artifact type", file=sys.stderr)
                sys.exit(1)
            
            embedding = generate_column_semantic_embedding(
                args.column_name, 
                args.column_type,
                args.table_name or ""
            )
            
    except Exception as e:
        print("[]")
        print(f"Error generating semantic embedding: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not embedding:
        print("[]")
        print("Failed to generate semantic embedding", file=sys.stderr)
        sys.exit(1)
    
    print(json.dumps(embedding))


if __name__ == "__main__":
    main()

