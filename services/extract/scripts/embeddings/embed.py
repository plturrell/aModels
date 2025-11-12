#!/usr/bin/env python3
"""
Enhanced embedding generation script for ETL artifacts.
Supports SQL queries, tables, columns, Control-M jobs, process sequences, and Petri nets.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
# SCRIPT_DIR is now embeddings/, so parent.parent is scripts/, parent.parent.parent is services/extract
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_TRAINING_ROOT = REPO_ROOT / "agenticAiETH_layer4_Training"
TRAINING_ROOT = Path(os.environ.get("RELATIONAL_MODEL_ROOT", DEFAULT_TRAINING_ROOT)).expanduser()
TRAINING_ROOT = TRAINING_ROOT if TRAINING_ROOT.exists() else TRAINING_ROOT

# Try to use LocalAI for text-based embeddings if available
LOCALAI_URL = os.environ.get("LOCALAI_URL", "http://localhost:8080")
USE_LOCALAI = os.environ.get("USE_LOCALAI_FOR_EMBEDDINGS", "false").lower() == "true"

# Try to use sap-rpt-1-oss for semantic embeddings if available
USE_SAP_RPT = os.environ.get("USE_SAP_RPT_EMBEDDINGS", "false").lower() == "true"
SAP_RPT_PATH = Path(os.environ.get("SAP_RPT_PATH", REPO_ROOT / "models" / "sap-rpt-1-oss-main"))

if TRAINING_ROOT.exists():
    sys.path.append(str(TRAINING_ROOT))
else:
    print("[]")
    print(f"Training project not found at {TRAINING_ROOT}", file=sys.stderr)
    sys.exit(0)

try:
    from models.relational_transformer.data import (
        CellTokenizer,
        FrozenTextEncoder,
        RelationalDatabase,
        RelationalTableSpec,
    )
    from models.relational_transformer.model import RelationalTransformer
except ImportError as exc:
    print("[]")
    print(f"Failed to import relational transformer modules: {exc}", file=sys.stderr)
    sys.exit(0)

TABLE_REGEX = re.compile(r"(?i)from\s+([a-z0-9_.\"$]+)")
COLUMN_REGEX = re.compile(r"(?is)select\s+(.*?)\s+from")


def split_columns(fragment: str) -> list[str]:
    """Split column list from SQL SELECT clause."""
    columns: list[str] = []
    buf: list[str] = []
    depth = 0

    def flush() -> None:
        if not buf:
            return
        part = "".join(buf).strip()
        if not part:
            buf.clear()
            return
        columns.append(strip_alias(part))
        buf.clear()

    for char in fragment:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            flush()
            continue
        buf.append(char)

    flush()
    return columns


def strip_alias(fragment: str) -> str:
    """Strip alias from column expression."""
    frag = fragment.strip()
    lower = frag.lower()
    if " as " in lower:
        parts = lower.rsplit(" as ", 1)
        alias = frag[len(parts[0]) + 4 :]
        return alias.strip()

    tokens = frag.split()
    if len(tokens) >= 2:
        return tokens[-1].strip()
    return frag


def parse_sql(sql: str) -> Tuple[Optional[str], List[str]]:
    """Parse SQL to extract table name and columns."""
    table_matches = TABLE_REGEX.findall(sql)
    table_name = table_matches[0] if table_matches else None

    column_match = COLUMN_REGEX.search(sql)
    if not column_match:
        return table_name, []

    column_fragment = column_match.group(1)
    columns = [col.strip() for col in split_columns(column_fragment) if col.strip()]
    seen = set()
    unique_columns = []
    for col in columns:
        key = col.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_columns.append(col)

    return table_name, unique_columns


def load_model(model_path: Path) -> RelationalTransformer:
    """Load RelationalTransformer model from checkpoint."""
    model = RelationalTransformer()
    state = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    return model


def generate_sql_embedding(sql: str, model_path: Optional[Path] = None) -> list[float]:
    """Generate embedding for SQL query using RelationalTransformer.
    
    This function merges functionality from embed_sql.py for backward compatibility.
    If model_path is not provided, uses default from environment or training root.
    """
    table_name, columns = parse_sql(sql)
    if not table_name or not columns:
        return []

    # Use provided model_path or default
    if model_path is None:
        default_model_path = os.environ.get(
            "SQL_EMBED_MODEL_PATH",
            str(TRAINING_ROOT / "checkpoints" / "main_schedule" / "rt_pretrain.pt"),
        )
        model_path = Path(default_model_path)
    
    if not model_path.is_file():
        print(f"Model checkpoint not found at {model_path}", file=sys.stderr)
        return []

    data = {col: [1.0] for col in columns}
    df = pd.DataFrame(data)

    table_spec = RelationalTableSpec(
        name=table_name,
        dataframe=df,
        primary_key=columns[0],
    )
    db = RelationalDatabase([table_spec])

    text_encoder = FrozenTextEncoder()
    tokenizer = CellTokenizer(db, text_encoder)

    handle = db.resolve_row(table_name, 1.0)
    tokens = tokenizer.tokenize_row(handle, row_id=0)

    value_embeddings = torch.stack([t.value_embedding for t in tokens]).unsqueeze(0)
    schema_embeddings = torch.stack([t.schema_embedding for t in tokens]).unsqueeze(0)
    temporal_embeddings = torch.stack([t.temporal_embedding for t in tokens]).unsqueeze(0)
    role_embeddings = torch.stack([t.role_embedding for t in tokens]).unsqueeze(0)
    dtype_ids = torch.tensor([t.dtype_id for t in tokens], dtype=torch.long).unsqueeze(0)

    masks = {
        "column": torch.ones(len(tokens), len(tokens), dtype=torch.bool),
        "feature": torch.ones(len(tokens), len(tokens), dtype=torch.bool),
        "neighbor": torch.ones(len(tokens), len(tokens), dtype=torch.bool),
        "full": torch.ones(len(tokens), len(tokens), dtype=torch.bool),
    }

    model = load_model(model_path)
    with torch.no_grad():
        token_embeddings = model(
            value_embeddings=value_embeddings,
            schema_embeddings=schema_embeddings,
            temporal_embeddings=temporal_embeddings,
            role_embeddings=role_embeddings,
            dtype_ids=dtype_ids,
            masks=masks,
        )

    query_embedding = token_embeddings.mean(dim=1).squeeze(0)
    return query_embedding.tolist()


def generate_table_embedding(table_name: str, columns: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> list[float]:
    """Generate embedding for table schema using RelationalTransformer or sap-rpt-1-oss."""
    # Try sap-rpt-1-oss semantic embedding first if enabled
    if USE_SAP_RPT and SAP_RPT_PATH.exists():
        try:
            sys.path.insert(0, str(SAP_RPT_PATH))
            from sap_rpt_oss.data.tokenizer import Tokenizer
            from sap_rpt_oss.constants import ZMQ_PORT_DEFAULT
            from sap_rpt_oss.scripts.start_embedding_server import start_embedding_server, wait_until_done
            
            zmq_port = int(os.environ.get("SAP_RPT_ZMQ_PORT", ZMQ_PORT_DEFAULT))
            tokenizer = Tokenizer(zmq_port=zmq_port)
            try:
                tokenizer.socket_init()
            except:
                # Start server if not running
                start_embedding_server(Tokenizer.sentence_embedding_model_name)
                wait_until_done(timeout=30)
                tokenizer.socket_init()
            
            # Prepare texts
            texts = [table_name]
            for col in columns:
                col_name = col.get("name", "")
                if col_name:
                    texts.append(f"{table_name}.{col_name}")
            
            # Generate embeddings
            embeddings = tokenizer.texts_to_tensor(texts)
            table_embedding = embeddings.mean(dim=0).cpu().numpy().astype(float).tolist()
            
            # Convert to float32 for consistency
            return [float(x) for x in table_embedding]
        except Exception as e:
            # Fallback to RelationalTransformer
            pass
    
    if not columns:
        return []

    # Create DataFrame with column types as dummy data
    column_data = {}
    for col in columns:
        col_name = col.get("name", "")
        col_type = col.get("type", "string")
        # Use type as placeholder value
        column_data[col_name] = [1.0]

    df = pd.DataFrame(column_data)

    table_spec = RelationalTableSpec(
        name=table_name,
        dataframe=df,
        primary_key=columns[0].get("name", "") if columns else "",
    )
    db = RelationalDatabase([table_spec])

    text_encoder = FrozenTextEncoder()
    tokenizer = CellTokenizer(db, text_encoder)

    handle = db.resolve_row(table_name, 1.0)
    tokens = tokenizer.tokenize_row(handle, row_id=0)

    value_embeddings = torch.stack([t.value_embedding for t in tokens]).unsqueeze(0)
    schema_embeddings = torch.stack([t.schema_embedding for t in tokens]).unsqueeze(0)
    temporal_embeddings = torch.stack([t.temporal_embedding for t in tokens]).unsqueeze(0)
    role_embeddings = torch.stack([t.role_embedding for t in tokens]).unsqueeze(0)
    dtype_ids = torch.tensor([t.dtype_id for t in tokens], dtype=torch.long).unsqueeze(0)

    masks = {
        "column": torch.ones(len(tokens), len(tokens), dtype=torch.bool),
        "feature": torch.ones(len(tokens), len(tokens), dtype=torch.bool),
        "neighbor": torch.ones(len(tokens), len(tokens), dtype=torch.bool),
        "full": torch.ones(len(tokens), len(tokens), dtype=torch.bool),
    }

    default_model_path = os.environ.get(
        "SQL_EMBED_MODEL_PATH",
        str(TRAINING_ROOT / "checkpoints" / "main_schedule" / "rt_pretrain.pt"),
    )
    model = load_model(Path(default_model_path))
    with torch.no_grad():
        token_embeddings = model(
            value_embeddings=value_embeddings,
            schema_embeddings=schema_embeddings,
            temporal_embeddings=temporal_embeddings,
            role_embeddings=role_embeddings,
            dtype_ids=dtype_ids,
            masks=masks,
        )

    query_embedding = token_embeddings.mean(dim=1).squeeze(0)
    return query_embedding.tolist()


def generate_column_embedding(column_name: str, column_type: str, constraints: Optional[Dict[str, Any]] = None) -> list[float]:
    """Generate embedding for column definition using RelationalTransformer or sap-rpt-1-oss."""
    # Try sap-rpt-1-oss semantic embedding first if enabled
    if USE_SAP_RPT and SAP_RPT_PATH.exists():
        try:
            sys.path.insert(0, str(SAP_RPT_PATH))
            from sap_rpt_oss.data.tokenizer import Tokenizer
            from sap_rpt_oss.constants import ZMQ_PORT_DEFAULT
            from sap_rpt_oss.scripts.start_embedding_server import start_embedding_server, wait_until_done
            
            zmq_port = int(os.environ.get("SAP_RPT_ZMQ_PORT", ZMQ_PORT_DEFAULT))
            tokenizer = Tokenizer(zmq_port=zmq_port)
            try:
                tokenizer.socket_init()
            except:
                start_embedding_server(Tokenizer.sentence_embedding_model_name)
                wait_until_done(timeout=30)
                tokenizer.socket_init()
            
            # Generate semantic embedding for column
            text = f"{column_name} ({column_type})"
            embedding_tensor = tokenizer.texts_to_tensor([text])
            embedding = embedding_tensor.squeeze(0).cpu().numpy().astype(float).tolist()
            
            return [float(x) for x in embedding]
        except Exception as e:
            # Fallback to RelationalTransformer
            pass
    
    # Create minimal table with single column
    data = {column_name: [1.0]}
    df = pd.DataFrame(data)

    table_spec = RelationalTableSpec(
        name="single_column_table",
        dataframe=df,
        primary_key=column_name,
    )
    db = RelationalDatabase([table_spec])

    text_encoder = FrozenTextEncoder()
    tokenizer = CellTokenizer(db, text_encoder)

    handle = db.resolve_row("single_column_table", 1.0)
    tokens = tokenizer.tokenize_row(handle, row_id=0)

    value_embeddings = torch.stack([t.value_embedding for t in tokens]).unsqueeze(0)
    schema_embeddings = torch.stack([t.schema_embedding for t in tokens]).unsqueeze(0)
    temporal_embeddings = torch.stack([t.temporal_embedding for t in tokens]).unsqueeze(0)
    role_embeddings = torch.stack([t.role_embedding for t in tokens]).unsqueeze(0)
    dtype_ids = torch.tensor([t.dtype_id for t in tokens], dtype=torch.long).unsqueeze(0)

    masks = {
        "column": torch.ones(len(tokens), len(tokens), dtype=torch.bool),
        "feature": torch.ones(len(tokens), len(tokens), dtype=torch.bool),
        "neighbor": torch.ones(len(tokens), len(tokens), dtype=torch.bool),
        "full": torch.ones(len(tokens), len(tokens), dtype=torch.bool),
    }

    default_model_path = os.environ.get(
        "SQL_EMBED_MODEL_PATH",
        str(TRAINING_ROOT / "checkpoints" / "main_schedule" / "rt_pretrain.pt"),
    )
    model = load_model(Path(default_model_path))
    with torch.no_grad():
        token_embeddings = model(
            value_embeddings=value_embeddings,
            schema_embeddings=schema_embeddings,
            temporal_embeddings=temporal_embeddings,
            role_embeddings=role_embeddings,
            dtype_ids=dtype_ids,
            masks=masks,
        )

    query_embedding = token_embeddings.mean(dim=1).squeeze(0)
    return query_embedding.tolist()


def generate_text_embedding_via_localai(text: str) -> list[float]:
    """Generate embedding for text using LocalAI."""
    try:
        import requests
        response = requests.post(
            f"{LOCALAI_URL}/v1/embeddings",
            json={
                "model": "text-embedding-ada-002",  # Default model
                "input": text,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            return data["data"][0]["embedding"]
    except Exception as e:
        print(f"LocalAI embedding failed: {e}", file=sys.stderr)
    return []


def generate_job_embedding(job_name: str, command: str, conditions: List[str], metadata: Optional[Dict[str, Any]] = None) -> list[float]:
    """Generate embedding for Control-M job using text-based embedding."""
    # Combine job information into text
    text_parts = [f"Job: {job_name}"]
    if command:
        text_parts.append(f"Command: {command}")
    if conditions:
        text_parts.append(f"Conditions: {', '.join(conditions)}")
    if metadata:
        for key, value in metadata.items():
            text_parts.append(f"{key}: {value}")
    
    text = " ".join(text_parts)
    
    if USE_LOCALAI:
        return generate_text_embedding_via_localai(text)
    
    # Fallback: use RelationalTransformer with dummy data
    # This is less ideal but works if LocalAI is unavailable
    return generate_sql_embedding(f"SELECT * FROM {job_name}", Path(os.environ.get(
        "SQL_EMBED_MODEL_PATH",
        str(TRAINING_ROOT / "checkpoints" / "main_schedule" / "rt_pretrain.pt"),
    )))


def generate_sequence_embedding(sequence_id: str, tables: List[str], order: int, metadata: Optional[Dict[str, Any]] = None) -> list[float]:
    """Generate embedding for process sequence."""
    # Combine sequence information into text
    text_parts = [f"Sequence: {sequence_id}"]
    text_parts.append(f"Tables: {', '.join(tables)}")
    text_parts.append(f"Order: {order}")
    if metadata:
        for key, value in metadata.items():
            text_parts.append(f"{key}: {value}")
    
    text = " ".join(text_parts)
    
    if USE_LOCALAI:
        return generate_text_embedding_via_localai(text)
    
    # Fallback: use table embedding approach
    columns = [{"name": table, "type": "string"} for table in tables]
    return generate_table_embedding(sequence_id, columns, metadata)


def generate_petri_net_embedding(petri_net: Dict[str, Any]) -> list[float]:
    """Generate embedding for Petri net workflow."""
    # Combine Petri net information into text
    text_parts = [f"Petri Net: {petri_net.get('name', 'unnamed')}"]
    
    places = petri_net.get("places", [])
    transitions = petri_net.get("transitions", [])
    arcs = petri_net.get("arcs", [])
    
    text_parts.append(f"Places: {len(places)}")
    text_parts.append(f"Transitions: {len(transitions)}")
    text_parts.append(f"Arcs: {len(arcs)}")
    
    # Add transition labels
    if transitions:
        transition_labels = [t.get("label", "") for t in transitions[:10]]  # Limit to first 10
        text_parts.append(f"Transitions: {', '.join(transition_labels)}")
    
    text = " ".join(text_parts)
    
    if USE_LOCALAI:
        return generate_text_embedding_via_localai(text)
    
    # Fallback: use sequence embedding approach
    transition_names = [t.get("label", "") for t in transitions]
    return generate_sequence_embedding(petri_net.get("id", "petri_net"), transition_names, 0, petri_net.get("metadata"))


def main() -> None:
    """Main entry point for embedding generation."""
    default_model_path = os.environ.get(
        "SQL_EMBED_MODEL_PATH",
        str(TRAINING_ROOT / "checkpoints" / "main_schedule" / "rt_pretrain.pt"),
    )

    parser = argparse.ArgumentParser(description="Generate embeddings for ETL artifacts")
    parser.add_argument("--artifact-type", required=True, choices=["sql", "table", "column", "job", "sequence", "petri_net"])
    parser.add_argument("--sql", help="SQL query (for sql type)")
    parser.add_argument("--table-name", help="Table name (for table/column type)")
    parser.add_argument("--columns", help="JSON array of column definitions (for table type)")
    parser.add_argument("--column-name", help="Column name (for column type)")
    parser.add_argument("--column-type", help="Column type (for column type)")
    parser.add_argument("--job-name", help="Job name (for job type)")
    parser.add_argument("--command", help="Command (for job type)")
    parser.add_argument("--conditions", help="JSON array of conditions (for job type)")
    parser.add_argument("--sequence-id", help="Sequence ID (for sequence type)")
    parser.add_argument("--tables", help="JSON array of table names (for sequence type)")
    parser.add_argument("--order", type=int, help="Order (for sequence type)")
    parser.add_argument("--petri-net", help="JSON object of Petri net (for petri_net type)")
    parser.add_argument("--metadata", help="JSON object of metadata")
    parser.add_argument("--model-path", default=default_model_path)
    
    args = parser.parse_args()

    embedding: list[float] = []
    metadata_obj = json.loads(args.metadata) if args.metadata else None

    try:
        if args.artifact_type == "sql":
            if not args.sql:
                print("[]")
                print("--sql required for sql artifact type", file=sys.stderr)
                sys.exit(1)
            model_path = Path(args.model_path).expanduser() if args.model_path else None
            embedding = generate_sql_embedding(args.sql, model_path)
            
        elif args.artifact_type == "table":
            if not args.table_name or not args.columns:
                print("[]")
                print("--table-name and --columns required for table artifact type", file=sys.stderr)
                sys.exit(1)
            columns = json.loads(args.columns)
            embedding = generate_table_embedding(args.table_name, columns, metadata_obj)
            
        elif args.artifact_type == "column":
            if not args.column_name or not args.column_type:
                print("[]")
                print("--column-name and --column-type required for column artifact type", file=sys.stderr)
                sys.exit(1)
            constraints = json.loads(args.metadata) if args.metadata else None
            embedding = generate_column_embedding(args.column_name, args.column_type, constraints)
            
        elif args.artifact_type == "job":
            if not args.job_name:
                print("[]")
                print("--job-name required for job artifact type", file=sys.stderr)
                sys.exit(1)
            conditions = json.loads(args.conditions) if args.conditions else []
            embedding = generate_job_embedding(args.job_name, args.command or "", conditions, metadata_obj)
            
        elif args.artifact_type == "sequence":
            if not args.sequence_id or not args.tables:
                print("[]")
                print("--sequence-id and --tables required for sequence artifact type", file=sys.stderr)
                sys.exit(1)
            tables = json.loads(args.tables)
            embedding = generate_sequence_embedding(args.sequence_id, tables, args.order or 0, metadata_obj)
            
        elif args.artifact_type == "petri_net":
            if not args.petri_net:
                print("[]")
                print("--petri-net required for petri_net artifact type", file=sys.stderr)
                sys.exit(1)
            petri_net = json.loads(args.petri_net)
            embedding = generate_petri_net_embedding(petri_net)
            
    except Exception as e:
        print("[]")
        print(f"Error generating embedding: {e}", file=sys.stderr)
        sys.exit(1)

    if not embedding:
        print("[]")
        print("Failed to generate embedding", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(embedding))


if __name__ == "__main__":
    main()

