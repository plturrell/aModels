import argparse
import json
import os
import re
import sys
from pathlib import Path

from typing import Optional, Tuple, List

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_TRAINING_ROOT = REPO_ROOT / "agenticAiETH_layer4_Training"
TRAINING_ROOT = Path(os.environ.get("RELATIONAL_MODEL_ROOT", DEFAULT_TRAINING_ROOT)).expanduser()
TRAINING_ROOT = TRAINING_ROOT if TRAINING_ROOT.exists() else TRAINING_ROOT

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
except ImportError as exc:  # pragma: no cover - defensive
    print("[]")
    print(f"Failed to import relational transformer modules: {exc}", file=sys.stderr)
    sys.exit(0)

TABLE_REGEX = re.compile(r"(?i)from\s+([a-z0-9_.\"$]+)")
COLUMN_REGEX = re.compile(r"(?is)select\s+(.*?)\s+from")


def split_columns(fragment: str) -> list[str]:
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
    table_matches = TABLE_REGEX.findall(sql)
    table_name = table_matches[0] if table_matches else None

    column_match = COLUMN_REGEX.search(sql)
    if not column_match:
        return table_name, []

    column_fragment = column_match.group(1)
    columns = [col.strip() for col in split_columns(column_fragment) if col.strip()]
    # Prefer unique ordering while preserving first occurrence
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
    model = RelationalTransformer()
    state = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    return model


def generate_embedding(sql: str, model_path: Path) -> list[float]:
    table_name, columns = parse_sql(sql)
    if not table_name or not columns:
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


def main() -> None:
    default_model_path = os.environ.get(
        "SQL_EMBED_MODEL_PATH",
        str(TRAINING_ROOT / "checkpoints" / "main_schedule" / "rt_pretrain.pt"),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--sql", required=True)
    parser.add_argument("--model-path", default=default_model_path)
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser()
    if not model_path.is_file():
        print("[]")
        print(f"Model checkpoint not found at {model_path}", file=sys.stderr)
        return

    embedding = generate_embedding(args.sql, model_path)
    print(json.dumps(embedding))


if __name__ == "__main__":
    main()
