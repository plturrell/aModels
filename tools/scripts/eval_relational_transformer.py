#!/usr/bin/env python3
"""Evaluate Relational Transformer checkpoints on binary and regression tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.relational_transformer import (  # noqa: E402
    CellTokenizer,
    ContextSampler,
    ForeignKeySpec,
    RelationalInferenceConfig,
    RelationalInferenceEngine,
    RelationalDatabase,
    RelationalDataset,
    RelationalTableSpec,
    RelationalTransformer,
    TargetSpec,
)
from models.relational_transformer.data import RELATIONAL_ID_TO_DTYPE, FrozenTextEncoder  # noqa: E402


def build_database(config: Dict[str, Any]) -> RelationalDatabase:
    tables: List[RelationalTableSpec] = []
    for table_cfg in config.get("tables", []):
        fmt = table_cfg.get("format", "csv").lower()
        path = Path(table_cfg["path"]).expanduser()
        if fmt == "csv":
            dataframe = pd.read_csv(path)
        elif fmt == "parquet":
            dataframe = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported table format {fmt}")
        fks = [
            ForeignKeySpec(
                parent_table=fk["parent_table"],
                parent_column=fk["parent_column"],
                child_column=fk["child_column"],
            )
            for fk in table_cfg.get("foreign_keys", [])
        ]
        tables.append(
            RelationalTableSpec(
                name=table_cfg["name"],
                dataframe=dataframe,
                primary_key=table_cfg["primary_key"],
                timestamp_column=table_cfg.get("timestamp_column"),
                foreign_keys=fks,
            )
        )
    return RelationalDatabase(tables)


def build_targets(database: RelationalDatabase, eval_cfg: Dict[str, Any]) -> List[TargetSpec]:
    targets: List[TargetSpec] = []
    for target_cfg in eval_cfg.get("targets", []):
        table = target_cfg["table"]
        column = target_cfg["column"]
        pk_column = target_cfg.get("primary_key", database.primary_key(table))
        dataframe = database.dataframe(table)
        rows = target_cfg.get("rows")
        limit = target_cfg.get("limit")
        if rows is None:
            rows = dataframe[pk_column].tolist()
            if limit:
                rows = rows[:limit]
        for pk_value in rows:
            targets.append(TargetSpec(table=table, primary_key_value=pk_value, column=column))
    return targets


def compute_metrics(predictions: List[float], targets: List[float], task_type: str) -> Dict[str, float]:
    preds = np.asarray(predictions)
    gold = np.asarray(targets)
    if task_type == "binary":
        return {
            "auroc": float(roc_auc_score(gold, preds)) if len(np.unique(gold)) > 1 else float("nan"),
        }
    if task_type == "regression":
        ss_res = float(np.sum((gold - preds) ** 2))
        ss_tot = float(np.sum((gold - np.mean(gold)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return {"r2": r2}
    raise ValueError(f"Unknown task type {task_type}")


def run_zero_shot(
    inferencer: RelationalInferenceEngine,
    dataset: RelationalDataset,
    eval_cfg: Dict[str, Any],
    task_type: str,
) -> Dict[str, float]:
    predictions: List[float] = []
    targets: List[float] = []
    limit = eval_cfg.get("limit") or len(dataset)
    for idx in range(min(limit, len(dataset))):
        sample = dataset[idx]
        result = inferencer.infer_sample(sample)
        dtype_ids = sample["dtype_ids"].numpy()
        loss_mask = sample["loss_mask"].numpy()
        target_values = sample["target_values"].numpy()
        for pos, mask in enumerate(loss_mask):
            if not mask:
                continue
            dtype_id = dtype_ids[pos]
            dtype_name = RELATIONAL_ID_TO_DTYPE.get(int(dtype_id), "numeric")
            prediction = result.get(dtype_name)
            if prediction is None:
                continue
            predictions.append(float(prediction.squeeze()[pos].item()))
            targets.append(float(target_values[pos]))
    return compute_metrics(predictions, targets, task_type)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Base YAML configuration")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate")
    parser.add_argument("--eval-config", required=True, help="Evaluation YAML specifying targets and task type")
    parser.add_argument("--output", required=True, help="Where to write metrics JSON")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with open(args.eval_config, "r", encoding="utf-8") as handle:
        eval_cfg = yaml.safe_load(handle)

    database = build_database(config)
    context_cfg = config.get("context", {})
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    sampler = ContextSampler(
        database,
        max_context_cells=context_cfg.get("max_cells", 1024),
        width_bound=context_cfg.get("width_bound", 8),
        random_state=context_cfg.get("random_seed"),
    )
    text_encoder = None
    text_encoder_name = model_cfg.get("text_encoder")
    if text_encoder_name:
        text_encoder = FrozenTextEncoder(model_name=text_encoder_name, device=training_cfg.get("device"))
    tokenizer = CellTokenizer(
        database,
        text_encoder=text_encoder,
        value_dim=model_cfg.get("value_dim", 384),
        schema_dim=model_cfg.get("schema_dim", 384),
        temporal_dim=model_cfg.get("temporal_dim", 5),
        role_dim=model_cfg.get("role_dim", 64),
        schema_seed=context_cfg.get("schema_seed", 0),
    )

    targets = build_targets(database, eval_cfg)
    temporal_lookback_seconds = context_cfg.get("temporal_lookback_seconds")
    if temporal_lookback_seconds is None and context_cfg.get("temporal_lookback_hours") is not None:
        temporal_lookback_seconds = float(context_cfg["temporal_lookback_hours"]) * 3600.0

    dataset = RelationalDataset(
        database=database,
        sampler=sampler,
        tokenizer=tokenizer,
        targets=targets,
        context_cells=context_cfg.get("max_cells", 1024),
        mask_probability=0.0,
        include_text=context_cfg.get("include_text", False),
        allow_temporal_leakage=context_cfg.get("allow_temporal_leakage", False),
        temporal_lookback_seconds=temporal_lookback_seconds,
    )

    model = RelationalTransformer(
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_layers=model_cfg.get("num_layers", 12),
        num_heads=model_cfg.get("num_heads", 8),
        mlp_hidden_dim=model_cfg.get("mlp_hidden_dim", 1024),
        value_dim=model_cfg.get("value_dim", 384),
        schema_dim=model_cfg.get("schema_dim", 384),
        temporal_dim=model_cfg.get("temporal_dim", 5),
        role_dim=model_cfg.get("role_dim", 64),
        dropout=model_cfg.get("dropout", 0.1),
    )
    inferencer = RelationalInferenceEngine(
        model=model,
        config=RelationalInferenceConfig(device=training_cfg.get("device")),
    )
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        inferencer.load_checkpoint(str(checkpoint_path))
    else:
        raise FileNotFoundError(checkpoint_path)

    metrics = run_zero_shot(inferencer, dataset, eval_cfg, eval_cfg.get("task_type", "binary"))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"metrics": metrics}, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
