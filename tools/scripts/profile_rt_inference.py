#!/usr/bin/env python3
"""Profile Relational Transformer inference throughput on a dedicated GPU host."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import yaml

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
from models.relational_transformer.data import FrozenTextEncoder  # noqa: E402


def build_database(config: Dict[str, Any]) -> RelationalDatabase:
    tables_cfg = config.get("tables", [])
    if not tables_cfg:
        raise ValueError("Configuration must include at least one table definition.")

    tables: List[RelationalTableSpec] = []
    for table_cfg in tables_cfg:
        fmt = table_cfg.get("format", "csv").lower()
        path = Path(table_cfg["path"]).expanduser()
        if fmt == "csv":
            dataframe = pd.read_csv(path)
        elif fmt == "parquet":
            dataframe = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported table format '{fmt}'")
        foreign_keys = [
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
                foreign_keys=foreign_keys,
            )
        )
    return RelationalDatabase(tables)


def build_targets(database: RelationalDatabase, config: Dict[str, Any]) -> List[TargetSpec]:
    targets_cfg = config.get("targets", [])
    if not targets_cfg:
        raise ValueError("Configuration must define 'targets'.")

    targets: List[TargetSpec] = []
    for target in targets_cfg:
        table = target["table"]
        column = target["column"]
        pk_column = target.get("primary_key", database.primary_key(table))
        rows = target.get("rows")
        limit = target.get("limit")
        if rows is None:
            dataframe = database.dataframe(table)
            rows = dataframe[pk_column].tolist()
            if limit:
                rows = rows[:limit]
        for pk in rows:
            targets.append(TargetSpec(table=table, primary_key_value=pk, column=column))
    return targets


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_inference(
    inferencer: RelationalInferenceEngine,
    dataset: RelationalDataset,
    limit: Optional[int],
    warmup: int,
) -> Dict[str, Any]:
    total_samples = min(limit, len(dataset)) if limit is not None else len(dataset)
    if total_samples == 0:
        raise ValueError("Dataset is empty; nothing to profile.")

    timings: List[float] = []
    for idx in range(total_samples):
        sample = dataset[idx]
        start = time.perf_counter()
        inferencer.infer_sample(sample)
        synchronize_if_needed(inferencer.device)
        elapsed = time.perf_counter() - start
        if idx >= warmup:
            timings.append(elapsed)

    mean_latency = statistics.mean(timings)
    p95_latency = statistics.quantiles(timings, n=100)[94] if len(timings) >= 100 else max(timings)
    throughput = 1.0 / mean_latency if mean_latency > 0 else float("inf")

    return {
        "total_samples": total_samples,
        "warmup": warmup,
        "mean_latency_s": mean_latency,
        "p95_latency_s": p95_latency,
        "throughput_samples_per_s": throughput,
        "device": inferencer.device.type,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML configuration describing tables and targets.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to load into the inference engine.")
    parser.add_argument("--limit", type=int, help="Maximum number of samples to profile.")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup samples to discard from metrics.")
    parser.add_argument("--report-json", help="Optional path to write the metrics report as JSON.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

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

    targets = build_targets(database, config)
    temporal_lookback_seconds = context_cfg.get("temporal_lookback_seconds")
    temporal_lookback_hours = context_cfg.get("temporal_lookback_hours")
    if temporal_lookback_seconds is None and temporal_lookback_hours is not None:
        temporal_lookback_seconds = float(temporal_lookback_hours) * 3600.0

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
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    inferencer.load_checkpoint(str(checkpoint_path))

    metrics = profile_inference(
        inferencer=inferencer,
        dataset=dataset,
        limit=args.limit,
        warmup=max(args.warmup, 0),
    )
    print(json.dumps(metrics, indent=2))

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
