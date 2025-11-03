#!/usr/bin/env python3
"""Produce Relational Transformer training samples to a Redis stream."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import redis
import yaml

from models.relational_transformer import (
    CellTokenizer,
    ContextSampler,
    ForeignKeySpec,
    RelationalDatabase,
    RelationalDataset,
    RelationalTableSpec,
    TargetSpec,
)
from models.relational_transformer.data import FrozenTextEncoder
from models.relational_transformer.stream import get_redis_client, serialize_sample


def build_database(config) -> RelationalDatabase:
    tables = []
    for table_cfg in config.get("tables", []):
        path = Path(table_cfg["path"]).expanduser()
        fmt = table_cfg.get("format", "csv").lower()
        if fmt == "csv":
            import pandas as pd

            dataframe = pd.read_csv(path)
        elif fmt == "parquet":
            import pandas as pd

            dataframe = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format {fmt}")
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


def build_targets(database: RelationalDatabase, config) -> list[TargetSpec]:
    targets = []
    for target_cfg in config.get("targets", []):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config (same as trainer)")
    parser.add_argument("--redis-url", default="redis://127.0.0.1:6379/0")
    parser.add_argument("--redis-stream", default="rt-training")
    parser.add_argument("--maxlen", type=int, help="Trim stream to this many messages (approximate)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional pause between messages (seconds)")
    parser.add_argument("--loop", action="store_true", help="Continuously loop over dataset")
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
    dataset = RelationalDataset(
        database=database,
        sampler=sampler,
        tokenizer=tokenizer,
        targets=targets,
        context_cells=context_cfg.get("max_cells", 1024),
        mask_probability=training_cfg.get("mask_probability", 0.15),
        include_text=context_cfg.get("include_text", False),
        allow_temporal_leakage=context_cfg.get("allow_temporal_leakage", False),
    )

    client = get_redis_client(args.redis_url)
    maxlen = args.maxlen
    loop = True
    iteration = 0
    while loop:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            payload = serialize_sample(sample)
            client.xadd(
                args.redis_stream,
                {"payload": payload},
                maxlen=maxlen,
                approximate=maxlen is not None,
            )
            iteration += 1
            if args.sleep:
                time.sleep(args.sleep)
        if not args.loop:
            break

    print(f"✅ Published {iteration} samples to stream '{args.redis_stream}'.")


if __name__ == "__main__":
    main()
