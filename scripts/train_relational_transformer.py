#!/usr/bin/env python3
"""Command-line entry point for Relational Transformer training and inference."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.relational_transformer import (  # noqa: E402
    CellTokenizer,
    ContextSampler,
    ForeignKeySpec,
    RelationalInferenceConfig,
    RelationalInferenceEngine,
    RelationalDatabase,
    RelationalDataset,
    RelationalTableSpec,
    RelationalTrainer,
    RelationalTrainingConfig,
    RelationalTransformer,
    TargetSpec,
)
from models.relational_transformer.data import RELATIONAL_ID_TO_DTYPE, FrozenTextEncoder  # noqa: E402
from models.relational_transformer.stream import get_redis_client, stream_batches  # noqa: E402


def _load_table_dataframe(path: str, fmt: str) -> pd.DataFrame:
    fmt = fmt.lower()
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format '{fmt}'")


def build_database(config: Dict[str, Any]) -> RelationalDatabase:
    tables_cfg = config.get("tables", [])
    if not tables_cfg:
        raise ValueError("Configuration must include at least one table definition.")

    table_specs: List[RelationalTableSpec] = []
    for table_cfg in tables_cfg:
        path = table_cfg["path"]
        fmt = table_cfg.get("format", "csv")
        dataframe = _load_table_dataframe(path, fmt)
        foreign_keys = [
            ForeignKeySpec(
                parent_table=fk["parent_table"],
                parent_column=fk["parent_column"],
                child_column=fk["child_column"],
            )
            for fk in table_cfg.get("foreign_keys", [])
        ]
        table_specs.append(
            RelationalTableSpec(
                name=table_cfg["name"],
                dataframe=dataframe,
                primary_key=table_cfg["primary_key"],
                timestamp_column=table_cfg.get("timestamp_column"),
                foreign_keys=foreign_keys,
            )
        )
    return RelationalDatabase(table_specs)


def build_targets(database: RelationalDatabase, config: Dict[str, Any]) -> List[TargetSpec]:
    targets_cfg = config.get("targets", [])
    if not targets_cfg:
        raise ValueError("Configuration must define 'targets'.")

    targets: List[TargetSpec] = []
    for target in targets_cfg:
        table = target["table"]
        column = target["column"]
        pk_column = target.get("primary_key", database.primary_key(table))

        if "rows" in target:
            pk_values = target["rows"]
        else:
            dataframe = database.dataframe(table)
            pk_values = dataframe[pk_column].tolist()
            limit = target.get("limit")
            if limit:
                pk_values = pk_values[:limit]

        for pk in pk_values:
            targets.append(TargetSpec(table=table, primary_key_value=pk, column=column))

    return targets

def run_zero_shot(
    inferencer: RelationalInferenceEngine,
    dataset: RelationalDataset,
    num_examples: int,
    output_path: Optional[str],
) -> None:
    results = []
    limit = min(num_examples, len(dataset))
    for idx in range(limit):
        sample = dataset[idx]
        predictions = inferencer.infer_sample(sample)
        loss_mask = sample["loss_mask"].bool()
        dtype_ids = sample["dtype_ids"]
        target_values = sample["target_values"]

        target_indices = torch.nonzero(loss_mask, as_tuple=False).flatten().tolist()
        for position in target_indices:
            dtype_id = int(dtype_ids[position].item())
            dtype_name = RELATIONAL_ID_TO_DTYPE.get(dtype_id, "unknown")
            pred_tensor = predictions.get(dtype_name)
            if pred_tensor is None:
                continue
            raw_prediction = pred_tensor[position].item()
            target_value = float(target_values[position].item())
            results.append(
                {
                    "example_index": idx,
                    "position": position,
                    "dtype": dtype_name,
                    "prediction": raw_prediction,
                    "target_normalized": target_value,
                }
            )

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    else:
        print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML configuration file for RT training.")
    parser.add_argument(
        "--mode",
        choices=['pretrain', 'fine-tune', 'zero-shot', 'stream-pretrain', 'stream-fine-tune'],
        required=True,
        help="Operational mode.",
    )
    parser.add_argument("--checkpoint-in", help="Checkpoint to load (for zero-shot or explicit resume).")
    parser.add_argument("--checkpoint-out", help="Path to save the trained checkpoint.")
    parser.add_argument("--predictions-out", help="Where to write zero-shot predictions (JSON).")
    parser.add_argument('--redis-url', default='redis://127.0.0.1:6379/0', help='Redis connection URL for streaming modes')
    parser.add_argument('--redis-stream', default='rt-training', help='Redis stream key for training batches')
    parser.add_argument('--redis-group', default='rt-trainer', help='Redis consumer group name')
    parser.add_argument('--redis-consumer', help='Redis consumer name (defaults to hostname-pid)')
    parser.add_argument('--redis-block-ms', type=int, default=5000, help='Block timeout (ms) when reading from Redis stream')
    parser.add_argument("--resume-from", help="Resume training from an existing checkpoint.")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Automatically resume from --checkpoint-out if it already exists.",
    )
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision (AMP) if CUDA is available.")
    parser.add_argument("--grad-clip", type=float, help="Gradient clipping norm (overrides config).")
    parser.add_argument("--loss-scale", type=float, help="Initial loss scale for AMP.")
    parser.add_argument(
        "--disable-dynamic-loss-scale",
        action="store_true",
        help="Disable dynamic loss scaling when AMP is enabled.",
    )
    parser.add_argument("--ablate-column", action="store_true")
    parser.add_argument("--ablate-feature", action="store_true")
    parser.add_argument("--ablate-neighbor", action="store_true")
    parser.add_argument("--ablate-temporal", action="store_true")
    parser.add_argument("--ablate-full", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    database = build_database(config)
    context_cfg = config.get("context", {})
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    context_length = context_cfg.get("max_cells", 1024)
    width_bound = context_cfg.get("width_bound", 8)
    allow_temporal_leakage = context_cfg.get("allow_temporal_leakage", False)

    trainer_config = RelationalTrainingConfig(
        context_length=context_length,
        batch_size=training_cfg.get("batch_size", 32),
        mask_probability=training_cfg.get("mask_probability", 0.15),
        pretrain_learning_rate=training_cfg.get("pretrain_learning_rate", 1e-3),
        pretrain_weight_decay=training_cfg.get("pretrain_weight_decay", 0.01),
        pretrain_steps=training_cfg.get("pretrain_steps", 100_000),
        fine_tune_learning_rate=training_cfg.get("fine_tune_learning_rate", 1e-4),
        fine_tune_weight_decay=training_cfg.get("fine_tune_weight_decay", 0.0),
        fine_tune_steps=training_cfg.get("fine_tune_steps", 33_000),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        num_workers=training_cfg.get("num_workers", 0),
        device=training_cfg.get("device"),
        use_amp=training_cfg.get("amp", False),
        grad_clip=training_cfg.get("grad_clip"),
        loss_scale=training_cfg.get("loss_scale"),
        dynamic_loss_scale=training_cfg.get("dynamic_loss_scale", True),
    )
    if args.amp:
        trainer_config.use_amp = True
    if args.grad_clip is not None:
        trainer_config.grad_clip = args.grad_clip
    if args.loss_scale is not None:
        trainer_config.loss_scale = args.loss_scale
        trainer_config.use_amp = True
    if args.disable_dynamic_loss_scale:
        trainer_config.dynamic_loss_scale = False
    trainer_config.ablate_column = trainer_config.ablate_column or args.ablate_column
    trainer_config.ablate_feature = trainer_config.ablate_feature or args.ablate_feature
    trainer_config.ablate_neighbor = trainer_config.ablate_neighbor or args.ablate_neighbor
    trainer_config.ablate_temporal = trainer_config.ablate_temporal or args.ablate_temporal
    trainer_config.ablate_full = trainer_config.ablate_full or args.ablate_full

    sampler = ContextSampler(
        database,
        max_context_cells=context_length,
        width_bound=width_bound,
        random_state=context_cfg.get("random_seed"),
    )
    text_encoder_name = model_cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2")
    text_encoder = None
    if text_encoder_name:
        text_device = training_cfg.get("device")
        text_encoder = FrozenTextEncoder(model_name=text_encoder_name, device=text_device)
    tokenizer = CellTokenizer(
        database,
        text_encoder=text_encoder,
        value_dim=model_cfg.get("value_dim", 384),
        schema_dim=model_cfg.get("schema_dim", 384),
        temporal_dim=model_cfg.get("temporal_dim", 5),
        role_dim=model_cfg.get("role_dim", 64),
        schema_seed=context_cfg.get("schema_seed", 0),
    )

    if args.mode == "zero-shot":
        mask_probability = 0.0
    else:
        mask_probability = trainer_config.mask_probability

    targets = build_targets(database, config)
    temporal_lookback_seconds = context_cfg.get("temporal_lookback_seconds")
    temporal_lookback_hours = context_cfg.get("temporal_lookback_hours")
    if temporal_lookback_hours is not None:
        temporal_lookback_seconds = float(temporal_lookback_hours) * 3600.0
    dataset = RelationalDataset(
        database=database,
        sampler=sampler,
        tokenizer=tokenizer,
        targets=targets,
        context_cells=context_length,
        mask_probability=mask_probability,
        include_text=context_cfg.get("include_text", False),
        allow_temporal_leakage=allow_temporal_leakage,
        temporal_lookback_seconds=temporal_lookback_seconds,
    )

    model = RelationalTransformer(
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_layers=model_cfg.get("num_layers", 12),
        num_heads=model_cfg.get("num_heads", 8),
        mlp_hidden_dim=model_cfg.get("mlp_hidden_dim", 1024),
        value_dim=model_cfg.get("value_dim", 384),
        schema_dim=model_cfg.get("schema_dim", 384),
        dropout=model_cfg.get("dropout", 0.1),
    )

    resume_path = args.resume_from or args.checkpoint_in
    if args.auto_resume and not resume_path and args.checkpoint_out:
        candidate = Path(args.checkpoint_out)
        if candidate.exists():
            resume_path = str(candidate)

    if args.mode == "zero-shot":
        if not resume_path:
            raise ValueError("Zero-shot mode requires --checkpoint-in or --resume-from to supply model weights.")
        checkpoint_path = Path(resume_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        inference_config = RelationalInferenceConfig(device=trainer_config.device)
        inferencer = RelationalInferenceEngine(model=model, config=inference_config)
        inferencer.load_checkpoint(str(checkpoint_path))
        zero_shot_cfg = config.get("zero_shot", {})
        num_examples = zero_shot_cfg.get("num_examples", 8)
        run_zero_shot(inferencer, dataset, num_examples=num_examples, output_path=args.predictions_out)
        return

    trainer = RelationalTrainer(model=model, dataset=dataset, config=trainer_config)

    if resume_path:
        resume_file = Path(resume_path)
        if resume_file.is_file():
            print(f"🔁 Resuming from checkpoint: {resume_file}")
            trainer.load_checkpoint(str(resume_file))
        else:
            print(f"⚠️  Resume path {resume_file} not found; starting fresh.")

    if args.mode == "pretrain":
        if not args.checkpoint_out:
            raise ValueError("--checkpoint-out is required when running in pretrain mode.")
        trainer.pretrain(steps=training_cfg.get("steps", {}).get("pretrain"))
        checkpoint_path = Path(args.checkpoint_out)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(checkpoint_path), metadata={"stage": "pretrain"})
    elif args.mode == "fine-tune":
        if not args.checkpoint_out:
            raise ValueError("--checkpoint-out is required when running in fine-tune mode.")
        trainer.fine_tune(
            steps=training_cfg.get("steps", {}).get("fine_tune"),
            learning_rate=training_cfg.get("fine_tune_learning_rate"),
            weight_decay=training_cfg.get("fine_tune_weight_decay"),
        )
        checkpoint_path = Path(args.checkpoint_out)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(checkpoint_path), metadata={"stage": "fine_tune"})
    elif args.mode in {"stream-pretrain", "stream-fine-tune"}:
        stream_key = args.redis_stream
        group = args.redis_group
        consumer = args.redis_consumer or f"{socket.gethostname()}-{os.getpid()}"
        client = get_redis_client(args.redis_url)
        stage = "pretrain" if args.mode == "stream-pretrain" else "fine_tune"
        trainer.loaded_optimizer_stage = stage
        steps_cfg = training_cfg.get("steps", {})
        if stage == "pretrain":
            steps = steps_cfg.get("pretrain") or trainer_config.pretrain_steps
        else:
            steps = steps_cfg.get("fine_tune") or trainer_config.fine_tune_steps
        if steps is None:
            raise ValueError("Streaming mode requires training.steps to be defined.")
        completed = 0
        iterator = stream_batches(client, stream_key, group, consumer, trainer.device, block_ms=args.redis_block_ms)
        while completed < steps:
            msg_id, batch = next(iterator)
            try:
                trainer.train_on_batch(batch)
            except Exception:
                raise
            else:
                client.xack(stream_key, group, msg_id)
                completed += 1
        if args.checkpoint_out:
            checkpoint_path = Path(args.checkpoint_out)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(checkpoint_path), metadata={"stage": stage})


if __name__ == "__main__":
    main()
