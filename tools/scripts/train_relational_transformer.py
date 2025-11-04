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

# Add services/training to path for Glean integration
ROOT_DIR = Path(__file__).resolve().parents[2]
TRAINING_SERVICE_DIR = ROOT_DIR / "services" / "training"
if str(TRAINING_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_SERVICE_DIR))

try:
    from glean_integration import ingest_glean_data_for_training, GleanTrainingClient
    from extract_client import ExtractServiceClient
    from pipeline import TrainingPipeline
    TRAINING_SERVICE_AVAILABLE = True
except ImportError:
    TRAINING_SERVICE_AVAILABLE = False
    print("⚠️  Training service integration not available. Install services/training/")

# ROOT_DIR is already set above for Glean integration
TRAINING_ROOT_DIR = Path(__file__).resolve().parents[1]

if str(TRAINING_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT_DIR))

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
    
    # Glean integration arguments
    parser.add_argument("--glean-project-id", help="Project ID for Glean historical data queries")
    parser.add_argument("--glean-system-id", help="System ID for Glean historical data queries")
    parser.add_argument("--glean-days-back", type=int, default=30, help="Number of days to look back in Glean (default: 30)")
    parser.add_argument("--glean-enable", action="store_true", help="Enable Glean Catalog integration for training")
    parser.add_argument("--glean-output-dir", help="Directory to save Glean training data")
    
    # Extract service integration arguments
    parser.add_argument("--extract-service-url", help="Extract service URL (default: from EXTRACT_SERVICE_URL env)")
    parser.add_argument("--extract-project-id", help="Project ID for Extract service knowledge graph")
    parser.add_argument("--extract-system-id", help="System ID for Extract service knowledge graph")
    parser.add_argument("--extract-json-tables", nargs="+", help="JSON table file paths for Extract service")
    parser.add_argument("--extract-hive-ddls", nargs="+", help="Hive DDL file paths for Extract service")
    parser.add_argument("--extract-controlm-files", nargs="+", help="Control-M XML file paths for Extract service")
    parser.add_argument("--extract-enable", action="store_true", help="Enable Extract service integration for training")
    
    # Training pipeline arguments
    parser.add_argument("--training-pipeline-enable", action="store_true", help="Enable full training pipeline (Extract + Glean + Pattern Learning)")
    parser.add_argument("--training-output-dir", help="Directory for training pipeline output")
    
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    # Run full training pipeline if enabled
    training_pipeline_results = None
    graph_data = None
    glean_data = None
    learned_patterns = None
    
    if args.training_pipeline_enable and TRAINING_SERVICE_AVAILABLE:
        print("🚀 Running full training pipeline (Extract + Glean + Pattern Learning)...")
        try:
            pipeline = TrainingPipeline(
                extract_service_url=args.extract_service_url,
                output_dir=args.training_output_dir
            )
            
            training_pipeline_results = pipeline.run_full_pipeline(
                project_id=args.extract_project_id or args.glean_project_id or "sgmi",
                system_id=args.extract_system_id or args.glean_system_id,
                json_tables=args.extract_json_tables,
                hive_ddls=args.extract_hive_ddls,
                control_m_files=args.extract_controlm_files,
                glean_days_back=args.glean_days_back,
                enable_glean=args.glean_enable,
                enable_temporal_analysis=True  # Enable temporal analysis by default
            )
            
            if training_pipeline_results.get("status") == "success":
                print("✅ Training pipeline completed successfully")
                print(f"   Steps completed: {', '.join(training_pipeline_results.get('steps', {}).keys())}")
                
                # Extract data for use in model training
                # Note: Actual graph_data and glean_data would come from the pipeline
                # For now, we'll use the pipeline results
                print("📊 Training pipeline data available for model training")
            else:
                print(f"⚠️  Training pipeline completed with issues: {training_pipeline_results}")
        except Exception as e:
            print(f"⚠️  Training pipeline failed (continuing with manual integration): {e}")
            training_pipeline_results = None
    elif args.training_pipeline_enable and not TRAINING_SERVICE_AVAILABLE:
        print("⚠️  Training pipeline requested but not available. Continuing with manual integration.")
    
    # Manual Extract service integration if enabled (without full pipeline)
    if args.extract_enable and TRAINING_SERVICE_AVAILABLE and not args.training_pipeline_enable:
        print("📊 Querying Extract service for knowledge graph...")
        try:
            extract_client = ExtractServiceClient(extract_service_url=args.extract_service_url)
            
            if not extract_client.health_check():
                print("⚠️  Extract service not available. Skipping Extract integration.")
            else:
                graph_data = extract_client.get_knowledge_graph(
                    project_id=args.extract_project_id or "sgmi",
                    system_id=args.extract_system_id,
                    json_tables=args.extract_json_tables,
                    hive_ddls=args.extract_hive_ddls,
                    control_m_files=args.extract_controlm_files
                )
                
                print(f"✅ Retrieved knowledge graph: {len(graph_data.get('nodes', []))} nodes, "
                      f"{len(graph_data.get('edges', []))} edges")
        except Exception as e:
            print(f"⚠️  Extract service integration failed (continuing without graph data): {e}")
            graph_data = None
    
    # Manual Glean integration if enabled (without full pipeline)
    if args.glean_enable and TRAINING_SERVICE_AVAILABLE and not args.training_pipeline_enable:
        print("📊 Ingesting historical data from Glean Catalog...")
        try:
            glean_data = ingest_glean_data_for_training(
                project_id=args.glean_project_id,
                system_id=args.glean_system_id,
                days_back=args.glean_days_back,
                output_dir=args.glean_output_dir
            )
            print(f"✅ Ingested {glean_data['metadata']['node_count']} nodes, {glean_data['metadata']['edge_count']} edges from Glean")
            if glean_data.get('metrics', {}).get('averages'):
                print(f"   Metrics: entropy_avg={glean_data['metrics']['averages'].get('metadata_entropy', 'N/A'):.2f}, "
                      f"kl_div_avg={glean_data['metrics']['averages'].get('kl_divergence', 'N/A'):.2f}")
        except Exception as e:
            print(f"⚠️  Glean ingestion failed (continuing without historical data): {e}")
            glean_data = None
    elif args.glean_enable and not TRAINING_SERVICE_AVAILABLE:
        print("⚠️  Glean integration requested but not available. Continuing without historical data.")
    
    # Learn patterns from graph and Glean data if available
    if (graph_data or glean_data) and TRAINING_SERVICE_AVAILABLE:
        print("🧠 Learning patterns from knowledge graph and Glean data...")
        try:
            from pattern_learning import PatternLearningEngine
            
            pattern_engine = PatternLearningEngine()
            
            # Extract nodes and edges
            nodes = graph_data.get("nodes", []) if graph_data else []
            edges = graph_data.get("edges", []) if graph_data else []
            metrics = graph_data.get("metrics", {}) if graph_data else {}
            
            # Add Glean data if available
            if glean_data:
                nodes.extend(glean_data.get("nodes", []))
                edges.extend(glean_data.get("edges", []))
            
            # Learn patterns
            learned_patterns = pattern_engine.learn_patterns(
                nodes=nodes,
                edges=edges,
                metrics=metrics,
                glean_data=glean_data
            )
            
            print(f"✅ Learned patterns: {learned_patterns['summary'].get('unique_column_types', 0)} column types, "
                  f"{learned_patterns['summary'].get('unique_edge_labels', 0)} relationship types")
        except Exception as e:
            print(f"⚠️  Pattern learning failed (continuing without patterns): {e}")
            learned_patterns = None
    
    database = build_database(config)
    context_cfg = config.get("context", {})
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    
    # Store training context for potential use in dataset/enrichment
    training_context = {
        "graph_data": graph_data,
        "glean_data": glean_data,
        "learned_patterns": learned_patterns,
        "pipeline_results": training_pipeline_results,
    }
    
    # TODO: Integrate training_context into training dataset
    # This could involve:
    # 1. Adding historical patterns as features
    # 2. Using temporal metrics for context
    # 3. Enriching training data with Glean insights
    # 4. Using learned patterns for data augmentation

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
        
        # Evaluate and export training metrics if training service available
        if TRAINING_SERVICE_AVAILABLE:
            try:
                from evaluation import evaluate_training_results, export_training_metrics_to_glean
                from glean_integration import GleanTrainingClient
                
                # Get training metrics (simplified - would need actual trainer metrics)
                model_metrics = {
                    "loss": getattr(trainer, 'last_loss', None),
                    "step": trainer.global_step,
                }
                
                # Evaluate results
                evaluation = evaluate_training_results(
                    model_metrics=model_metrics,
                    training_context=training_context if 'training_context' in locals() else None,
                    checkpoint_path=str(checkpoint_path)
                )
                
                # Export to Glean if enabled
                if args.glean_enable:
                    glean_client = GleanTrainingClient()
                    export_info = export_training_metrics_to_glean(
                        evaluation=evaluation,
                        glean_client=glean_client,
                        output_dir=args.training_output_dir or args.glean_output_dir
                    )
                    print(f"✅ Training metrics exported: {export_info.get('output_file', 'N/A')}")
            except Exception as e:
                print(f"⚠️  Training evaluation/export failed (non-fatal): {e}")
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
        
        # Evaluate and export training metrics if training service available
        if TRAINING_SERVICE_AVAILABLE:
            try:
                from evaluation import evaluate_training_results, export_training_metrics_to_glean
                from glean_integration import GleanTrainingClient
                
                # Get training metrics (simplified - would need actual trainer metrics)
                model_metrics = {
                    "loss": getattr(trainer, 'last_loss', None),
                    "step": trainer.global_step,
                }
                
                # Evaluate results
                evaluation = evaluate_training_results(
                    model_metrics=model_metrics,
                    training_context=training_context if 'training_context' in locals() else None,
                    checkpoint_path=str(checkpoint_path)
                )
                
                # Export to Glean if enabled
                if args.glean_enable:
                    glean_client = GleanTrainingClient()
                    export_info = export_training_metrics_to_glean(
                        evaluation=evaluation,
                        glean_client=glean_client,
                        output_dir=args.training_output_dir or args.glean_output_dir
                    )
                    print(f"✅ Training metrics exported: {export_info.get('output_file', 'N/A')}")
            except Exception as e:
                print(f"⚠️  Training evaluation/export failed (non-fatal): {e}")
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
