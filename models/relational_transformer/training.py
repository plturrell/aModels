"""Training utilities for the Relational Transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import RELATIONAL_DTYPE_TO_ID, RelationalDataset
from .model import RelationalTransformer


@dataclass
class RelationalTrainingConfig:
    context_length: int = 1024
    batch_size: int = 32
    mask_probability: float = 0.15
    pretrain_learning_rate: float = 1e-3
    pretrain_weight_decay: float = 0.01
    pretrain_steps: int = 100_000
    fine_tune_learning_rate: float = 1e-4
    fine_tune_weight_decay: float = 0.0
    fine_tune_steps: int = 33_000
    max_grad_norm: float = 1.0
    num_workers: int = 0
    device: Optional[str] = None
    use_amp: bool = False
    grad_clip: Optional[float] = None
    loss_scale: Optional[float] = None
    dynamic_loss_scale: bool = True
    link_loss_weight: float = 0.1
    ablate_column: bool = False
    ablate_feature: bool = False
    ablate_neighbor: bool = False
    ablate_temporal: bool = False
    ablate_full: bool = False


class RelationalTrainer:
    """Handles pretraining, fine-tuning, and inference for RT models."""

    def __init__(
        self,
        model: RelationalTransformer,
        dataset: RelationalDataset,
        config: Optional[RelationalTrainingConfig] = None,
    ):
        self.model = model
        self.dataset = dataset
        self.config = config or RelationalTrainingConfig()
        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model.to(self.device)

        self.boolean_loss = nn.BCEWithLogitsLoss()
        self.regression_loss = nn.SmoothL1Loss()
        self.optimizer = self._build_optimizer(
            lr=self.config.pretrain_learning_rate,
            weight_decay=self.config.pretrain_weight_decay,
        )
        self.use_amp = self.config.use_amp and torch.cuda.is_available()
        if self.config.use_amp and not torch.cuda.is_available():
            print("⚠️  AMP requested but CUDA is unavailable. Continuing with standard precision.")
        self.grad_clip = self.config.grad_clip
        self.dynamic_loss_scale = self.config.dynamic_loss_scale
        if self.use_amp:
            scaler_kwargs = {}
            if self.config.loss_scale is not None:
                scaler_kwargs["init_scale"] = float(self.config.loss_scale)
            if not self.dynamic_loss_scale:
                scaler_kwargs["growth_factor"] = 1.0
                scaler_kwargs["backoff_factor"] = 1.0
                scaler_kwargs["growth_interval"] = 1
            self.scaler = torch.cuda.amp.GradScaler(enabled=True, **scaler_kwargs)
        else:
            self.scaler = None

        self.link_loss_weight = self.config.link_loss_weight
        self.link_loss_fn = nn.BCEWithLogitsLoss()

        self.global_step = 0
        self.loaded_optimizer_stage: Optional[str] = None

    def _build_optimizer(self, lr: float, weight_decay: float) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def _build_dataloader(self, shuffle: bool = True, batch_size: Optional[int] = None) -> DataLoader:
        batch = batch_size or self.config.batch_size
        return DataLoader(
            self.dataset,
            batch_size=batch,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
        )

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        value_embeddings = batch["value_embeddings"].to(self.device)
        schema_embeddings = batch["schema_embeddings"].to(self.device)
        temporal_embeddings = batch.get("temporal_embeddings")
        if temporal_embeddings is not None:
            temporal_embeddings = temporal_embeddings.to(self.device)
        role_embeddings = batch.get("role_embeddings")
        if role_embeddings is not None:
            role_embeddings = role_embeddings.to(self.device)
        dtype_ids = batch["dtype_ids"].to(self.device)
        padding_mask = batch["padding_mask"].to(self.device)
        is_masked = batch["is_masked"].to(self.device)
        row_ids = batch["row_ids"].long().to(self.device)

        masks = {
            "column": batch["column_attention"].to(self.device),
            "feature": batch["feature_attention"].to(self.device),
            "neighbor": batch["neighbor_attention"].to(self.device),
            "temporal": batch.get("temporal_attention", batch["full_attention"]).to(self.device),
            "full": batch["full_attention"].to(self.device),
        }
        if self.config.ablate_column:
            masks["column"] = torch.zeros_like(masks["column"], device=self.device)
        if self.config.ablate_feature:
            masks["feature"] = torch.zeros_like(masks["feature"], device=self.device)
        if self.config.ablate_neighbor:
            masks["neighbor"] = torch.zeros_like(masks["neighbor"], device=self.device)
        if self.config.ablate_temporal:
            masks["temporal"] = torch.zeros_like(masks["temporal"], device=self.device)
        if self.config.ablate_full:
            masks["full"] = torch.zeros_like(masks["full"], device=self.device)

        hidden_states = self.model(
            value_embeddings=value_embeddings,
            schema_embeddings=schema_embeddings,
            temporal_embeddings=temporal_embeddings,
            role_embeddings=role_embeddings,
            dtype_ids=dtype_ids,
            masks=masks,
            is_masked=is_masked,
            padding_mask=padding_mask,
        )

        if row_ids.dim() == 1:
            row_ids = row_ids.unsqueeze(0)

        link_pairs = self._prepare_edge_pairs(batch.get("edge_index"), batch.get("edge_labels"))
        if link_pairs and self.link_loss_weight > 0:
            if len(link_pairs) == 1 and row_ids.size(0) > 1:
                link_pairs = link_pairs * row_ids.size(0)
            link_losses = []
            for b in range(min(len(link_pairs), row_ids.size(0))):
                edges, labels = link_pairs[b]
                if isinstance(edges, torch.Tensor):
                    edges_t = edges.to(self.device)
                elif edges is None:
                    continue
                else:
                    edges_t = torch.tensor(edges, dtype=torch.long, device=self.device)
                if edges_t.numel() == 0:
                    continue
                if isinstance(labels, torch.Tensor):
                    labels_t = labels.to(self.device)
                else:
                    labels_t = torch.tensor(labels, dtype=torch.float32, device=self.device)
                rows_b = row_ids[b]
                hidden_b = hidden_states[b]
                num_rows = int(rows_b.max().item()) + 1 if rows_b.numel() > 0 else 0
                if num_rows == 0:
                    continue
                row_emb = torch.zeros((num_rows, hidden_b.size(-1)), device=self.device)
                counts = torch.zeros(num_rows, device=self.device)
                row_emb.index_add_(0, rows_b, hidden_b)
                counts.index_add_(0, rows_b, torch.ones_like(rows_b, dtype=torch.float32))
                row_emb = row_emb / counts.clamp(min=1.0).unsqueeze(-1)
                logits_lp = self.model.link_logits(row_emb, edges_t)
                link_losses.append(self.link_loss_fn(logits_lp, labels_t))
            if link_losses:
                link_loss = torch.stack(link_losses).mean()
                loss = loss + self.link_loss_weight * link_loss

        logits = self.model.decode(hidden_states)
        loss_mask = batch["loss_mask"].to(self.device)
        targets = batch["target_values"].to(self.device)

        total_loss = torch.tensor(0.0, device=self.device)
        loss_components = 0

        for dtype, dtype_id in RELATIONAL_DTYPE_TO_ID.items():
            if dtype not in logits:
                continue
            dtype_mask = (dtype_ids == dtype_id) & loss_mask
            if not dtype_mask.any():
                continue
            predictions = logits[dtype][dtype_mask]
            target_slice = targets[dtype_mask]
            if dtype == "boolean":
                loss_val = self.boolean_loss(predictions, target_slice)
            else:
                loss_val = self.regression_loss(predictions, target_slice)
            total_loss = total_loss + loss_val
            loss_components += 1

        if loss_components == 0:
            return total_loss
        return total_loss / loss_components

    def _step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        autocast_enabled = self.use_amp
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            loss = self._compute_loss(batch)

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            elif self.config.max_grad_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            clip_norm = self.grad_clip if self.grad_clip is not None else self.config.max_grad_norm
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            self.optimizer.step()

        self.global_step += 1
        return float(loss.detach().cpu().item())

    def pretrain(self, steps: Optional[int] = None, progress_callback=None) -> None:
        target_steps = steps or self.config.pretrain_steps
        dataloader = self._build_dataloader(shuffle=True)
        self.model.train()
        self.loaded_optimizer_stage = "pretrain"
        completed = 0
        total_required = target_steps
        while completed < total_required:
            for batch in dataloader:
                loss = self._step(batch)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_required, loss)
                if completed >= total_required:
                    break

    def fine_tune(
        self,
        steps: Optional[int] = None,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        progress_callback=None,
    ) -> None:
        target_steps = steps or self.config.fine_tune_steps
        lr = learning_rate or self.config.fine_tune_learning_rate
        wd = weight_decay if weight_decay is not None else self.config.fine_tune_weight_decay
        if self.loaded_optimizer_stage == "fine_tune":
            for group in self.optimizer.param_groups:
                group["lr"] = lr
                group["weight_decay"] = wd
        else:
            self.optimizer = self._build_optimizer(lr=lr, weight_decay=wd)
        self.loaded_optimizer_stage = "fine_tune"

        dataloader = self._build_dataloader(shuffle=True)
        self.model.train()
        completed = 0
        total_required = target_steps
        while completed < total_required:
            for batch in dataloader:
                loss = self._step(batch)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_required, loss)
                if completed >= total_required:
                    break

    def train_on_batch(self, batch: Dict[str, torch.Tensor]) -> float:
        """Train on a single batch (streaming or manual feeding)."""
        self.model.train()
        return self._step(batch)

    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, str]] = None) -> None:
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
            "config": self.config.__dict__,
            "metadata": metadata or {},
        }
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        torch.save(state, path)

        if metadata:
            self.loaded_optimizer_stage = metadata.get("stage")

    def _prepare_edge_pairs(self, edge_index, edge_labels):
        if edge_index is None or edge_labels is None:
            return []
        if isinstance(edge_index, torch.Tensor):
            if edge_index.dim() == 3:
                return [(edge_index[i], edge_labels[i]) for i in range(edge_index.size(0))]
            return [(edge_index, edge_labels)]
        if isinstance(edge_index, list):
            return list(zip(edge_index, edge_labels))
        return []

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            incompatible = self.model.load_state_dict(checkpoint["model"], strict=False)
            missing = incompatible.missing_keys if hasattr(incompatible, "missing_keys") else incompatible[0]
            unexpected = incompatible.unexpected_keys if hasattr(incompatible, "unexpected_keys") else incompatible[1]
            if missing:
                print(f"⚠️  Missing parameters in checkpoint: {sorted(missing)}")
            if unexpected:
                print(f"⚠️  Unexpected parameters in checkpoint: {sorted(unexpected)}")
            if "optimizer" in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                except ValueError:
                    print("⚠️  Optimizer state in checkpoint is incompatible with current optimizer; skipping.")
            if self.scaler is not None and "scaler" in checkpoint:
                try:
                    self.scaler.load_state_dict(checkpoint["scaler"])
                except ValueError:
                    print("⚠️  AMP scaler state incompatible with current configuration; skipping scaler restore.")
            self.global_step = int(checkpoint.get("step", 0))
            metadata = checkpoint.get("metadata", {})
            if isinstance(metadata, dict):
                self.loaded_optimizer_stage = metadata.get("stage")
        else:
            # Backward compatibility: checkpoint is the model state dict
            self.model.load_state_dict(checkpoint, strict=False)
            self.global_step = 0
            self.loaded_optimizer_stage = None
