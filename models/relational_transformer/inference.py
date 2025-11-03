"""Inference utilities for the Relational Transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .model import RelationalTransformer


@dataclass
class RelationalInferenceConfig:
    """Configuration for running Relational Transformer inference."""

    device: Optional[str] = None
    apply_sigmoid_to_boolean: bool = True
    move_outputs_to_cpu: bool = True


class RelationalInferenceEngine:
    """Lightweight inference wrapper designed for deployment on dedicated GPUs."""

    def __init__(
        self,
        model: RelationalTransformer,
        config: Optional[RelationalInferenceConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or RelationalInferenceConfig()
        device_name = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name)
        self.model.to(self.device)
        self.model.eval()

    def load_checkpoint(self, path: str) -> None:
        """Load model weights from a training checkpoint, ignoring optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        state_dict: Dict[str, torch.Tensor]
        metadata = {}
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
            metadata = checkpoint.get("metadata", {})
        else:
            state_dict = checkpoint
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                "⚠️  Loaded checkpoint with mismatched keys. "
                f"Missing: {incompatible.missing_keys} Unexpected: {incompatible.unexpected_keys}"
            )
        if metadata:
            stage = metadata.get("stage")
            if stage:
                print(f"✅ Loaded checkpoint trained during '{stage}' stage.")
        self.model.eval()

    def infer_sample(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run inference on a single dataset sample."""
        batch = {
            key: value.unsqueeze(0) if isinstance(value, torch.Tensor) and value.dim() <= 2 else value
            for key, value in sample.items()
            if isinstance(value, torch.Tensor)
        }
        return self.infer_batch(batch)

    def infer_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run inference on a prepared batch dictionary."""
        self.model.eval()
        with torch.no_grad():
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
            masks = {
                "column": batch["column_attention"].to(self.device),
                "feature": batch["feature_attention"].to(self.device),
                "neighbor": batch["neighbor_attention"].to(self.device),
                "temporal": batch.get("temporal_attention", batch["full_attention"]).to(self.device),
                "full": batch["full_attention"].to(self.device),
            }
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
            logits = self.model.decode(hidden_states)
            predictions: Dict[str, torch.Tensor] = {}
            for dtype, tensor in logits.items():
                squeezed = tensor.squeeze(0)
                if dtype == "boolean" and self.config.apply_sigmoid_to_boolean:
                    squeezed = torch.sigmoid(squeezed)
                if self.config.move_outputs_to_cpu:
                    squeezed = squeezed.cpu()
                predictions[dtype] = squeezed
            return predictions
