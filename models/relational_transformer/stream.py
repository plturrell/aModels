"""Redis stream helpers for Relational Transformer training."""

from __future__ import annotations

import json
from typing import Dict, Iterable, Iterator, Optional, Tuple

import redis
import torch


def serialize_sample(sample: Dict[str, torch.Tensor]) -> str:
    """Serialize a dataset sample into a JSON payload."""

    def tolist(key: str) -> list:
        tensor = sample[key]
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().tolist()
        return tensor

    payload = {
        "value_embeddings": tolist("value_embeddings"),
        "schema_embeddings": tolist("schema_embeddings"),
        "temporal_embeddings": tolist("temporal_embeddings"),
        "role_embeddings": tolist("role_embeddings"),
        "dtype_ids": tolist("dtype_ids"),
        "column_ids": tolist("column_ids"),
        "row_ids": tolist("row_ids"),
        "padding_mask": tolist("padding_mask"),
        "is_masked": tolist("is_masked"),
        "loss_mask": tolist("loss_mask"),
        "target_values": tolist("target_values"),
        "column_attention": tolist("column_attention"),
        "feature_attention": tolist("feature_attention"),
        "neighbor_attention": tolist("neighbor_attention"),
        "temporal_attention": tolist("temporal_attention"),
        "edge_index": tolist("edge_index"),
        "edge_labels": tolist("edge_labels"),
        "full_attention": tolist("full_attention"),
        "seq_len": int(sample["seq_len"].item()) if isinstance(sample["seq_len"], torch.Tensor) else int(sample["seq_len"]),
    }
    return json.dumps(payload)


def _tensor_from_list(data, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype)


def deserialize_sample(
    payload: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Deserialize JSON payload back into a batch dictionary expected by the trainer."""
    data = json.loads(payload)
    batch: Dict[str, torch.Tensor] = {}

    batch["value_embeddings"] = _tensor_from_list(data["value_embeddings"], torch.float32).unsqueeze(0).to(device)
    batch["schema_embeddings"] = _tensor_from_list(data["schema_embeddings"], torch.float32).unsqueeze(0).to(device)
    temporal_list = data.get("temporal_embeddings")
    if temporal_list is None:
        temporal_dim = 5
        temporal_list = [[0.0] * temporal_dim for _ in range(len(data["value_embeddings"]))]
    batch["temporal_embeddings"] = _tensor_from_list(temporal_list, torch.float32).unsqueeze(0).to(device)

    role_list = data.get("role_embeddings")
    if role_list is None:
        role_dim = 64
        role_list = [[0.0] * role_dim for _ in range(len(data["value_embeddings"]))]
    batch["role_embeddings"] = _tensor_from_list(role_list, torch.float32).unsqueeze(0).to(device)
    batch["dtype_ids"] = _tensor_from_list(data["dtype_ids"], torch.long).unsqueeze(0).to(device)
    batch["column_ids"] = _tensor_from_list(data["column_ids"], torch.long).unsqueeze(0).to(device)
    batch["row_ids"] = _tensor_from_list(data["row_ids"], torch.long).unsqueeze(0).to(device)
    batch["padding_mask"] = _tensor_from_list(data["padding_mask"], torch.bool).unsqueeze(0).to(device)
    batch["is_masked"] = _tensor_from_list(data["is_masked"], torch.bool).unsqueeze(0).to(device)
    batch["loss_mask"] = _tensor_from_list(data["loss_mask"], torch.bool).unsqueeze(0).to(device)
    batch["target_values"] = _tensor_from_list(data["target_values"], torch.float32).unsqueeze(0).to(device)

    # Attention masks are square matrices
    def attention_tensor(key: str) -> torch.Tensor:
        matrix = data[key]
        return _tensor_from_list(matrix, torch.bool).unsqueeze(0).to(device)

    batch["column_attention"] = attention_tensor("column_attention")
    batch["feature_attention"] = attention_tensor("feature_attention")
    batch["neighbor_attention"] = attention_tensor("neighbor_attention")
    if "temporal_attention" in data:
        batch["temporal_attention"] = attention_tensor("temporal_attention")
    else:
        batch["temporal_attention"] = attention_tensor("full_attention")
    batch["full_attention"] = attention_tensor("full_attention")
    edge_list = data.get("edge_index") or []
    if edge_list:
        batch["edge_index"] = torch.tensor(edge_list, dtype=torch.long).unsqueeze(0).to(device)
        batch["edge_labels"] = torch.tensor(data.get("edge_labels", [1] * len(edge_list)), dtype=torch.float32).unsqueeze(0).to(device)
    else:
        batch["edge_index"] = torch.empty((1, 0, 2), dtype=torch.long, device=device)
        batch["edge_labels"] = torch.empty((1, 0), dtype=torch.float32, device=device)
    batch["seq_len"] = torch.tensor([data["seq_len"]], dtype=torch.long, device=device)
    return batch


def get_redis_client(url: str) -> redis.Redis:
    return redis.Redis.from_url(url)


def ensure_consumer_group(client: redis.Redis, stream_key: str, group: str) -> None:
    try:
        client.xgroup_create(stream_key, group, id="0", mkstream=True)
    except redis.exceptions.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def stream_batches(
    client: redis.Redis,
    stream_key: str,
    group: str,
    consumer: str,
    device: torch.device,
    block_ms: int = 5000,
) -> Iterator[Tuple[str, Dict[str, torch.Tensor]]]:
    """Yield (message ID, batch) tuples from a Redis stream consumer group."""
    ensure_consumer_group(client, stream_key, group)
    while True:
        response = client.xreadgroup(group, consumer, {stream_key: ">"}, count=1, block=block_ms)
        if not response:
            continue
        for _stream, messages in response:
            for msg_id, fields in messages:
                payload = fields.get("payload") or fields.get(b"payload")
                if isinstance(payload, bytes):
                    payload = payload.decode("utf-8")
                if payload is None:
                    client.xack(stream_key, group, msg_id)
                    continue
                batch = deserialize_sample(payload, device=device)
                yield msg_id.decode("utf-8") if isinstance(msg_id, bytes) else msg_id, batch
