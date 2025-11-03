"""Data utilities for the Relational Transformer training pipeline."""

from __future__ import annotations

import math
import random
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# --------------------------------------------------------------------------- #
# Schema specifications and relationships
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ForeignKeySpec:
    """Describes a foreign-key relationship for a child table."""

    parent_table: str
    parent_column: str
    child_column: str
    role: Optional[str] = None


@dataclass
class RelationalTableSpec:
    """Schema and data for a relational table."""

    name: str
    dataframe: pd.DataFrame
    primary_key: str
    timestamp_column: Optional[str] = None
    foreign_keys: Sequence[ForeignKeySpec] = field(default_factory=list)


@dataclass(frozen=True)
class RowHandle:
    """Unique identifier for a row within a table by index."""

    table: str
    row_index: int


@dataclass
class ContextSample:
    """Rows and adjacency information selected for a training context."""

    row_order: List[RowHandle]
    row_to_index: Dict[RowHandle, int]
    parents: Dict[int, Set[int]]
    children: Dict[int, Set[int]]


# --------------------------------------------------------------------------- #
# Database representation and statistics
# --------------------------------------------------------------------------- #


RELATIONAL_DTYPE_TO_ID = {"numeric": 0, "boolean": 1, "datetime": 2, "text": 3}
RELATIONAL_ID_TO_DTYPE = {v: k for k, v in RELATIONAL_DTYPE_TO_ID.items()}


class RelationalDatabase:
    """In-memory representation of a relational database for RT training."""

    def __init__(self, tables: Sequence[RelationalTableSpec]):
        self.tables: Dict[str, RelationalTableSpec] = {}
        self._dataframes: Dict[str, pd.DataFrame] = {}
        self._primary_keys: Dict[str, str] = {}
        self._timestamp_columns: Dict[str, Optional[str]] = {}
        self._foreign_keys: Dict[str, Sequence[ForeignKeySpec]] = {}
        self._pk_to_index: Dict[str, Dict[Union[str, int], int]] = {}
        self._column_ids: Dict[Tuple[str, str], int] = {}
        self._column_types: Dict[Tuple[str, str], str] = {}
        self._column_stats: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self._datetime_values: List[float] = []
        self._timestamps: Dict[RowHandle, float] = {}
        self._parents: Dict[RowHandle, Set[RowHandle]] = defaultdict(set)
        self._children: Dict[RowHandle, Set[RowHandle]] = defaultdict(set)
        self._column_roles: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self._next_column_id = 0

        for table in tables:
            self.add_table(table)

        self._finalize_datetime_stats()
        self._build_relationships()

    # ------------------------------------------------------------------ #
    # Table ingestion and statistics
    # ------------------------------------------------------------------ #
    def add_table(self, spec: RelationalTableSpec) -> None:
        """Register a table and compute column statistics."""
        name = spec.name
        if name in self.tables:
            raise ValueError(f"Duplicate table registration: {name}")

        df = spec.dataframe.reset_index(drop=True)
        if spec.primary_key not in df.columns:
            raise ValueError(f"Primary key {spec.primary_key} missing in table {name}")

        self.tables[name] = spec
        self._dataframes[name] = df
        self._primary_keys[name] = spec.primary_key
        self._timestamp_columns[name] = spec.timestamp_column
        self._foreign_keys[name] = list(spec.foreign_keys)

        for column in df.columns:
            self._column_roles[(name, column)]
        self._add_column_role(name, spec.primary_key, 'primary_key')

        pk_map: Dict[Union[str, int], int] = {}
        for idx, value in enumerate(df[spec.primary_key].tolist()):
            pk_map[value] = idx
            handle = RowHandle(name, idx)
            timestamp = self._extract_timestamp(name, idx)
            if timestamp is not None:
                self._timestamps[handle] = timestamp
        self._pk_to_index[name] = pk_map

        for column in df.columns:
            column_key = (name, column)
            if column_key not in self._column_ids:
                self._column_ids[column_key] = self._next_column_id
                self._next_column_id += 1

            dtype = self._infer_column_dtype(df[column])
            self._column_types[column_key] = dtype
            if dtype in {"numeric", "boolean"}:
                series = pd.to_numeric(df[column], errors="coerce")
                mean = float(series.mean())
                std = float(series.std()) if not math.isclose(float(series.std()), 0.0) else 1.0
                if math.isnan(mean):
                    mean = 0.0
                if math.isnan(std) or math.isclose(std, 0.0):
                    std = 1.0
                self._column_stats[column_key] = (mean, std)
            elif dtype == "datetime":
                series = pd.to_datetime(df[column], errors="coerce")
                seconds = series.view("int64") / 1e9
                seconds = seconds.replace([np.inf, -np.inf], np.nan)
                seconds = seconds.dropna()
                self._datetime_values.extend(seconds.tolist())
                mean = float(seconds.mean()) if len(seconds) else 0.0
                std = float(seconds.std()) if len(seconds) else 1.0
                if math.isnan(mean):
                    mean = 0.0
                if math.isnan(std) or math.isclose(std, 0.0):
                    std = 1.0
                self._column_stats[column_key] = (mean, std)
            else:
                # Text columns do not require numerical stats
                self._column_stats[column_key] = (0.0, 1.0)

    def _finalize_datetime_stats(self) -> None:
        if not self._datetime_values:
            self._datetime_global_mean = 0.0
            self._datetime_global_std = 1.0
            return
        arr = np.array(self._datetime_values, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std()) if arr.std() > 0 else 1.0
        if math.isnan(mean):
            mean = 0.0
        if math.isnan(std) or math.isclose(std, 0.0):
            std = 1.0
        self._datetime_global_mean = mean
        self._datetime_global_std = std

    def _build_relationships(self) -> None:
        """Materialize parent/child adjacency using foreign keys."""
        for child_name, fks in self._foreign_keys.items():
            child_df = self._dataframes[child_name]
            child_pk = self._primary_keys[child_name]
            for fk in fks:
                parent_name = fk.parent_table
                if parent_name not in self.tables:
                    raise ValueError(f"Unknown parent table '{parent_name}' in FK for {child_name}")

                role_label = fk.role or f"fk:{child_name}.{fk.child_column}->{parent_name}.{fk.parent_column}"
                self._add_column_role(child_name, fk.child_column, role_label)
                self._add_column_role(child_name, fk.child_column, 'foreign_key')
                self._add_column_role(parent_name, fk.parent_column, f"referenced_by:{child_name}.{fk.child_column}")

                parent_pk_map = self._pk_to_index[parent_name]
                for idx, row in child_df.iterrows():
                    parent_key = row.get(fk.child_column)
                    if pd.isna(parent_key):
                        continue
                    parent_idx = parent_pk_map.get(parent_key)
                    if parent_idx is None:
                        continue
                    parent_handle = RowHandle(parent_name, parent_idx)
                    child_handle = RowHandle(child_name, idx)
                    self._parents[child_handle].add(parent_handle)
                    self._children[parent_handle].add(child_handle)

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    def resolve_row(self, table: str, primary_key_value: Union[int, str]) -> RowHandle:
        if table not in self.tables:
            raise KeyError(f"Unknown table {table}")
        pk_map = self._pk_to_index[table]
        if primary_key_value not in pk_map:
            raise KeyError(f"Primary key {primary_key_value} not found in table {table}")
        return RowHandle(table, pk_map[primary_key_value])

    def get_row(self, handle: RowHandle) -> pd.Series:
        return self._dataframes[handle.table].iloc[handle.row_index]

    def get_timestamp(self, handle: RowHandle) -> Optional[float]:
        return self._timestamps.get(handle)

    def parents(self, handle: RowHandle) -> Set[RowHandle]:
        return self._parents.get(handle, set())

    def children(self, handle: RowHandle) -> Set[RowHandle]:
        return self._children.get(handle, set())

    def column_id(self, table: str, column: str) -> int:
        return self._column_ids[(table, column)]

    def column_dtype(self, table: str, column: str) -> str:
        return self._column_types[(table, column)]

    def column_stats(self, table: str, column: str) -> Tuple[float, float]:
        return self._column_stats[(table, column)]

    def datetime_stats(self) -> Tuple[float, float]:
        return self._datetime_global_mean, self._datetime_global_std

    def dataframe(self, table: str) -> pd.DataFrame:
        if table not in self._dataframes:
            raise KeyError(f"Unknown table {table}")
        return self._dataframes[table]

    def primary_key(self, table: str) -> str:
        if table not in self._primary_keys:
            raise KeyError(f"Unknown table {table}")
        return self._primary_keys[table]

    def _add_column_role(self, table: str, column: str, role: str) -> None:
        roles = self._column_roles[(table, column)]
        if role not in roles:
            roles.append(role)

    def column_roles(self, table: str, column: str) -> List[str]:
        return list(self._column_roles.get((table, column), []))

    def _extract_timestamp(self, table: str, row_index: int) -> Optional[float]:
        column = self._timestamp_columns.get(table)
        if not column:
            return None
        value = self._dataframes[table].iloc[row_index][column]
        if pd.isna(value):
            return None
        dt_value = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt_value):
            return None
        return float(dt_value.value / 1e9)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _infer_column_dtype(column: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(column):
            return "boolean"
        if pd.api.types.is_numeric_dtype(column):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(column):
            return "datetime"
        # Attempt to parse datetimes even if dtype is object
        sample = pd.to_datetime(column, errors="coerce")
        if sample.notna().sum() > 0.7 * len(column):
            return "datetime"
        return "text"


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #


class FrozenTextEncoder:
    """Wrapper around a frozen Transformer encoder."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.squeeze(0).cpu()


@dataclass
class TokenizedCell:
    """Token representation for a single table cell."""

    table: str
    row_index: int
    row_id: int
    column: str
    column_id: int
    dtype: str
    dtype_id: int
    value_embedding: torch.Tensor
    schema_embedding: torch.Tensor
    temporal_embedding: torch.Tensor
    role_embedding: torch.Tensor
    scalar_target: Optional[float]
    timestamp: Optional[float]
    key_value: Optional[str] = None
    parent_timestamp: Optional[float] = None
    is_masked: bool = False


class CellTokenizer:
    """Constructs RT token embeddings for relational cells."""

    def __init__(
        self,
        database: RelationalDatabase,
        text_encoder: Optional[FrozenTextEncoder] = None,
        value_dim: int = 384,
        schema_dim: int = 384,
        temporal_dim: int = 5,
        role_dim: int = 64,
        schema_seed: int = 0,
    ):
        self.database = database
        self.text_encoder = text_encoder
        self.value_dim = value_dim
        self.schema_dim = schema_dim
        self.temporal_dim = temporal_dim
        self.role_dim = role_dim
        self.schema_seed = schema_seed
        self._schema_cache: Dict[Tuple[str, str], torch.Tensor] = {}
        self._role_cache: Dict[str, torch.Tensor] = {}

    def tokenize_row(
        self,
        handle: RowHandle,
        row_id: int,
        mask_columns: Optional[Set[str]] = None,
        seed_timestamp: Optional[float] = None,
    ) -> List[TokenizedCell]:
        row = self.database.get_row(handle)
        mask_columns = mask_columns or set()
        tokens: List[TokenizedCell] = []
        row_timestamp = self.database.get_timestamp(handle)

        for column, value in row.items():
            if pd.isna(value):
                continue

            column_id = self.database.column_id(handle.table, column)
            dtype = self.database.column_dtype(handle.table, column)
            dtype_id = RELATIONAL_DTYPE_TO_ID[dtype]

            value_vec, scalar_target = self._encode_value(handle.table, column, dtype, value)
            schema_vec = self._encode_schema(handle.table, column)
            temporal_vec = self._temporal_embedding(row_timestamp, seed_timestamp)
            role_vec = self._role_embedding(handle.table, column)

            is_masked = column in mask_columns

            tokens.append(
                TokenizedCell(
                    table=handle.table,
                    row_index=handle.row_index,
                    row_id=row_id,
                    column=column,
                    column_id=column_id,
                    dtype=dtype,
                    dtype_id=dtype_id,
                    value_embedding=value_vec,
                    schema_embedding=schema_vec,
                    temporal_embedding=temporal_vec,
                    role_embedding=role_vec,
                    scalar_target=scalar_target,
                    timestamp=row_timestamp,
                    is_masked=is_masked,
                )
            )
        return tokens

    def _encode_value(
        self,
        table: str,
        column: str,
        dtype: str,
        value: object,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        vec = torch.zeros(self.value_dim, dtype=torch.float32)
        if dtype == "numeric":
            mean, std = self.database.column_stats(table, column)
            numeric = float(value)
            normalized = (numeric - mean) / std
            vec[0] = normalized
            return vec, normalized
        if dtype == "boolean":
            normalized = float(bool(value))
            vec[0] = normalized
            return vec, normalized
        if dtype == "datetime":
            dt = pd.to_datetime(value, errors="coerce")
            if pd.isna(dt):
                normalized = 0.0
            else:
                seconds = float(dt.value / 1e9)
                mean, std = self.database.datetime_stats()
                normalized = (seconds - mean) / std
            vec[0] = normalized
            return vec, normalized

        # Text value
        if self.text_encoder is not None:
            embedding = self.text_encoder.encode(str(value))
            vec = self._fit_dimension(embedding, self.value_dim)
        else:
            vec = self._hashed_embedding(f"value:{table}.{column}:{value}", self.value_dim)
        return vec, None

    def _encode_schema(self, table: str, column: str) -> torch.Tensor:
        key = (table, column)
        if key in self._schema_cache:
            return self._schema_cache[key]
        phrase = f"{column} of {table}"
        if self.text_encoder is not None:
            embedding = self.text_encoder.encode(phrase)
            fitted = self._fit_dimension(embedding, self.schema_dim)
        else:
            fitted = self._hashed_embedding(f"schema:{phrase}", self.schema_dim)
        self._schema_cache[key] = fitted
        return fitted

    @staticmethod
    def _fit_dimension(vector: torch.Tensor, dim: int) -> torch.Tensor:
        if vector.shape[0] == dim:
            return vector.clone().detach()
        if vector.shape[0] > dim:
            return vector[:dim].clone().detach()
        padded = torch.zeros(dim, dtype=torch.float32)
        padded[: vector.shape[0]] = vector
        return padded

    def _temporal_embedding(
        self,
        timestamp: Optional[float],
        seed_timestamp: Optional[float],
    ) -> torch.Tensor:
        vec = torch.zeros(self.temporal_dim, dtype=torch.float32)
        if timestamp is None:
            return vec

        delta_hours = 0.0
        if seed_timestamp is not None:
            delta_hours = (timestamp - seed_timestamp) / 3600.0
        vec[0] = float(delta_hours)

        dt = pd.Timestamp.utcfromtimestamp(timestamp)
        day_fraction = dt.weekday() / 7.0
        seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        hour_fraction = seconds / 86400.0

        vec[1] = float(np.sin(2 * np.pi * day_fraction))
        vec[2] = float(np.cos(2 * np.pi * day_fraction))
        vec[3] = float(np.sin(2 * np.pi * hour_fraction))
        vec[4] = float(np.cos(2 * np.pi * hour_fraction))

        return vec

    def _role_embedding(self, table: str, column: str) -> torch.Tensor:
        roles = self.database.column_roles(table, column)
        if not roles:
            return torch.zeros(self.role_dim, dtype=torch.float32)
        accum = torch.zeros(self.role_dim, dtype=torch.float32)
        for role in roles:
            key = f"role:{role}"
            cached = self._role_cache.get(key)
            if cached is None:
                cached = self._hashed_embedding(key, self.role_dim)
                self._role_cache[key] = cached
            accum += cached
        accum /= float(len(roles))
        return accum

    def _hashed_embedding(self, key: str, dim: int) -> torch.Tensor:
        seed_material = f"{key}|{self.schema_seed}".encode("utf-8")
        digest = hashlib.sha256(seed_material).digest()
        values: List[int] = []
        while len(values) < dim:
            digest = hashlib.sha256(digest).digest()
            for i in range(0, len(digest), 4):
                chunk = digest[i : i + 4]
                if len(chunk) < 4:
                    continue
                integer = int.from_bytes(chunk, byteorder="little", signed=False)
                values.append(integer)
                if len(values) >= dim:
                    break

        tensor = torch.zeros(dim, dtype=torch.float32)
        for idx in range(dim):
            raw = values[idx]
            tensor[idx] = (raw / 2**32) * 2.0 - 1.0
        return tensor


# --------------------------------------------------------------------------- #
# Context sampling
# --------------------------------------------------------------------------- #


class ContextSampler:
    """Samples relational neighborhoods following the RT BFS procedure."""

    def __init__(
        self,
        database: RelationalDatabase,
        max_context_cells: int = 1024,
        width_bound: int = 8,
        random_state: Optional[int] = None,
    ):
        self.database = database
        self.max_context_cells = max_context_cells
        self.width_bound = width_bound
        self.random = random.Random(random_state)

    def sample(
        self,
        seed: Union[RowHandle, Tuple[str, Union[int, str]]],
        timestamp_cutoff: bool = True,
    ) -> ContextSample:
        """Return a BFS context seeded at the target row."""
        if isinstance(seed, tuple):
            table, pk_value = seed
            handle = self.database.resolve_row(table, pk_value)
        else:
            handle = seed

        seed_timestamp = self.database.get_timestamp(handle) if timestamp_cutoff else None
        visited: Set[RowHandle] = set()
        queue: deque[RowHandle] = deque([handle])
        row_order: List[RowHandle] = []

        while queue and len(row_order) < self.max_context_cells:
            current = queue.popleft()
            if current in visited:
                continue
            current_timestamp = self.database.get_timestamp(current)
            if seed_timestamp is not None and current_timestamp is not None and current_timestamp > seed_timestamp:
                continue

            visited.add(current)
            row_order.append(current)

            # Ensure all parents are included
            for parent in self.database.parents(current):
                if parent not in visited:
                    queue.appendleft(parent)

            # Sample children with width constraint
            children = list(self.database.children(current))
            self.random.shuffle(children)
            for child in children[: self.width_bound]:
                if child not in visited:
                    queue.append(child)

            if len(row_order) >= self.max_context_cells:
                break

        row_to_index = {handle: idx for idx, handle in enumerate(row_order)}
        parents: Dict[int, Set[int]] = defaultdict(set)
        children: Dict[int, Set[int]] = defaultdict(set)

        for handle in row_order:
            idx = row_to_index[handle]
            for parent in self.database.parents(handle):
                if parent in row_to_index:
                    parents[idx].add(row_to_index[parent])
            for child in self.database.children(handle):
                if child in row_to_index:
                    children[idx].add(row_to_index[child])

        return ContextSample(
            row_order=row_order,
            row_to_index=row_to_index,
            parents=parents,
            children=children,
        )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


@dataclass
class TargetSpec:
    table: str
    primary_key_value: Union[int, str]
    column: str


class RelationalDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapping RT tokenization and context sampling."""

    def __init__(
        self,
        database: RelationalDatabase,
        sampler: ContextSampler,
        tokenizer: CellTokenizer,
        targets: Sequence[TargetSpec],
        context_cells: int = 1024,
        mask_probability: float = 0.15,
        include_text: bool = False,
        allow_temporal_leakage: bool = False,
        temporal_lookback_seconds: Optional[float] = None,
    ):
        self.database = database
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.targets = list(targets)
        self.context_cells = context_cells
        self.mask_probability = mask_probability
        self.include_text = include_text
        self.allow_temporal_leakage = allow_temporal_leakage
        self.temporal_lookback_seconds = temporal_lookback_seconds

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        target = self.targets[idx]
        seed = self.database.resolve_row(target.table, target.primary_key_value)
        context = self.sampler.sample(seed, timestamp_cutoff=not self.allow_temporal_leakage)

        tokens: List[TokenizedCell] = []
        total_cells = 0
        target_token_index: Optional[int] = None
        seed_timestamp = self.database.get_timestamp(seed)

        for handle in context.row_order:
            row_id = context.row_to_index[handle]
            mask_columns = {target.column} if handle == seed else set()
            row_tokens = self.tokenizer.tokenize_row(
                handle,
                row_id=row_id,
                mask_columns=mask_columns,
                seed_timestamp=seed_timestamp,
            )

            for token in row_tokens:
                if not self.include_text and token.dtype == "text":
                    continue
                tokens.append(token)
                if handle == seed and token.column == target.column:
                    target_token_index = len(tokens) - 1
                total_cells += 1
                if total_cells >= self.context_cells:
                    break
            if total_cells >= self.context_cells:
                break

        if target_token_index is None:
            raise RuntimeError("Target column not present in context window.")

        tokens = tokens[: self.context_cells]
        seq_len = len(tokens)

        value_dim = self.tokenizer.value_dim
        schema_dim = self.tokenizer.schema_dim
        temporal_dim = self.tokenizer.temporal_dim
        role_dim = self.tokenizer.role_dim
        context_len = self.context_cells

        value_embeddings = torch.zeros(context_len, value_dim, dtype=torch.float32)
        schema_embeddings = torch.zeros(context_len, schema_dim, dtype=torch.float32)
        temporal_embeddings = torch.zeros(context_len, temporal_dim, dtype=torch.float32)
        role_embeddings = torch.zeros(context_len, role_dim, dtype=torch.float32)
        dtype_ids = torch.zeros(context_len, dtype=torch.long)
        column_ids = torch.zeros(context_len, dtype=torch.long)
        row_ids = torch.zeros(context_len, dtype=torch.long)
        padding_mask = torch.zeros(context_len, dtype=torch.bool)
        is_masked = torch.zeros(context_len, dtype=torch.bool)
        loss_mask = torch.zeros(context_len, dtype=torch.bool)
        target_values = torch.zeros(context_len, dtype=torch.float32)
        timestamps: List[Optional[float]] = [None] * context_len

        # Build column/feature/neighbor masks
        column_mask = torch.zeros(context_len, context_len, dtype=torch.bool)
        feature_mask = torch.zeros(context_len, context_len, dtype=torch.bool)
        neighbor_mask = torch.zeros(context_len, context_len, dtype=torch.bool)
        full_mask = torch.zeros(context_len, context_len, dtype=torch.bool)
        temporal_mask = torch.zeros(context_len, context_len, dtype=torch.bool)

        for i, token in enumerate(tokens):
            value_embeddings[i] = token.value_embedding
            schema_embeddings[i] = token.schema_embedding
            temporal_embeddings[i] = token.temporal_embedding
            role_embeddings[i] = token.role_embedding
            dtype_ids[i] = token.dtype_id
            column_ids[i] = token.column_id
            row_ids[i] = token.row_id
            padding_mask[i] = True
            is_masked[i] = token.is_masked
            timestamps[i] = token.timestamp

            if token.scalar_target is not None and token.is_masked:
                loss_mask[i] = True
                target_values[i] = float(token.scalar_target)

        # Randomly mask additional cells for MTP (excluding text and primary target)
        for i, token in enumerate(tokens):
            if i == target_token_index:
                continue
            if token.dtype == "text":
                continue
            if token.scalar_target is None:
                continue
            if random.random() < self.mask_probability:
                is_masked[i] = True
                loss_mask[i] = True
                target_values[i] = float(token.scalar_target)

        positive_edges = set()
        for child_idx, parents in context.parents.items():
            for parent_idx in parents:
                if child_idx < seq_len and parent_idx < seq_len:
                    positive_edges.add((child_idx, parent_idx))
                    positive_edges.add((parent_idx, child_idx))

        negative_edges = set()
        if seq_len >= 2:
            max_negatives = max(len(positive_edges), 1)
            attempts = 0
            while len(negative_edges) < max_negatives and attempts < 10_000:
                attempts += 1
                a = random.randrange(seq_len)
                b = random.randrange(seq_len)
                if a == b or (a, b) in positive_edges:
                    continue
                negative_edges.add((a, b))

        edge_list = list(positive_edges) + list(negative_edges)
        edge_labels = [1] * len(positive_edges) + [0] * len(negative_edges)
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long)
            edge_label_tensor = torch.tensor(edge_labels, dtype=torch.float32)
        else:
            edge_index = torch.empty((0, 2), dtype=torch.long)
            edge_label_tensor = torch.empty((0,), dtype=torch.float32)

        for i in range(seq_len):
            for j in range(seq_len):
                same_column = column_ids[i] == column_ids[j]
                row_i = int(row_ids[i].item())
                row_j = int(row_ids[j].item())
                same_row = row_i == row_j
                parent_rows = context.parents.get(row_i, set())
                child_rows = context.children.get(row_j, set())

                column_mask[i, j] = bool(same_column)
                feature_mask[i, j] = bool(same_row or row_j in parent_rows)
                neighbor_mask[i, j] = bool(row_i in child_rows)
                full_mask[i, j] = True

                allow_temporal = True
                ts_i = timestamps[i]
                ts_j = timestamps[j]
                if ts_i is not None and ts_j is not None:
                    if ts_j > ts_i:
                        allow_temporal = False
                    if (
                        allow_temporal
                        and self.temporal_lookback_seconds is not None
                        and (ts_i - ts_j) > self.temporal_lookback_seconds
                    ):
                        allow_temporal = False
                temporal_mask[i, j] = bool(allow_temporal)

        return {
            "value_embeddings": value_embeddings,
            "schema_embeddings": schema_embeddings,
            "temporal_embeddings": temporal_embeddings,
            "role_embeddings": role_embeddings,
            "dtype_ids": dtype_ids,
            "column_ids": column_ids,
            "row_ids": row_ids,
            "padding_mask": padding_mask,
            "is_masked": is_masked,
            "loss_mask": loss_mask,
            "target_values": target_values,
            "column_attention": column_mask,
            "feature_attention": feature_mask,
            "neighbor_attention": neighbor_mask,
            "temporal_attention": temporal_mask,
            "full_attention": full_mask,
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
        }
