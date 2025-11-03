#!/usr/bin/env python3
"""
Utilities for building SGMI lineage payloads.

This module is invoked by ``run_sgmi_full_graph.sh`` but can also be imported
from tests to verify view parsing, type inference, join capture, and summary
metrics in isolation.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path
from typing import Iterable, Sequence

import sys
import tempfile

from simple_ddl_parser import DDLParser
from sqlglot import expressions as exp
from sqlglot import parse_one

NUMERIC_TYPES = {
    "INT",
    "INTEGER",
    "BIGINT",
    "SMALLINT",
    "TINYINT",
    "DOUBLE",
    "FLOAT",
    "DECIMAL",
    "NUMERIC",
    "REAL",
}
BOOLEAN_TYPES = {"BOOL", "BOOLEAN"}

VIEW_REGEX = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+`?([\w\.]+)`?\s+AS\s+(.*)",
    re.IGNORECASE | re.DOTALL,
)

COMMENT_REGEX = re.compile(r"--.*?$", re.MULTILINE)


@dataclass(frozen=True)
class SourceRef:
    table: str
    column: str

    def as_dict(self) -> dict[str, str]:
        return {"table": self.table, "column": self.column}


def _normalise_name(value: str | None) -> str:
    if not value:
        return ""
    return value.strip("`").strip().lower()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _split_statements(text: str) -> list[str]:
    cleaned = COMMENT_REGEX.sub("", text)
    return [stmt.strip() for stmt in cleaned.split(";") if stmt.strip()]


def _is_create_table(stmt_upper: str) -> bool:
    return stmt_upper.startswith("CREATE TABLE") or stmt_upper.startswith("CREATE EXTERNAL TABLE")


def _sanitize_table_statement(statement: str) -> str:
    lines: list[str] = []
    for raw_line in statement.splitlines():
        line = raw_line.strip()
        upper = line.upper()
        if not line:
            continue
        if upper.startswith(("CREATE DATABASE", "USE ")):
            continue
        if upper.startswith(
            (
                "ROW FORMAT",
                "STORED AS",
                "OUTPUTFORMAT",
                "LOCATION",
                "TBLPROPERTIES",
                "WITH SERDEPROPERTIES",
            )
        ):
            continue
        if upper.startswith("'ORG.APACHE") or upper.startswith("'HDFS://"):
            continue
        lines.append(raw_line)
    return "\n".join(lines)


def _collect_definitions(ddl_paths: Sequence[Path]) -> tuple[list[str], list[dict[str, str]]]:
    table_statements: list[str] = []
    view_definitions: list[dict[str, str]] = []

    for ddl_path in ddl_paths:
        for statement in _split_statements(_read_text(ddl_path)):
            upper_stmt = statement.upper()
            if "CREATE VIEW" in upper_stmt:
                match = VIEW_REGEX.match(statement)
                if match:
                    view_definitions.append({"name": match.group(1), "select": match.group(2)})
                continue
            if _is_create_table(upper_stmt):
                sanitized = _sanitize_table_statement(statement)
                if sanitized:
                    table_statements.append(f"{sanitized};")
    return table_statements, view_definitions


def _build_column_type_lookup(statements: list[str]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    if not statements:
        return lookup
    try:
        parsed_tables = DDLParser("\n".join(statements)).run()
    except Exception:
        return lookup

    for table in parsed_tables or []:
        raw_table = (table.get("schema") or "") + "." if table.get("schema") else ""
        raw_table += table.get("table_name", "") or ""
        table_variants = {_normalise_name(raw_table)}
        simple_name = _normalise_name(table.get("table_name"))
        if simple_name:
            table_variants.add(simple_name)
        for column in table.get("columns", []):
            col_name = _normalise_name(column.get("name"))
            dtype = (column.get("type") or "STRING").upper()
            for variant in table_variants:
                if variant:
                    lookup[(variant, col_name)] = dtype
    return lookup


def _resolve_type(sources: Iterable[SourceRef], lookup: dict[tuple[str, str], str]) -> str:
    for source in sources:
        table = _normalise_name(source.table)
        column = _normalise_name(source.column)
        if (table, column) in lookup:
            return lookup[(table, column)]
    return "STRING"


def _infer_literal_type(literal: exp.Literal) -> str:
    value = literal.this
    if literal.is_string:
        return "STRING"
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return "BOOLEAN"
    if literal.is_number or isinstance(value, (int, float)):
        text = str(value or "")
        return "DECIMAL" if "." in text else "BIGINT"
    return "STRING"


def _merge_numeric(left: str, right: str) -> str:
    if left in NUMERIC_TYPES and right in NUMERIC_TYPES:
        return "DECIMAL"
    if left in NUMERIC_TYPES:
        return left
    if right in NUMERIC_TYPES:
        return right
    return left or right or "STRING"


def _infer_expression_type(expr: exp.Expression | None) -> str:
    if expr is None:
        return "STRING"
    if isinstance(expr, exp.Literal):
        return _infer_literal_type(expr)
    if isinstance(expr, exp.Column):
        return "STRING"
    if isinstance(expr, exp.Cast):
        target = expr.args.get("to")
        if target is not None:
            return target.sql(dialect="hive").upper()
        return _infer_expression_type(expr.this)
    if isinstance(expr, exp.Func):
        func_name = (expr.sql_name() or "").upper()
        if func_name in {"COUNT", "COUNT_DISTINCT"}:
            return "BIGINT"
        if func_name in {"SUM", "AVG"}:
            return "DECIMAL"
        if func_name in {"MAX", "MIN"}:
            first_arg = next(iter(expr.args.get("expressions") or []), None)
            if first_arg is not None:
                return _infer_expression_type(first_arg)
        if func_name in {"COALESCE", "NVL"}:
            for arg in expr.args.get("expressions") or []:
                inferred = _infer_expression_type(arg)
                if inferred != "STRING":
                    return inferred
    if isinstance(expr, exp.Case):
        branches = [expr.args.get("default"), *expr.args.get("ifs", [])]
        inferred = {_infer_expression_type(branch) for branch in branches if branch is not None}
        if inferred:
            if any(t in NUMERIC_TYPES for t in inferred):
                return "DECIMAL"
            if any(t in BOOLEAN_TYPES for t in inferred):
                return "BOOLEAN"
    if isinstance(expr, exp.Binary):
        left_type = _infer_expression_type(expr.left)
        right_type = _infer_expression_type(expr.right)
        if left_type == right_type:
            return left_type
        return _merge_numeric(left_type, right_type)
    if isinstance(expr, exp.Paren):
        return _infer_expression_type(expr.this)
    return "STRING"


def _alias_map_from_tables(select: exp.Select) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for table in select.find_all(exp.Table):
        actual = table.name
        alias_expr = table.args.get("alias")
        alias = alias_expr.name if alias_expr else actual
        if actual and alias:
            alias_map[_normalise_name(alias)] = actual
    return alias_map


def _collect_tables(select: exp.Select, alias_map: dict[str, str]) -> list[dict[str, str]]:
    tables: dict[str, dict[str, str]] = {}
    from_expr = select.args.get("from")
    if isinstance(from_expr, exp.From):
        base = from_expr.this
        alias = (base.alias_or_name or base.sql()).replace("`", "")
        tables[_normalise_name(alias)] = {
            "alias": alias,
            "table": alias_map.get(_normalise_name(alias), base.name),
        }
    for join in select.args.get("joins") or []:
        table_expr = join.this
        alias = (table_expr.alias_or_name or table_expr.sql()).replace("`", "")
        tables[_normalise_name(alias)] = {
            "alias": alias,
            "table": alias_map.get(_normalise_name(alias), table_expr.name),
        }
    return list(tables.values())


def _collect_join_details(select: exp.Select, alias_map: dict[str, str]) -> list[dict[str, object]]:
    joins: list[dict[str, object]] = []
    for join in select.args.get("joins") or []:
        join_type = (join.args.get("kind") or "INNER").upper()
        table_expr = join.this
        alias = (table_expr.alias_or_name or table_expr.sql()).replace("`", "")
        condition_expr = join.args.get("on")
        condition_sql = condition_expr.sql(dialect="hive") if condition_expr else ""
        condition_columns: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        if condition_expr is not None:
            for column in condition_expr.find_all(exp.Column):
                alias_name = (column.table or alias or "").replace("`", "")
                table_name = alias_map.get(_normalise_name(alias_name), alias_name)
                column_name = column.name or ""
                key = (_normalise_name(table_name), _normalise_name(column_name))
                if key in seen:
                    continue
                condition_columns.append(
                    {
                        "alias": alias_name,
                        "table": table_name,
                        "column": column_name,
                    }
                )
                seen.add(key)
        joins.append(
            {
                "type": join_type,
                "right_alias": alias,
                "right_table": alias_map.get(_normalise_name(alias), table_expr.name),
                "condition": condition_sql,
                "condition_columns": condition_columns,
            }
        )
    return joins


def _sources_from_projection(
    projection: exp.Expression,
    alias_map: dict[str, str],
    default_source: str,
) -> list[SourceRef]:
    sources: list[SourceRef] = []
    for column in projection.find_all(exp.Column):
        table_part = column.table or default_source
        actual_table = alias_map.get(_normalise_name(table_part), table_part)
        column_name = column.name or ""
        sources.append(SourceRef(actual_table, column_name))
    return sources


def _column_info(
    select: exp.Select,
    lookup: dict[tuple[str, str], str],
) -> tuple[list[dict[str, object]], set[str], list[dict[str, object]], list[dict[str, str]]]:
    alias_map = _alias_map_from_tables(select)
    default_source = next(iter(alias_map.values()), "")
    columns_info: list[dict[str, object]] = []
    source_tables: set[str] = set()

    expressions = select.expressions or []
    for idx, projection in enumerate(expressions):
        alias = projection.alias_or_name or f"col_{idx + 1}"
        alias = alias.replace("`", "")
        inner_expr = projection.this if isinstance(projection, exp.Alias) else projection
        sources = _sources_from_projection(inner_expr or projection, alias_map, default_source)
        for source in sources:
            if source.table:
                source_tables.add(source.table)
        resolved_type = _resolve_type(sources, lookup)
        if resolved_type == "STRING" or not sources:
            inferred_type = _infer_expression_type(inner_expr)
            if inferred_type != "STRING" or not sources:
                resolved_type = inferred_type

        columns_info.append(
            {
                "name": alias,
                "type": resolved_type,
                "sources": [source.as_dict() for source in sources],
            }
        )
    joins = _collect_join_details(select, alias_map)
    table_details = _collect_tables(select, alias_map)
    return columns_info, source_tables, joins, table_details


def _column_metrics(
    columns_info: Sequence[dict[str, object]],
    joins: Sequence[dict[str, object]],
    source_tables: Iterable[str],
) -> dict[str, float | int]:
    total_columns = len(columns_info)
    mapped_columns = sum(1 for column in columns_info if column.get("sources"))
    typed_columns = sum(
        1 for column in columns_info if (column.get("type") or "").upper() not in {"STRING", "UNKNOWN"}
    )
    referenced_tables = {
        _normalise_name(source.get("table"))
        for column in columns_info
        for source in column.get("sources", [])
    }
    all_tables = {_normalise_name(table) for table in source_tables if table}
    info_loss = 1.0 - (mapped_columns / total_columns) if total_columns else 0.0
    type_coverage = typed_columns / total_columns if total_columns else 0.0
    similarity = (
        len(referenced_tables.intersection(all_tables)) / len(all_tables)
        if all_tables
        else (1.0 if not referenced_tables else 0.0)
    )
    return {
        "column_count": total_columns,
        "columns_with_sources": mapped_columns,
        "columns_with_types": typed_columns,
        "information_loss": round(info_loss, 3),
        "type_coverage": round(type_coverage, 3),
        "source_similarity": round(similarity, 3),
        "join_count": len(joins),
    }


def _write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _build_summary(view_lineage: Sequence[dict[str, object]]) -> dict[str, object]:
    metrics = [entry.get("metrics") or {} for entry in view_lineage]
    if not metrics:
        summary_metrics = {}
    else:
        def average(key: str) -> float:
            values = [m.get(key) for m in metrics if isinstance(m.get(key), (int, float))]
            return round(statistics.mean(values), 3) if values else 0.0

        summary_metrics = {
            "view_count": len(metrics),
            "total_columns": int(sum(m.get("column_count", 0) for m in metrics)),
            "avg_information_loss": average("information_loss"),
            "avg_type_coverage": average("type_coverage"),
            "avg_source_similarity": average("source_similarity"),
            "views_with_joins": sum(1 for m in metrics if (m.get("join_count") or 0) > 0),
            "views_missing_sources": sum(
                1 for m in metrics if (m.get("columns_with_sources") or 0) < (m.get("column_count") or 0)
            ),
        }
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "metrics": summary_metrics,
        "notes": "Derived from sgmi_view_builder.py; regenerated on every SGMI ingest run.",
    }


def build_payload(
    json_paths: Sequence[Path],
    ddl_paths: Sequence[Path],
    xml_paths: Sequence[Path],
    view_tmp_dir: Path | None,
    view_registry_out: Path | None,
    view_summary_out: Path | None,
) -> dict[str, object]:
    table_statements, view_definitions = _collect_definitions(ddl_paths)
    column_type_lookup = _build_column_type_lookup(table_statements)

    view_sql_queries: list[str] = []
    view_json_paths: list[Path] = []
    view_lineage: list[dict[str, object]] = []

    for view_def in view_definitions:
        view_name = view_def["name"]
        select_sql = view_def["select"].strip().rstrip(";")
        if not select_sql:
            continue
        try:
            parsed = parse_one(select_sql, read="hive")
        except Exception:
            continue
        if not isinstance(parsed, exp.Select):
            continue

        columns_info, source_tables, joins, table_details = _column_info(parsed, column_type_lookup)
        if not columns_info:
            continue

        create_columns = ",\n".join(f"  `{col['name']}` {col['type']}" for col in columns_info)
        table_statements.append(f"CREATE TABLE `{view_name}` (\n{create_columns}\n); -- view")

        if view_tmp_dir:
            view_tmp_dir.mkdir(parents=True, exist_ok=True)
            sample_row = [{col["name"]: "" for col in columns_info}]
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f"sgmi_view_{view_name}_",
                suffix=".json",
                delete=False,
                dir=view_tmp_dir,
            ) as handle:
                json.dump(sample_row, handle)
                tmp_file = Path(handle.name)
            view_json_paths.append(tmp_file)

        view_sql_queries.append(f"INSERT INTO {view_name} {select_sql}")
        metrics = _column_metrics(columns_info, joins, source_tables)
        view_lineage.append(
            {
                "view": view_name,
                "select": select_sql,
                "select_hash": md5(select_sql.encode("utf-8")).hexdigest(),
                "columns": columns_info,
                "source_tables": sorted(set(source_tables)),
                "joins": joins,
                "tables": table_details,
                "metrics": metrics,
            }
        )

    if view_registry_out:
        _write_json_atomic(Path(view_registry_out), view_lineage)

    if view_summary_out:
        summary = _build_summary(view_lineage)
        _write_json_atomic(Path(view_summary_out), summary)

    payload = {
        "json_tables": [str(path) for path in json_paths + view_json_paths],
        "hive_ddls": table_statements,
        "sql_queries": view_sql_queries,
        "control_m_files": [str(path) for path in xml_paths],
        "project_id": "sgmi-full",
        "system_id": "sgmi-full-system",
        "information_system_id": "sgmi-full-info",
    }
    return payload


def _collect_env_paths(var_name: str) -> list[Path]:
    raw = os.environ.get(var_name, "")
    if not raw:
        return []
    return [Path(part) for part in raw.split(":") if part]


def main(output_path: Path) -> None:
    json_paths = _collect_env_paths("SGMI_JSON_FILES")
    ddl_paths = _collect_env_paths("SGMI_DDL_FILES")
    xml_paths = _collect_env_paths("SGMI_CONTROLM_FILES")
    view_tmp_dir_raw = os.environ.get("SGMI_VIEW_TMP_DIR", "")
    view_tmp_dir = Path(view_tmp_dir_raw) if view_tmp_dir_raw else None
    view_registry_raw = os.environ.get("SGMI_VIEW_REGISTRY_OUT", "")
    view_registry_out = Path(view_registry_raw) if view_registry_raw else None
    view_summary_raw = os.environ.get("SGMI_VIEW_SUMMARY_OUT", "")
    view_summary_out = Path(view_summary_raw) if view_summary_raw else None

    payload = build_payload(
        json_paths=json_paths,
        ddl_paths=ddl_paths,
        xml_paths=xml_paths,
        view_tmp_dir=view_tmp_dir,
        view_registry_out=view_registry_out,
        view_summary_out=view_summary_out,
    )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: sgmi_view_builder.py <output_json_path>")
    main(Path(sys.argv[1]))
