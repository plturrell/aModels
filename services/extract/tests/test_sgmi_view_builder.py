import json
from pathlib import Path

from agenticAiETH_layer4_Extract.scripts.sgmi_view_builder import build_payload


def _write_file(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_build_payload_with_join_and_metrics(tmp_path):
    json_file = _write_json(tmp_path / "table.json", [{"id": 1, "customer_id": 42}])
    ddl_file = _write_file(
        tmp_path / "schema.hql",
        """
        CREATE TABLE sgmisit.orders (
            order_id BIGINT,
            customer_id BIGINT,
            amount DECIMAL(10,2)
        );

        CREATE TABLE sgmisit.customer (
            customer_id BIGINT,
            name STRING
        );

        CREATE VIEW sgmisit.v_customer_orders AS
        SELECT
            o.order_id,
            o.customer_id,
            c.name AS customer_name,
            o.amount
        FROM sgmisit.orders o
        INNER JOIN sgmisit.customer c
            ON o.customer_id = c.customer_id;
        """,
    )
    xml_file = _write_file(tmp_path / "controlm.xml", "<ControlM />")

    registry_path = tmp_path / "sgmi_view_lineage.json"
    summary_path = tmp_path / "sgmi_view_summary.json"

    payload = build_payload(
        json_paths=[json_file],
        ddl_paths=[ddl_file],
        xml_paths=[xml_file],
        view_tmp_dir=tmp_path / "views",
        view_registry_out=registry_path,
        view_summary_out=summary_path,
    )

    assert payload["json_tables"][0] == str(json_file)
    assert any("CREATE TABLE `sgmisit.v_customer_orders`" in stmt for stmt in payload["hive_ddls"])

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry, "expected at least one view entry"
    view_entry = next(entry for entry in registry if entry["view"] == "sgmisit.v_customer_orders")

    assert {column["name"] for column in view_entry["columns"]} == {
        "order_id",
        "customer_id",
        "customer_name",
        "amount",
    }
    metrics = view_entry["metrics"]
    assert metrics["join_count"] == 1
    assert metrics["information_loss"] == 0.0
    assert metrics["columns_with_sources"] == metrics["column_count"] == 4
    assert view_entry["joins"][0]["type"] == "INNER"
    assert any(item["column"] == "customer_id" for item in view_entry["joins"][0]["condition_columns"])

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["metrics"]["view_count"] >= 1
    assert summary["metrics"]["views_with_joins"] >= 1


def test_infers_types_for_literals_and_aggregates(tmp_path):
    json_file = _write_json(tmp_path / "table.json", [{"order_id": 1, "amount": 100.0}])
    ddl_file = _write_file(
        tmp_path / "schema.hql",
        """
        CREATE TABLE sgmisit.orders (
            order_id BIGINT,
            amount DECIMAL(10,2),
            notes STRING
        );

        CREATE VIEW sgmisit.v_order_metrics AS
        SELECT
            o.order_id,
            SUM(o.amount) AS total_amount,
            COUNT(*) AS order_count,
            COALESCE(o.notes, 'NA') AS notes_filled,
            1 AS constant_one
        FROM sgmisit.orders o;
        """,
    )
    xml_file = _write_file(tmp_path / "controlm.xml", "<ControlM />")

    registry_path = tmp_path / "sgmi_view_lineage.json"
    summary_path = tmp_path / "sgmi_view_summary.json"

    build_payload(
        json_paths=[json_file],
        ddl_paths=[ddl_file],
        xml_paths=[xml_file],
        view_tmp_dir=tmp_path / "views",
        view_registry_out=registry_path,
        view_summary_out=summary_path,
    )

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    view_entry = next(entry for entry in registry if entry["view"] == "sgmisit.v_order_metrics")
    column_types = {column["name"]: column["type"] for column in view_entry["columns"]}

    assert column_types["order_id"] == "BIGINT"
    assert column_types["total_amount"] in {"DECIMAL", "NUMERIC"}
    assert column_types["order_count"] == "BIGINT"
    assert column_types["notes_filled"] == "STRING"
    assert column_types["constant_one"] == "BIGINT"
