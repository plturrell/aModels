# SGMI Pipeline Metamodel

This directory contains the canonical SGMI pipeline metamodel artifacts—trimmed representations of the Control-M schedules, Hive schema, and JSON table snapshots that describe how the SGMI data pipelines are wired. They can be fed to the `/graph` endpoint to model the pipeline topology without touching the operational configuration files.

Artifacts:

- `sgmi_table_pipeline.json` – trimmed JSON dataset used to infer column types and nullability.
- `sgmi_hive_pipeline.hql` – representative Hive DDL snippet for table metadata extraction.
- `sgmi_controlm_pipeline.xml` – Control-M job definition showing schedule metadata, IN/OUT conditions, and ODATE handling.

Use these in automation/tests instead of the original files in `SGMI-controlm/` or `HIVE DDLS/` when you just need a functional smoke test.
