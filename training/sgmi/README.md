# SGMI Training Data

This directory contains training data and scripts for the SGMI (SAP Global Manufacturing Intelligence) process training pipeline.

## Contents

- **hive-ddl/**: Hive Data Definition Language files for SGMI database schemas
  - Contains table definitions, views, and cleaned statements for various SGMI environments (sit, etl, stg)
  
- **pipeline_metamodel/**: Pipeline metamodel definitions
  - Control-M pipeline XML configurations
  - Hive pipeline HQL scripts
  - Table pipeline JSON definitions

- **sgmi-scripts/**: Execution scripts for SGMI processes
  - Spark job scripts for Tableau refresh operations
  - JBS framework scripts for ingestion, staging, and reporting
  - Environment configuration files

- **sgmi-controlm/**: Control-M migration and implementation documentation
  - Migration plans, release notes, and configuration files
  - Production Control-M definitions

- **json_with_changes.json**: Training dataset with annotated changes for process learning

## Usage

This data is used to train process understanding models that can:
- Extract and replicate database schemas
- Understand pipeline workflows and dependencies
- Learn from Control-M job definitions
- Process and normalize SGMI-specific transformations

## Notes

- Large archive files (`.zip`, `.7z`) and compiled binaries (`.jar`) are excluded from Git tracking
- The JSON file contains process annotations and changes for training purposes

