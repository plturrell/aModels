#!/usr/bin/env python3
"""Validate training data files."""
import pandas as pd
import yaml
import sys

with open('configs/relational_rt_example.yaml') as f:
    cfg = yaml.safe_load(f)

print("Validating training data files...")
for t in cfg['tables']:
    try:
        df = pd.read_csv(t['path'])
        print(f"✓ {t['name']}: {len(df)} rows, {len(df.columns)} cols")
        print(f"  Columns: {list(df.columns)[:5]}")
        print(f"  Primary key '{t['primary_key']}': {df[t['primary_key']].nunique()} unique values")
    except Exception as e:
        print(f"✗ {t['name']}: ERROR - {e}")
        sys.exit(1)

print("\nAll data files validated successfully!")

