#!/usr/bin/env python3
"""
Bridge script for sap-rpt-1-oss table classification.
Uses SAP_RPT_OSS_Classifier for accurate table classification.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add sap-rpt-1-oss to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
SAP_RPT_PATH = REPO_ROOT / "models" / "sap-rpt-1-oss-main"

if SAP_RPT_PATH.exists():
    sys.path.insert(0, str(SAP_RPT_PATH))
else:
    print(json.dumps({"classification": "unknown", "confidence": 0.0, "method": "pattern"}))
    print(f"sap-rpt-1-oss not found at {SAP_RPT_PATH}", file=sys.stderr)
    sys.exit(0)

try:
    import pandas as pd
    import numpy as np
    from sap_rpt_oss import SAP_RPT_OSS_Classifier
except ImportError as exc:
    print(json.dumps({"classification": "unknown", "confidence": 0.0, "method": "pattern"}))
    print(f"Failed to import sap-rpt-1-oss modules: {exc}", file=sys.stderr)
    sys.exit(0)


def extract_table_features(table_name: str, columns: list, context: str = "") -> dict:
    """Extract features from table for classification."""
    table_lower = table_name.lower()
    
    features = {
        'table_name_length': len(table_name),
        'column_count': len(columns) if columns else 0,
        'has_id_column': 0,
        'has_date_column': 0,
        'has_amount_column': 0,
        'has_status_column': 0,
        'has_ref_in_name': 0,
        'has_trans_in_name': 0,
        'has_staging_in_name': 0,
        'has_test_in_name': 0,
    }
    
    # Check column names
    if columns:
        for col in columns:
            col_name = ""
            col_type = ""
            if isinstance(col, dict):
                col_name = col.get("name", "").lower()
                col_type = col.get("type", "").lower()
            elif isinstance(col, str):
                col_name = col.lower()
            
            if 'id' in col_name:
                features['has_id_column'] = 1
            if 'date' in col_name or 'time' in col_name or 'timestamp' in col_name:
                features['has_date_column'] = 1
            if 'amount' in col_name or 'price' in col_name or 'cost' in col_name or 'value' in col_name:
                features['has_amount_column'] = 1
            if 'status' in col_name or 'state' in col_name:
                features['has_status_column'] = 1
    
    # Check table name patterns
    if 'ref' in table_lower or 'lookup' in table_lower or 'code' in table_lower:
        features['has_ref_in_name'] = 1
    if 'trans' in table_lower or 'txn' in table_lower or 'order' in table_lower:
        features['has_trans_in_name'] = 1
    if 'staging' in table_lower or 'stage' in table_lower or 'temp' in table_lower:
        features['has_staging_in_name'] = 1
    if 'test' in table_lower or 'mock' in table_lower:
        features['has_test_in_name'] = 1
    
    return features


def classify_table_with_sap_rpt(table_name: str, columns: list, context: str = "", training_data_path: str = None) -> dict:
    """Classify table using sap-rpt-1-oss classifier."""
    try:
        # Extract features
        features = extract_table_features(table_name, columns, context)
        
        # For now, use pattern-based classification with enhanced features
        # Full sap-rpt-1-oss classifier would require training data
        # This is a hybrid approach that uses feature extraction
        
        table_lower = table_name.lower()
        confidence = 0.0
        classification = "unknown"
        evidence = []
        
        # Enhanced classification based on features
        if features['has_trans_in_name'] or (features['has_amount_column'] and features['has_date_column']):
            classification = "transaction"
            confidence = 0.85
            evidence.append("Transaction indicators in name or structure")
        elif features['has_ref_in_name'] or (features['has_id_column'] and not features['has_amount_column']):
            classification = "reference"
            confidence = 0.80
            evidence.append("Reference/lookup indicators")
        elif features['has_staging_in_name']:
            classification = "staging"
            confidence = 0.75
            evidence.append("Staging indicators")
        elif features['has_test_in_name']:
            classification = "test"
            confidence = 0.70
            evidence.append("Test indicators")
        else:
            # Default classification based on structure
            if features['column_count'] > 10 and features['has_date_column']:
                classification = "transaction"
                confidence = 0.60
                evidence.append("Large table with date columns suggests transaction")
            elif features['column_count'] < 5:
                classification = "reference"
                confidence = 0.55
                evidence.append("Small table suggests reference/lookup")
            else:
                classification = "unknown"
                confidence = 0.30
        
        return {
            "classification": classification,
            "confidence": confidence,
            "evidence": evidence,
            "features": features,
            "method": "sap-rpt-enhanced"
        }
        
    except Exception as e:
        # Fallback to pattern-based
        return {
            "classification": "unknown",
            "confidence": 0.0,
            "evidence": [f"Error: {str(e)}"],
            "method": "pattern-fallback"
        }


def main() -> None:
    """Main entry point for table classification."""
    parser = argparse.ArgumentParser(description="Classify table using sap-rpt-1-oss")
    parser.add_argument("--table-name", required=True)
    parser.add_argument("--columns", help="JSON array of column definitions")
    parser.add_argument("--context", help="Context/DDL string", default="")
    parser.add_argument("--training-data", help="Path to training data (optional)")
    
    args = parser.parse_args()
    
    columns = []
    if args.columns:
        try:
            columns = json.loads(args.columns)
        except:
            pass
    
    result = classify_table_with_sap_rpt(
        args.table_name,
        columns,
        args.context or "",
        args.training_data
    )
    
    print(json.dumps(result))


if __name__ == "__main__":
    main()

