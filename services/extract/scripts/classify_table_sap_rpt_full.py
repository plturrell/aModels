#!/usr/bin/env python3
"""
Full classifier implementation for sap-rpt-1-oss table classification.
Uses SAP_RPT_OSS_Classifier for ML-based classification with training data.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Add sap-rpt-1-oss to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
SAP_RPT_PATH = REPO_ROOT / "models" / "sap-rpt-1-oss-main"

if SAP_RPT_PATH.exists():
    sys.path.insert(0, str(SAP_RPT_PATH))
else:
    print(json.dumps({"classification": "unknown", "confidence": 0.0, "method": "fallback"}))
    print(f"sap-rpt-1-oss not found at {SAP_RPT_PATH}", file=sys.stderr)
    sys.exit(1)

try:
    from sap_rpt_oss import SAP_RPT_OSS_Classifier
except ImportError as exc:
    print(json.dumps({"classification": "unknown", "confidence": 0.0, "method": "fallback"}))
    print(f"Failed to import SAP_RPT_OSS_Classifier: {exc}", file=sys.stderr)
    sys.exit(1)


# Global classifier instance for connection pooling
_classifier = None
_classifier_lock = None
_training_data_path = None


def get_classifier(training_data_path: Optional[str] = None) -> Optional[SAP_RPT_OSS_Classifier]:
    """Get or create classifier instance (connection pooling with training data)."""
    global _classifier, _classifier_lock, _training_data_path
    
    if _classifier_lock is None:
        import threading
        _classifier_lock = threading.Lock()
    
    # Check if we need to recreate classifier (training data changed)
    if training_data_path and training_data_path != _training_data_path:
        _classifier = None
        _training_data_path = training_data_path
    
    with _classifier_lock:
        if _classifier is None:
            if training_data_path and os.path.exists(training_data_path):
                try:
                    # Load training data
                    training_df = pd.read_json(training_data_path)
                    
                    if len(training_df) > 0 and 'classification' in training_df.columns:
                        # Prepare features and labels
                        feature_cols = [col for col in training_df.columns if col != 'classification']
                        X_train = training_df[feature_cols]
                        y_train = training_df['classification']
                        
                        # Initialize classifier with smaller context for faster training
                        _classifier = SAP_RPT_OSS_Classifier(
                            max_context_size=2048,  # Smaller for faster training
                            bagging=1  # No bagging for speed
                        )
                        
                        # Train the classifier
                        _classifier.fit(X_train, y_train)
                        _training_data_path = training_data_path
                    else:
                        print(f"No valid training data in {training_data_path}", file=sys.stderr)
                        return None
                except Exception as e:
                    print(f"Failed to load/train classifier: {e}", file=sys.stderr)
                    return None
            else:
                # No training data, return None (will use fallback)
                return None
    
    return _classifier


def extract_table_features_for_classifier(table_name: str, columns: list, context: str = "") -> Dict[str, Any]:
    """Extract features suitable for SAP_RPT_OSS_Classifier (DataFrame format)."""
    table_lower = table_name.lower()
    
    # Create feature dictionary
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
        'avg_column_name_length': 0.0,
        'numeric_column_ratio': 0.0,
        'string_column_ratio': 0.0,
    }
    
    # Check column names and types
    numeric_count = 0
    string_count = 0
    total_col_length = 0
    
    if columns:
        for col in columns:
            col_name = ""
            col_type = ""
            if isinstance(col, dict):
                col_name = col.get("name", "").lower()
                col_type = col.get("type", "").lower()
            elif isinstance(col, str):
                col_name = col.lower()
            
            total_col_length += len(col_name)
            
            if 'id' in col_name:
                features['has_id_column'] = 1
            if 'date' in col_name or 'time' in col_name or 'timestamp' in col_name:
                features['has_date_column'] = 1
            if 'amount' in col_name or 'price' in col_name or 'cost' in col_name or 'value' in col_name:
                features['has_amount_column'] = 1
            if 'status' in col_name or 'state' in col_name:
                features['has_status_column'] = 1
            
            # Type classification
            if col_type in ['int', 'integer', 'bigint', 'number', 'numeric', 'float', 'double', 'decimal']:
                numeric_count += 1
            elif col_type in ['string', 'varchar', 'text', 'char']:
                string_count += 1
    
    if len(columns) > 0:
        features['avg_column_name_length'] = total_col_length / len(columns)
        features['numeric_column_ratio'] = numeric_count / len(columns)
        features['string_column_ratio'] = string_count / len(columns)
    
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


def classify_table_with_full_classifier(
    table_name: str,
    columns: list,
    context: str = "",
    training_data_path: Optional[str] = None
) -> Dict[str, Any]:
    """Classify table using full SAP_RPT_OSS_Classifier with training data."""
    try:
        # Extract features
        features = extract_table_features_for_classifier(table_name, columns, context)
        
        # Try to use trained classifier if available
        classifier = get_classifier(training_data_path)
        
        if classifier is not None:
            try:
                # Convert features to DataFrame
                features_df = pd.DataFrame([features])
                
                # Predict using trained classifier
                predictions = classifier.predict(features_df)
                probabilities = classifier.predict_proba(features_df)
                
                if len(predictions) > 0:
                    classification = str(predictions[0])
                    
                    # Get confidence from probabilities
                    proba_dict = probabilities[0]
                    if isinstance(proba_dict, dict):
                        confidence = float(proba_dict.get(classification, 0.0))
                    elif isinstance(proba_dict, np.ndarray) and len(proba_dict) > 0:
                        # If array, use max probability
                        confidence = float(np.max(proba_dict))
                    else:
                        confidence = 0.8  # Default confidence for classifier
                    
                    return {
                        "classification": classification,
                        "confidence": confidence,
                        "evidence": [f"ML-based classification using trained SAP_RPT_OSS_Classifier"],
                        "features": features,
                        "method": "sap-rpt-full-classifier",
                        "probabilities": {k: float(v) for k, v in proba_dict.items()} if isinstance(proba_dict, dict) else {}
                    }
            except Exception as e:
                print(f"Classifier prediction failed: {e}", file=sys.stderr)
                # Fallback to feature-based classification
        
        # Fallback: use feature-based classification
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
            "method": "sap-rpt-enhanced-fallback"
        }
        
    except Exception as e:
        # Fallback to pattern-based
        return {
            "classification": "unknown",
            "confidence": 0.0,
            "evidence": [f"Error: {str(e)}"],
            "method": "pattern-fallback"
        }


def collect_training_data(
    table_name: str,
    columns: list,
    classification: str,
    confidence: float,
    output_path: str
) -> bool:
    """Collect training data for classifier training.
    
    Args:
        table_name: Name of the table
        columns: List of column definitions
        classification: Known classification (transaction, reference, staging, test)
        confidence: Confidence in the classification
        output_path: Path to save training data JSON
    
    Returns:
        True if successfully saved
    """
    try:
        # Extract features
        features = extract_table_features_for_classifier(table_name, columns)
        
        # Add classification label
        training_record = features.copy()
        training_record['classification'] = classification
        training_record['confidence'] = confidence
        training_record['table_name'] = table_name  # For reference
        
        # Load existing training data or create new
        training_data = []
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    training_data = json.load(f)
            except:
                training_data = []
        
        # Add new record
        training_data.append(training_record)
        
        # Save updated training data
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Failed to collect training data: {e}", file=sys.stderr)
        return False


def main() -> None:
    """Main entry point for table classification."""
    parser = argparse.ArgumentParser(description="Classify table using full sap-rpt-1-oss classifier")
    parser.add_argument("--table-name", required=True)
    parser.add_argument("--columns", help="JSON array of column definitions")
    parser.add_argument("--context", help="Context/DDL string", default="")
    parser.add_argument("--training-data", help="Path to training data JSON file")
    parser.add_argument("--collect-training", action="store_true", help="Collect training data")
    parser.add_argument("--known-classification", help="Known classification for training data collection")
    parser.add_argument("--training-output", help="Path to save training data")
    
    args = parser.parse_args()
    
    columns = []
    if args.columns:
        try:
            columns = json.loads(args.columns)
        except:
            pass
    
    # Collect training data if requested
    if args.collect_training and args.known_classification and args.training_output:
        success = collect_training_data(
            args.table_name,
            columns,
            args.known_classification,
            1.0,  # Full confidence for known classifications
            args.training_output
        )
        if success:
            print(json.dumps({"status": "collected", "table": args.table_name}))
        else:
            print(json.dumps({"status": "failed"}))
        return
    
    # Classify table
    result = classify_table_with_full_classifier(
        args.table_name,
        columns,
        args.context or "",
        args.training_data
    )
    
    print(json.dumps(result))


if __name__ == "__main__":
    main()

