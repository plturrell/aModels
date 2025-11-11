#!/usr/bin/env python3
"""
Advanced sap-rpt-1-oss usage: Multi-task learning (classification + regression).
Supports quality score prediction, active learning, and enhanced features.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

# Add sap-rpt-1-oss to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
SAP_RPT_PATH = REPO_ROOT / "models" / "sap-rpt-1-oss-main"

if SAP_RPT_PATH.exists():
    sys.path.insert(0, str(SAP_RPT_PATH))
else:
    print(json.dumps({"error": "sap-rpt-1-oss not found", "method": "fallback"}))
    sys.exit(1)

try:
    from sap_rpt_oss import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Regressor
except ImportError as exc:
    print(json.dumps({"error": f"Failed to import: {exc}", "method": "fallback"}))
    sys.exit(1)


# Global model instances
_classifier = None
_regressor = None
_model_lock = None


def get_models(training_data_path: Optional[str] = None) -> Tuple[Optional[SAP_RPT_OSS_Classifier], Optional[SAP_RPT_OSS_Regressor]]:
    """Get or create classifier and regressor instances."""
    global _classifier, _regressor, _model_lock
    
    if _model_lock is None:
        import threading
        _model_lock = threading.Lock()
    
    with _model_lock:
        if training_data_path and os.path.exists(training_data_path):
            try:
                training_df = pd.read_json(training_data_path)
                
                if len(training_df) > 0:
                    # Prepare features
                    feature_cols = [col for col in training_df.columns 
                                   if col not in ['classification', 'quality_score', 'confidence', 'table_name']]
                    
                    if 'classification' in training_df.columns:
                        X_train_clf = training_df[feature_cols]
                        y_train_clf = training_df['classification']
                        
                        if _classifier is None:
                            _classifier = SAP_RPT_OSS_Classifier(
                                max_context_size=2048,
                                bagging=2  # Moderate bagging for better accuracy
                            )
                            _classifier.fit(X_train_clf, y_train_clf)
                    
                    if 'quality_score' in training_df.columns:
                        X_train_reg = training_df[feature_cols]
                        y_train_reg = training_df['quality_score']
                        
                        if _regressor is None:
                            _regressor = SAP_RPT_OSS_Regressor(
                                max_context_size=2048,
                                bagging=2
                            )
                            _regressor.fit(X_train_reg, y_train_reg)
            except Exception as e:
                print(f"Failed to load models: {e}", file=sys.stderr)
    
    return _classifier, _regressor


def extract_enhanced_features(table_name: str, columns: list, context: str = "", 
                              metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Extract enhanced features with domain-specific knowledge."""
    table_lower = table_name.lower()
    metadata = metadata or {}
    
    # Basic features
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
        'has_primary_key': 0,
        'has_foreign_key': 0,
        'has_index': 0,
        'has_constraints': 0,
        'column_name_entropy': 0.0,
        'table_name_complexity': 0.0,
    }
    
    # Enhanced column analysis
    numeric_count = 0
    string_count = 0
    date_count = 0
    total_col_length = 0
    column_names = []
    has_pk = False
    has_fk = False
    
    if columns:
        for col in columns:
            col_name = ""
            col_type = ""
            col_metadata = {}
            
            if isinstance(col, dict):
                col_name = col.get("name", "").lower()
                col_type = col.get("type", "").lower()
                col_metadata = col
            elif isinstance(col, str):
                col_name = col.lower()
            
            column_names.append(col_name)
            total_col_length += len(col_name)
            
            # Type detection
            if 'id' in col_name or col_name.endswith('_id'):
                features['has_id_column'] = 1
                if 'primary' in str(col_metadata).lower() or 'pk' in str(col_metadata).lower():
                    has_pk = True
            if 'date' in col_name or 'time' in col_name or 'timestamp' in col_name:
                features['has_date_column'] = 1
                date_count += 1
            if any(x in col_name for x in ['amount', 'price', 'cost', 'value', 'total', 'sum']):
                features['has_amount_column'] = 1
            if 'status' in col_name or 'state' in col_name:
                features['has_status_column'] = 1
            if 'foreign' in str(col_metadata).lower() or 'fk' in str(col_metadata).lower():
                has_fk = True
            
            # Type classification
            if col_type in ['int', 'integer', 'bigint', 'number', 'numeric', 'float', 'double', 'decimal']:
                numeric_count += 1
            elif col_type in ['string', 'varchar', 'text', 'char']:
                string_count += 1
    
    # Calculate ratios
    if len(columns) > 0:
        features['avg_column_name_length'] = total_col_length / len(columns)
        features['numeric_column_ratio'] = numeric_count / len(columns)
        features['string_column_ratio'] = string_count / len(columns)
        features['date_column_ratio'] = date_count / len(columns)
        
        # Column name entropy (diversity of column names)
        if column_names:
            from collections import Counter
            char_counts = Counter(''.join(column_names))
            total_chars = sum(char_counts.values())
            if total_chars > 0:
                entropy = -sum((count/total_chars) * np.log2(count/total_chars) 
                             for count in char_counts.values() if count > 0)
                features['column_name_entropy'] = entropy
    
    # Table name patterns
    if any(x in table_lower for x in ['ref', 'lookup', 'code', 'dim', 'master']):
        features['has_ref_in_name'] = 1
    if any(x in table_lower for x in ['trans', 'txn', 'order', 'payment', 'invoice']):
        features['has_trans_in_name'] = 1
    if any(x in table_lower for x in ['staging', 'stage', 'temp', 'tmp']):
        features['has_staging_in_name'] = 1
    if any(x in table_lower for x in ['test', 'mock', 'sample']):
        features['has_test_in_name'] = 1
    
    # Table name complexity (number of underscores, camelCase transitions)
    underscore_count = table_name.count('_')
    camel_case_transitions = sum(1 for i in range(1, len(table_name)) 
                                if table_name[i].isupper() and table_name[i-1].islower())
    features['table_name_complexity'] = underscore_count + camel_case_transitions
    
    # Metadata-based features
    if metadata:
        if 'has_primary_key' in metadata:
            has_pk = metadata['has_primary_key']
        if 'has_foreign_key' in metadata:
            has_fk = metadata['has_foreign_key']
        if 'has_index' in metadata:
            features['has_index'] = 1 if metadata['has_index'] else 0
        if 'has_constraints' in metadata:
            features['has_constraints'] = 1 if metadata['has_constraints'] else 0
    
    features['has_primary_key'] = 1 if has_pk else 0
    features['has_foreign_key'] = 1 if has_fk else 0
    
    return features


def predict_multi_task(table_name: str, columns: list, context: str = "",
                       training_data_path: Optional[str] = None,
                       metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Multi-task prediction: classification + regression (quality score)."""
    try:
        features = extract_enhanced_features(table_name, columns, context, metadata)
        features_df = pd.DataFrame([features])
        
        classifier, regressor = get_models(training_data_path)
        
        result = {
            "features": features,
            "method": "sap-rpt-multi-task"
        }
        
        # Classification prediction
        if classifier is not None:
            try:
                predictions = classifier.predict(features_df)
                probabilities = classifier.predict_proba(features_df)
                
                if len(predictions) > 0:
                    classification = str(predictions[0])
                    proba_dict = probabilities[0]
                    
                    if isinstance(proba_dict, dict):
                        confidence = float(proba_dict.get(classification, 0.0))
                        result["classification"] = classification
                        result["classification_confidence"] = confidence
                        result["classification_probabilities"] = {k: float(v) for k, v in proba_dict.items()}
                    elif isinstance(proba_dict, np.ndarray) and len(proba_dict) > 0:
                        confidence = float(np.max(proba_dict))
                        result["classification"] = classification
                        result["classification_confidence"] = confidence
            except Exception as e:
                result["classification_error"] = str(e)
        
        # Regression prediction (quality score)
        if regressor is not None:
            try:
                quality_score = regressor.predict(features_df)
                if isinstance(quality_score, np.ndarray) and len(quality_score) > 0:
                    result["quality_score"] = float(quality_score[0])
                    result["quality_score_predicted"] = True
            except Exception as e:
                result["quality_score_error"] = str(e)
        
        # Active learning: identify uncertain predictions
        classification_conf = result.get("classification_confidence", 0.0)
        if classification_conf > 0 and classification_conf < 0.7:
            result["needs_review"] = True
            result["uncertainty_reason"] = f"Low confidence ({classification_conf:.2f})"
        elif classification_conf == 0:
            result["needs_review"] = True
            result["uncertainty_reason"] = "No classification available"
        else:
            result["needs_review"] = False
        
        # Fallback if no models available
        if "classification" not in result:
            # Use feature-based classification
            table_lower = table_name.lower()
            if features['has_trans_in_name'] or (features['has_amount_column'] and features['has_date_column']):
                result["classification"] = "transaction"
                result["classification_confidence"] = 0.85
            elif features['has_ref_in_name']:
                result["classification"] = "reference"
                result["classification_confidence"] = 0.80
            elif features['has_staging_in_name']:
                result["classification"] = "staging"
                result["classification_confidence"] = 0.75
            else:
                result["classification"] = "unknown"
                result["classification_confidence"] = 0.30
            result["method"] = "feature-based-fallback"
        
        # Default quality score if not predicted
        if "quality_score" not in result:
            # Heuristic quality score based on features
            quality = 0.5
            if features['has_primary_key']:
                quality += 0.1
            if features['column_count'] > 5:
                quality += 0.1
            if features['column_name_entropy'] > 3.0:
                quality += 0.1
            if features['has_date_column']:
                quality += 0.1
            result["quality_score"] = min(quality, 1.0)
            result["quality_score_predicted"] = False
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "method": "error-fallback"
        }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Advanced sap-rpt-1-oss multi-task prediction")
    parser.add_argument("--table-name", required=True)
    parser.add_argument("--columns", help="JSON array of column definitions")
    parser.add_argument("--context", help="Context/DDL string", default="")
    parser.add_argument("--training-data", help="Path to training data JSON file")
    parser.add_argument("--metadata", help="JSON metadata", default="{}")
    
    args = parser.parse_args()
    
    columns = []
    if args.columns:
        try:
            columns = json.loads(args.columns)
        except:
            pass
    
    metadata = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except:
            pass
    
    result = predict_multi_task(
        args.table_name,
        columns,
        args.context or "",
        args.training_data,
        metadata
    )
    
    print(json.dumps(result))


if __name__ == "__main__":
    main()

