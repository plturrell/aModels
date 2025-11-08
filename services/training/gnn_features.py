from typing import Dict, Any, List

def extract_node_features(node: Dict[str, Any], props: Dict[str, Any]) -> List[float]:
    features: List[float] = []
    node_type = node.get("type", node.get("label", "unknown"))
    type_features = [0.0] * 10
    type_map = {
        "table": 0,
        "column": 1,
        "view": 2,
        "database": 3,
        "schema": 4,
        "sql": 5,
        "control-m": 6,
        "project": 7,
        "system": 8,
        "information-system": 9,
    }
    idx = type_map.get(node_type)
    if idx is not None and idx < len(type_features):
        type_features[idx] = 1.0
    features.extend(type_features)
    if isinstance(props, dict):
        features.append(float(props.get("column_count", 0)))
        features.append(float(props.get("row_count", 0)))
        features.append(float(props.get("data_type_entropy", 0)))
        features.append(float(props.get("nullable_ratio", 0)))
        features.append(float(props.get("metadata_entropy", 0)))
    else:
        features.extend([0.0] * 5)
    while len(features) < 40:
        features.append(0.0)
    return features[:40]
