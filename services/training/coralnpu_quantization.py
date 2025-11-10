"""Model quantization pipeline for Coral NPU."""

import os
import logging
import pickle
import hashlib
from typing import Optional, Dict, Any, List, Callable, Tuple
from pathlib import Path
import numpy as np

from .coralnpu_client import CoralNPUClient

logger = logging.getLogger(__name__)


class CalibrationDatasetManager:
    """Manages calibration datasets for NPU quantization."""
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_cache_size: int = 10,
        calibration_size: Optional[int] = None
    ):
        """
        Initialize calibration dataset manager.
        
        Args:
            cache_dir: Directory for caching calibration datasets
            max_cache_size: Maximum number of cached datasets
            calibration_size: Default calibration dataset size
        """
        self.cache_dir = cache_dir or os.getenv("CORALNPU_CALIBRATION_CACHE", "./calibration_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.max_cache_size = max_cache_size
        self.calibration_size = calibration_size or int(
            os.getenv("CORALNPU_CALIBRATION_SIZE", "100")
        )
        
        self.cache_metadata = {}
        self._load_cache_metadata()
    
    def _load_cache_metadata(self):
        """Load cache metadata from disk."""
        metadata_path = Path(self.cache_dir) / "cache_metadata.pkl"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'rb') as f:
                    self.cache_metadata = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self.cache_metadata = {}
    
    def _save_cache_metadata(self):
        """Save cache metadata to disk."""
        metadata_path = Path(self.cache_dir) / "cache_metadata.pkl"
        try:
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.cache_metadata, f)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _get_dataset_hash(self, dataset: Any) -> str:
        """Generate hash for dataset."""
        try:
            if isinstance(dataset, np.ndarray):
                data_str = dataset.tobytes()
            elif hasattr(dataset, '__iter__'):
                # Sample first few items for hashing
                sample = list(dataset)[:10]
                data_str = str(sample).encode()
            else:
                data_str = str(dataset).encode()
            
            return hashlib.md5(data_str).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to hash dataset: {e}")
            return hashlib.md5(str(dataset).encode()).hexdigest()
    
    def create_from_numpy(
        self,
        data: np.ndarray,
        cache_key: Optional[str] = None
    ) -> List[np.ndarray]:
        """Create calibration dataset from numpy array.
        
        Args:
            data: Numpy array of shape (N, ...)
            cache_key: Optional cache key for reuse
        
        Returns:
            List of calibration samples
        """
        if cache_key and cache_key in self.cache_metadata:
            return self._load_from_cache(cache_key)
        
        # Sample calibration data
        num_samples = min(self.calibration_size, len(data))
        indices = np.random.choice(len(data), num_samples, replace=False)
        calibration_data = [data[i] for i in indices]
        
        if cache_key:
            self._save_to_cache(cache_key, calibration_data)
        
        return calibration_data
    
    def create_from_torch(
        self,
        data_loader: Any,
        cache_key: Optional[str] = None
    ) -> List[np.ndarray]:
        """Create calibration dataset from PyTorch data loader.
        
        Args:
            data_loader: PyTorch DataLoader
            cache_key: Optional cache key for reuse
        
        Returns:
            List of calibration samples as numpy arrays
        """
        if cache_key and cache_key in self.cache_metadata:
            return self._load_from_cache(cache_key)
        
        calibration_data = []
        try:
            import torch
            
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                elif hasattr(batch, 'x'):
                    inputs = batch.x
                else:
                    inputs = batch
                
                # Convert to numpy
                if isinstance(inputs, torch.Tensor):
                    inputs_np = inputs.cpu().numpy()
                else:
                    inputs_np = np.array(inputs)
                
                # Flatten batch dimension
                if len(inputs_np.shape) > 1:
                    for sample in inputs_np:
                        calibration_data.append(sample)
                else:
                    calibration_data.append(inputs_np)
                
                if len(calibration_data) >= self.calibration_size:
                    break
        except Exception as e:
            logger.error(f"Failed to create calibration dataset from torch: {e}")
        
        if cache_key:
            self._save_to_cache(cache_key, calibration_data)
        
        return calibration_data[:self.calibration_size]
    
    def create_from_tf_data(
        self,
        dataset: Any,
        cache_key: Optional[str] = None
    ) -> List[np.ndarray]:
        """Create calibration dataset from TensorFlow dataset.
        
        Args:
            dataset: TensorFlow dataset
            cache_key: Optional cache key for reuse
        
        Returns:
            List of calibration samples as numpy arrays
        """
        if cache_key and cache_key in self.cache_metadata:
            return self._load_from_cache(cache_key)
        
        calibration_data = []
        try:
            import tensorflow as tf
            
            count = 0
            for sample in dataset.take(self.calibration_size):
                if isinstance(sample, dict):
                    # Get first input tensor
                    inputs = list(sample.values())[0]
                elif isinstance(sample, tuple):
                    inputs = sample[0]
                else:
                    inputs = sample
                
                # Convert to numpy
                if isinstance(inputs, tf.Tensor):
                    inputs_np = inputs.numpy()
                else:
                    inputs_np = np.array(inputs)
                
                calibration_data.append(inputs_np)
                count += 1
                
                if count >= self.calibration_size:
                    break
        except Exception as e:
            logger.error(f"Failed to create calibration dataset from tf.data: {e}")
        
        if cache_key:
            self._save_to_cache(cache_key, calibration_data)
        
        return calibration_data
    
    def _save_to_cache(self, cache_key: str, data: List[np.ndarray]):
        """Save dataset to cache."""
        try:
            cache_path = Path(self.cache_dir) / f"{cache_key}.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            self.cache_metadata[cache_key] = {
                "path": str(cache_path),
                "size": len(data),
                "shape": data[0].shape if data else None
            }
            
            # Enforce cache size limit
            if len(self.cache_metadata) > self.max_cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = list(self.cache_metadata.keys())[0]
                self._remove_from_cache(oldest_key)
            
            self._save_cache_metadata()
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def _load_from_cache(self, cache_key: str) -> List[np.ndarray]:
        """Load dataset from cache."""
        if cache_key not in self.cache_metadata:
            return []
        
        try:
            cache_path = Path(self.cache_metadata[cache_key]["path"])
            if not cache_path.exists():
                del self.cache_metadata[cache_key]
                return []
            
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Loaded calibration dataset from cache: {cache_key}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return []
    
    def _remove_from_cache(self, cache_key: str):
        """Remove dataset from cache."""
        if cache_key in self.cache_metadata:
            cache_path = Path(self.cache_metadata[cache_key]["path"])
            if cache_path.exists():
                cache_path.unlink()
            del self.cache_metadata[cache_key]
            self._save_cache_metadata()
    
    def compute_calibration_metrics(
        self,
        original_model: Any,
        quantized_model: Any,
        test_data: List[np.ndarray]
    ) -> Dict[str, float]:
        """Compute calibration metrics (quantization error, accuracy drop).
        
        Args:
            original_model: Original model
            quantized_model: Quantized model
            test_data: Test dataset
        
        Returns:
            Dictionary with metrics
        """
        metrics = {
            "quantization_error": 0.0,
            "accuracy_drop": 0.0,
            "mean_absolute_error": 0.0
        }
        
        try:
            errors = []
            original_outputs = []
            quantized_outputs = []
            
            for sample in test_data:
                # Get predictions from both models
                orig_pred = self._predict(original_model, sample)
                quant_pred = self._predict(quantized_model, sample)
                
                original_outputs.append(orig_pred)
                quantized_outputs.append(quant_pred)
                
                # Compute error
                error = np.abs(orig_pred - quant_pred).mean()
                errors.append(error)
            
            metrics["quantization_error"] = np.mean(errors)
            metrics["mean_absolute_error"] = np.mean([np.abs(o - q).mean() 
                                                      for o, q in zip(original_outputs, quantized_outputs)])
            
            # Accuracy drop (if classification)
            if len(original_outputs[0].shape) == 1:
                orig_acc = np.mean([np.argmax(o) == np.argmax(q) 
                                   for o, q in zip(original_outputs, quantized_outputs)])
                metrics["accuracy_drop"] = 1.0 - orig_acc
        
        except Exception as e:
            logger.warning(f"Failed to compute calibration metrics: {e}")
        
        return metrics
    
    def _predict(self, model: Any, sample: np.ndarray) -> np.ndarray:
        """Get prediction from model."""
        try:
            # Try different model interfaces
            if hasattr(model, 'predict'):
                return model.predict(sample.reshape(1, -1) if len(sample.shape) == 1 else sample[np.newaxis])
            elif callable(model):
                return model(sample)
            else:
                logger.warning("Unknown model interface")
                return np.zeros(10)  # Dummy output
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return np.zeros(10)


class CoralNPUQuantizer:
    """Handles model quantization for Coral NPU."""
    
    def __init__(
        self,
        client: Optional[CoralNPUClient] = None,
        enabled: Optional[bool] = None,
    ):
        """
        Initialize Coral NPU quantizer.
        
        Args:
            client: Optional Coral NPU client (auto-created if None)
            enabled: Whether quantization is enabled (from env if None)
        """
        self.client = client or CoralNPUClient()
        self.enabled = enabled if enabled is not None else (
            os.getenv("CORALNPU_QUANTIZE_MODELS", "false").lower() == "true"
        )
    
    def quantize_pytorch_model(
        self,
        model_path: str,
        output_path: Optional[str] = None,
        representative_dataset: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Quantize a PyTorch model for Coral NPU.
        
        This converts PyTorch -> ONNX -> TensorFlow -> TFLite -> Edge TPU.
        
        Args:
            model_path: Path to PyTorch model
            output_path: Optional output path
            representative_dataset: Optional representative dataset
            
        Returns:
            Path to quantized model, or None if quantization failed
        """
        if not self.enabled:
            logger.info("Coral NPU quantization not enabled")
            return None
        
        try:
            # Step 1: Convert PyTorch to ONNX
            onnx_path = self._pytorch_to_onnx(model_path)
            if onnx_path is None:
                return None
            
            # Step 2: Convert ONNX to TensorFlow
            tf_path = self._onnx_to_tensorflow(onnx_path)
            if tf_path is None:
                return None
            
            # Step 3: Quantize TensorFlow to TFLite
            tflite_path = self.client.quantize_model(
                tf_path,
                output_path=output_path,
                representative_dataset=representative_dataset,
            )
            
            # Cleanup intermediate files
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)
            if os.path.exists(tf_path):
                import shutil
                shutil.rmtree(tf_path, ignore_errors=True)
            
            return tflite_path
        
        except Exception as e:
            logger.error(f"PyTorch model quantization failed: {e}")
            return None
    
    def _detect_model_type_and_shape(self, model_path: str) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Detect model type and input shape from saved model.
        
        Args:
            model_path: Path to PyTorch model
            
        Returns:
            Tuple of (model_type, input_shape_dict, dynamic_axes_dict)
            model_type: "gnn", "transformer", "cnn", or "unknown"
            input_shape_dict: Dictionary with input shapes
            dynamic_axes_dict: Dictionary with dynamic axes for ONNX export
        """
        try:
            import torch
            
            # Load model state
            state = torch.load(model_path, map_location="cpu")
            
            # Detect model type based on saved state keys
            model_type = "unknown"
            input_shape = {}
            dynamic_axes = {}
            
            # Check for GNN model indicators
            if "num_node_features" in state:
                model_type = "gnn"
                num_node_features = state.get("num_node_features", 64)
                # GNN models typically take (num_nodes, num_features) and edge_index
                # Use dynamic batch size for nodes
                input_shape = {
                    "x": (1, num_node_features),  # (batch_size, num_features) - will be dynamic
                    "edge_index": (2, 10)  # (2, num_edges) - will be dynamic
                }
                dynamic_axes = {
                    "x": {0: "num_nodes"},  # Dynamic number of nodes
                    "edge_index": {1: "num_edges"}  # Dynamic number of edges
                }
                logger.info(f"Detected GNN model with {num_node_features} node features")
            
            # Check for Transformer model indicators
            elif "num_heads" in state or "d_model" in state or "n_layers" in state:
                model_type = "transformer"
                d_model = state.get("d_model", 512)
                seq_len = state.get("max_seq_len", 128)
                input_shape = {
                    "input_ids": (1, seq_len),  # (batch_size, seq_len)
                    "attention_mask": (1, seq_len)
                }
                dynamic_axes = {
                    "input_ids": {0: "batch_size", 1: "seq_len"},
                    "attention_mask": {0: "batch_size", 1: "seq_len"}
                }
                logger.info(f"Detected Transformer model with d_model={d_model}, seq_len={seq_len}")
            
            # Check for CNN model indicators
            elif "num_classes" in state and "num_node_features" not in state:
                # Could be CNN or other classifier
                # Try to infer from model structure if available
                model_type = "cnn"
                # Default CNN input shape (can be overridden)
                input_shape = {
                    "input": (1, 3, 224, 224)  # (batch_size, channels, height, width)
                }
                dynamic_axes = {
                    "input": {0: "batch_size"}  # Dynamic batch size
                }
                logger.info("Detected CNN model (using default shape)")
            
            else:
                # Unknown model type - use default
                model_type = "unknown"
                input_shape = {
                    "input": (1, 3, 224, 224)
                }
                dynamic_axes = {
                    "input": {0: "batch_size"}
                }
                logger.warning(f"Unknown model type, using default input shape")
            
            return model_type, input_shape, dynamic_axes
        
        except Exception as e:
            logger.error(f"Failed to detect model type and shape: {e}")
            # Return defaults
            return "unknown", {"input": (1, 3, 224, 224)}, {"input": {0: "batch_size"}}
    
    def _pytorch_to_onnx(self, model_path: str) -> Optional[str]:
        """Convert PyTorch model to ONNX with dynamic shapes.
        
        Args:
            model_path: Path to PyTorch model
            
        Returns:
            Path to ONNX model or None if conversion failed
        """
        try:
            import torch
            
            # Load PyTorch model
            model = torch.load(model_path, map_location="cpu")
            model.eval()
            
            # Detect model type and input shape
            model_type, input_shape, dynamic_axes = self._detect_model_type_and_shape(model_path)
            
            # Create dummy inputs based on detected model type
            dummy_inputs = {}
            
            if model_type == "gnn":
                # GNN models need node features and edge index
                num_node_features = input_shape["x"][1]
                num_nodes = 10  # Example number of nodes
                num_edges = 20  # Example number of edges
                
                dummy_inputs["x"] = torch.randn(num_nodes, num_node_features)
                dummy_inputs["edge_index"] = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
                
                # For ONNX export, we need to pass inputs as tuple or dict
                # PyTorch Geometric models typically take (x, edge_index) as positional args
                dummy_input = (dummy_inputs["x"], dummy_inputs["edge_index"])
                input_names = ["x", "edge_index"]
                output_names = ["output"]
                
            elif model_type == "transformer":
                # Transformer models need input_ids and attention_mask
                seq_len = input_shape["input_ids"][1]
                vocab_size = 10000  # Default vocab size
                
                dummy_inputs["input_ids"] = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
                dummy_inputs["attention_mask"] = torch.ones(1, seq_len, dtype=torch.long)
                
                dummy_input = (dummy_inputs["input_ids"], dummy_inputs["attention_mask"])
                input_names = ["input_ids", "attention_mask"]
                output_names = ["output"]
                
            else:
                # CNN or unknown - use default
                dummy_input = torch.randn(*input_shape["input"])
                input_names = ["input"]
                output_names = ["output"]
            
            onnx_path = str(Path(model_path).with_suffix(".onnx"))
            
            # Export to ONNX with dynamic axes
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes if dynamic_axes else None,
            )
            
            logger.info(f"PyTorch model converted to ONNX with dynamic shapes: {onnx_path} (model_type={model_type})")
            return onnx_path
        
        except Exception as e:
            logger.error(f"PyTorch to ONNX conversion failed: {e}", exc_info=True)
            return None
    
    def _onnx_to_tensorflow(self, onnx_path: str) -> Optional[str]:
        """Convert ONNX model to TensorFlow."""
        try:
            import onnx
            from onnx_tf.backend import prepare
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Convert to TensorFlow
            tf_rep = prepare(onnx_model)
            
            # Save TensorFlow model
            tf_path = str(Path(onnx_path).with_suffix(""))
            tf_rep.export_graph(tf_path)
            
            logger.info(f"ONNX model converted to TensorFlow: {tf_path}")
            return tf_path
        
        except Exception as e:
            logger.error(f"ONNX to TensorFlow conversion failed: {e}")
            return None
    
    def quantize_gnn_model(
        self,
        model_path: str,
        output_path: Optional[str] = None,
        representative_dataset: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Quantize a GNN model for Coral NPU.
        
        Args:
            model_path: Path to GNN model
            output_path: Optional output path
            representative_dataset: Optional representative dataset
            
        Returns:
            Path to quantized model, or None if quantization failed
        """
        return self.quantize_pytorch_model(
            model_path,
            output_path=output_path,
            representative_dataset=representative_dataset,
        )
    
    def quantize_transformer_model(
        self,
        model_path: str,
        output_path: Optional[str] = None,
        representative_dataset: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Quantize a transformer model for Coral NPU.
        
        Args:
            model_path: Path to transformer model
            output_path: Optional output path
            representative_dataset: Optional representative dataset
            
        Returns:
            Path to quantized model, or None if quantization failed
        """
        return self.quantize_pytorch_model(
            model_path,
            output_path=output_path,
            representative_dataset=representative_dataset,
        )


def create_quantizer(
    client: Optional[CoralNPUClient] = None,
    enabled: Optional[bool] = None,
) -> CoralNPUQuantizer:
    """
    Create a Coral NPU quantizer instance.
    
    Args:
        client: Optional Coral NPU client
        enabled: Whether quantization is enabled
        
    Returns:
        CoralNPUQuantizer instance
    """
    return CoralNPUQuantizer(client=client, enabled=enabled)

