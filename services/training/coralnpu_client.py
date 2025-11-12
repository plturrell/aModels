"""Coral NPU client wrapper for model quantization, compilation, and inference."""

import os
import logging
import time
from typing import Optional, Dict, Any, Callable, Tuple, List
from pathlib import Path
from collections import OrderedDict

from .coralnpu_detection import CoralNPUDetector, detect_coral_npu

logger = logging.getLogger(__name__)


class ModelCache:
    """LRU cache for compiled Edge TPU models."""
    
    def __init__(self, max_size: int = 10):
        """
        Initialize model cache.
        
        Args:
            max_size: Maximum number of models to cache
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.access_times: Dict[str, float] = {}
    
    def get(self, model_path: str) -> Optional[Any]:
        """Get cached interpreter for model.
        
        Args:
            model_path: Path to model
        
        Returns:
            Cached interpreter or None
        """
        if model_path in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(model_path)
            self.access_times[model_path] = time.time()
            return self.cache[model_path].get("interpreter")
        return None
    
    def put(self, model_path: str, interpreter: Any):
        """Cache interpreter for model.
        
        Args:
            model_path: Path to model
            interpreter: Compiled interpreter to cache
        """
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and model_path not in self.cache:
            # Remove least recently used
            oldest = next(iter(self.cache))
            del self.cache[oldest]
            del self.access_times[oldest]
        
        self.cache[model_path] = {"interpreter": interpreter, "cached_at": time.time()}
        self.access_times[model_path] = time.time()
        # Move to end (most recently used)
        self.cache.move_to_end(model_path)
    
    def clear(self):
        """Clear all cached models."""
        self.cache.clear()
        self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "cached_models": list(self.cache.keys())
        }


class CoralNPUClient:
    """Client for Coral NPU operations: quantization, compilation, and inference."""
    
    def __init__(
        self,
        detector: Optional[CoralNPUDetector] = None,
        enabled: Optional[bool] = None,
        fallback_to_cpu: bool = True,
        metrics_collector: Optional[Callable[[str, str, float, bool], None]] = None,
    ):
        """
        Initialize Coral NPU client.
        
        Args:
            detector: Optional Coral NPU detector (auto-detected if None)
            enabled: Whether Coral NPU is enabled (from env if None)
            fallback_to_cpu: Whether to fallback to CPU/GPU if NPU unavailable
            metrics_collector: Optional metrics collector function
        """
        self.detector = detector or detect_coral_npu()
        self.enabled = enabled if enabled is not None else (
            os.getenv("CORALNPU_ENABLED", "false").lower() == "true"
        )
        self.fallback_to_cpu = fallback_to_cpu or (
            os.getenv("CORALNPU_FALLBACK_TO_CPU", "true").lower() == "true"
        )
        self.metrics_collector = metrics_collector
        
        # Model cache
        cache_enabled = os.getenv("CORALNPU_CACHE_ENABLED", "true").lower() == "true"
        cache_size = int(os.getenv("CORALNPU_CACHE_SIZE", "10"))
        self.cache = ModelCache(max_size=cache_size) if cache_enabled else None
        
        # Runtime imports (optional)
        self._pycoral = None
        self._edgetpu = None
        self._tflite = None
        
        if self.enabled:
            self._try_import_runtime()
    
    def _try_import_runtime(self):
        """Try to import Coral NPU runtime libraries."""
        try:
            import pycoral
            import pycoral.utils.edgetpu as edgetpu
            self._pycoral = pycoral
            self._edgetpu = edgetpu
            logger.info("Coral NPU runtime libraries imported successfully")
        except ImportError:
            logger.warning("Coral NPU runtime (pycoral) not available")
        
        try:
            import tensorflow.lite as tflite
            self._tflite = tflite
        except ImportError:
            logger.warning("TensorFlow Lite not available")
    
    def is_available(self) -> bool:
        """
        Check if Coral NPU is available for use.
        
        Returns:
            True if NPU is available and enabled, False otherwise
        """
        if not self.enabled:
            return False
        return self.detector.is_available()
    
    def can_use_npu(self) -> bool:
        """
        Check if we can use NPU (available and has permissions).
        
        Returns:
            True if NPU can be used, False otherwise
        """
        if not self.is_available():
            return False
        return self.detector.check_permissions()
    
    def quantize_model(
        self,
        model_path: str,
        output_path: Optional[str] = None,
        representative_dataset: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Quantize a model for Coral NPU.
        
        Args:
            model_path: Path to input model (TensorFlow/Keras)
            output_path: Optional output path for quantized model
            representative_dataset: Optional representative dataset for quantization
            
        Returns:
            Path to quantized model, or None if quantization failed
        """
        if not self.enabled:
            logger.warning("Coral NPU not enabled, skipping quantization")
            return None
        
        if not self.is_available():
            logger.warning("Coral NPU not available, skipping quantization")
            return None
        
        try:
            # Use TensorFlow Lite quantization
            import tensorflow as tf
            
            converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if representative_dataset is not None:
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            
            tflite_model = converter.convert()
            
            if output_path is None:
                output_path = str(Path(model_path).with_suffix(".tflite"))
            
            with open(output_path, "wb") as f:
                f.write(tflite_model)
            
            logger.info(f"Model quantized successfully: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Model quantization failed: {e}")
            return None
    
    def compile_model(
        self,
        tflite_model_path: str,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Compile a TensorFlow Lite model for Coral NPU.
        
        Args:
            tflite_model_path: Path to quantized TFLite model
            output_path: Optional output path for compiled model
            
        Returns:
            Path to compiled model, or None if compilation failed
        """
        if not self.enabled:
            logger.warning("Coral NPU not enabled, skipping compilation")
            return None
        
        if not self.is_available():
            logger.warning("Coral NPU not available, skipping compilation")
            return None
        
        try:
            # Use Edge TPU compiler
            import subprocess
            
            if output_path is None:
                output_path = str(Path(tflite_model_path).with_suffix(".edgetpu.tflite"))
            
            # Check if edgetpu_compiler is available
            compiler_path = os.getenv("EDGETPU_COMPILER", "edgetpu_compiler")
            
            result = subprocess.run(
                [compiler_path, "-s", tflite_model_path, "-o", output_path],
                capture_output=True,
                text=True,
            )
            
            if result.returncode == 0:
                logger.info(f"Model compiled successfully: {output_path}")
                return output_path
            else:
                logger.error(f"Model compilation failed: {result.stderr}")
                return None
        
        except FileNotFoundError:
            logger.error("Edge TPU compiler not found. Install: apt-get install edgetpu-compiler")
            return None
        except Exception as e:
            logger.error(f"Model compilation failed: {e}")
            return None
    
    def create_inference_engine(
        self,
        model_path: str,
        fallback_to_cpu: Optional[bool] = None,
    ) -> Optional[Any]:
        """
        Create an inference engine for Coral NPU.
        
        Args:
            model_path: Path to compiled Edge TPU model
            fallback_to_cpu: Whether to fallback to CPU (uses instance default if None)
            
        Returns:
            Inference engine instance, or None if unavailable
        """
        fallback = fallback_to_cpu if fallback_to_cpu is not None else self.fallback_to_cpu
        
        # Check cache first
        if self.cache:
            cached_interpreter = self.cache.get(model_path)
            if cached_interpreter is not None:
                logger.info(f"Using cached interpreter for {model_path}")
                return cached_interpreter
        
        if not self.can_use_npu():
            if fallback:
                logger.info("Coral NPU not available, using CPU fallback")
                return self._create_cpu_inference_engine(model_path)
            else:
                logger.warning("Coral NPU not available and fallback disabled")
                return None
        
        try:
            if self._pycoral is None or self._edgetpu is None:
                if fallback:
                    return self._create_cpu_inference_engine(model_path)
                return None
            
            # Create Edge TPU interpreter
            interpreter = self._edgetpu.make_interpreter(model_path)
            interpreter.allocate_tensors()
            
            # Cache the interpreter
            if self.cache:
                self.cache.put(model_path, interpreter)
            
            logger.info(f"Coral NPU inference engine created for {model_path}")
            return interpreter
        
        except Exception as e:
            logger.error(f"Failed to create Coral NPU inference engine: {e}")
            if fallback:
                logger.info("Falling back to CPU inference")
                return self._create_cpu_inference_engine(model_path)
            return None
    
    def _create_cpu_inference_engine(self, model_path: str) -> Optional[Any]:
        """Create CPU inference engine as fallback."""
        try:
            if self._tflite is None:
                logger.warning("TensorFlow Lite not available for CPU fallback")
                return None
            
            interpreter = self._tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            logger.info("CPU inference engine created as fallback")
            return interpreter
        
        except Exception as e:
            logger.error(f"Failed to create CPU inference engine: {e}")
            return None
    
    def run_inference(
        self,
        interpreter: Any,
        input_data: Any,
        collect_metrics: bool = True,
    ) -> Optional[Any]:
        """
        Run inference using the provided interpreter.
        
        Args:
            interpreter: Inference engine (NPU or CPU)
            input_data: Input data for inference
            collect_metrics: Whether to collect metrics
            
        Returns:
            Inference result, or None if inference failed
        """
        import time
        
        start_time = time.time()
        is_npu = self.can_use_npu()
        
        try:
            # Get input and output tensors
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Set input
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            latency = time.time() - start_time
            
            if collect_metrics and self.metrics_collector:
                self.metrics_collector("coralnpu", "inference", latency, is_npu)
            
            return output_data
        
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None
    
    def run_batch_inference(
        self,
        interpreter: Any,
        input_batch: List[Any],
        batch_size: Optional[int] = None,
        collect_metrics: bool = True,
    ) -> Optional[List[Any]]:
        """
        Run batch inference using the provided interpreter.
        
        Args:
            interpreter: Inference engine (NPU or CPU)
            input_batch: List of input data for batch inference
            batch_size: Batch size (auto-detect if None)
            collect_metrics: Whether to collect metrics
        
        Returns:
            List of inference results, or None if inference failed
        """
        import time
        import numpy as np
        
        if not input_batch:
            return []
        
        # Auto-detect batch size based on NPU memory if not specified
        if batch_size is None:
            batch_size = int(os.getenv("CORALNPU_BATCH_SIZE", "1"))
            # Edge TPU typically supports batch size 1, but we can process multiple sequentially
            if batch_size <= 0:
                batch_size = 1
        
        start_time = time.time()
        is_npu = self.can_use_npu()
        
        try:
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Determine if model supports native batching
            input_shape = input_details[0]['shape']
            supports_native_batch = len(input_shape) > 0 and input_shape[0] is None or input_shape[0] > 1
            
            results = []
            
            if supports_native_batch and len(input_batch) <= batch_size:
                # Try native batch inference
                try:
                    # Stack inputs into batch
                    batch_input = np.stack(input_batch, axis=0)
                    
                    # Ensure batch dimension matches
                    if input_shape[0] is not None and batch_input.shape[0] != input_shape[0]:
                        # Pad or truncate to match
                        if batch_input.shape[0] < input_shape[0]:
                            padding = np.zeros((input_shape[0] - batch_input.shape[0],) + batch_input.shape[1:])
                            batch_input = np.concatenate([batch_input, padding], axis=0)
                        else:
                            batch_input = batch_input[:input_shape[0]]
                    
                    interpreter.set_tensor(input_details[0]['index'], batch_input)
                    interpreter.invoke()
                    batch_output = interpreter.get_tensor(output_details[0]['index'])
                    
                    # Split batch output
                    for i in range(len(input_batch)):
                        results.append(batch_output[i])
                    
                except Exception as e:
                    logger.warning(f"Native batch inference failed: {e}, falling back to sequential")
                    supports_native_batch = False
            
            if not supports_native_batch:
                # Sequential batch processing
                for i in range(0, len(input_batch), batch_size):
                    batch = input_batch[i:i + batch_size]
                    
                    for input_data in batch:
                        # Ensure input shape matches (remove batch dimension if needed)
                        if isinstance(input_data, np.ndarray) and len(input_data.shape) > 1:
                            # Check if first dimension is batch
                            if input_data.shape[0] == 1:
                                input_data = input_data[0]
                        
                        interpreter.set_tensor(input_details[0]['index'], input_data)
                        interpreter.invoke()
                        output_data = interpreter.get_tensor(output_details[0]['index'])
                        results.append(output_data)
            
            latency = time.time() - start_time
            
            if collect_metrics and self.metrics_collector:
                self.metrics_collector("coralnpu", "batch_inference", latency, is_npu)
            
            logger.info(f"Batch inference completed: {len(results)} samples in {latency:.3f}s")
            return results
        
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get model cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if self.cache:
            return self.cache.get_stats()
        return {"enabled": False}
    
    def clear_cache(self):
        """Clear model cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Model cache cleared")


def create_coralnpu_client(
    enabled: Optional[bool] = None,
    fallback_to_cpu: bool = True,
    metrics_collector: Optional[Callable[[str, str, float, bool], None]] = None,
) -> CoralNPUClient:
    """
    Create a Coral NPU client instance.
    
    Args:
        enabled: Whether Coral NPU is enabled
        fallback_to_cpu: Whether to fallback to CPU
        metrics_collector: Optional metrics collector
        
    Returns:
        CoralNPUClient instance
    """
    return CoralNPUClient(
        enabled=enabled,
        fallback_to_cpu=fallback_to_cpu,
        metrics_collector=metrics_collector,
    )

