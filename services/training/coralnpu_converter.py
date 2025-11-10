"""Model conversion utilities for Coral NPU."""

import os
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from .coralnpu_client import CoralNPUClient
from .coralnpu_quantization import CoralNPUQuantizer

logger = logging.getLogger(__name__)


class CoralNPUConverter:
    """Converts models to Coral NPU format."""
    
    def __init__(
        self,
        client: Optional[CoralNPUClient] = None,
        quantizer: Optional[CoralNPUQuantizer] = None,
    ):
        """
        Initialize Coral NPU converter.
        
        Args:
            client: Optional Coral NPU client
            quantizer: Optional quantizer
        """
        self.client = client or CoralNPUClient()
        self.quantizer = quantizer or CoralNPUQuantizer(client=self.client)
    
    def convert_pytorch_to_coralnpu(
        self,
        model_path: str,
        output_path: Optional[str] = None,
        representative_dataset: Optional[Any] = None,
        compile_model: bool = True,
    ) -> Optional[str]:
        """
        Convert PyTorch model to Coral NPU format.
        
        Pipeline: PyTorch -> ONNX -> TensorFlow -> TFLite (quantized) -> Edge TPU
        
        Args:
            model_path: Path to PyTorch model
            output_path: Optional output path for final model
            representative_dataset: Optional representative dataset for quantization
            compile_model: Whether to compile model for Edge TPU
            
        Returns:
            Path to converted model, or None if conversion failed
        """
        # Step 1: Quantize model
        tflite_path = self.quantizer.quantize_pytorch_model(
            model_path,
            output_path=None,  # Use temp path
            representative_dataset=representative_dataset,
        )
        
        if tflite_path is None:
            logger.error("Model quantization failed")
            return None
        
        # Step 2: Compile for Edge TPU if requested
        if compile_model:
            final_path = self.client.compile_model(
                tflite_path,
                output_path=output_path,
            )
            
            # Cleanup intermediate TFLite file if we have final path
            if final_path and final_path != tflite_path and os.path.exists(tflite_path):
                os.unlink(tflite_path)
            
            return final_path
        
        # If not compiling, use quantized TFLite model
        if output_path and output_path != tflite_path:
            import shutil
            shutil.copy2(tflite_path, output_path)
            os.unlink(tflite_path)
            return output_path
        
        return tflite_path
    
    def validate_converted_model(
        self,
        model_path: str,
        test_input: Optional[Any] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a converted Coral NPU model.
        
        Args:
            model_path: Path to converted model
            test_input: Optional test input for validation
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Create inference engine
            interpreter = self.client.create_inference_engine(model_path)
            if interpreter is None:
                return False, "Failed to create inference engine"
            
            # Run test inference if test input provided
            if test_input is not None:
                result = self.client.run_inference(interpreter, test_input, collect_metrics=False)
                if result is None:
                    return False, "Test inference failed"
            
            return True, None
        
        except Exception as e:
            return False, f"Validation failed: {e}"
    
    def optimize_for_npu(
        self,
        model_path: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Optimize model for NPU constraints.
        
        Args:
            model_path: Path to model
            constraints: Optional NPU constraints (memory, ops, etc.)
            
        Returns:
            Path to optimized model, or None if optimization failed
        """
        # This is a placeholder - actual implementation would apply NPU-specific optimizations
        # such as operator fusion, memory layout optimization, etc.
        logger.info(f"Optimizing model for NPU: {model_path}")
        
        # For now, just return the original path
        # In production, this would apply actual optimizations
        return model_path


def create_converter(
    client: Optional[CoralNPUClient] = None,
    quantizer: Optional[CoralNPUQuantizer] = None,
) -> CoralNPUConverter:
    """
    Create a Coral NPU converter instance.
    
    Args:
        client: Optional Coral NPU client
        quantizer: Optional quantizer
        
    Returns:
        CoralNPUConverter instance
    """
    return CoralNPUConverter(client=client, quantizer=quantizer)

