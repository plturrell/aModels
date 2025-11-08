"""Model Optimization for GNN.

This module provides model quantization, pruning, and inference optimization.
"""

import logging
import os
from typing import Dict, List, Optional, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.quantization as quantization
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class GNNModelQuantizer:
    """Quantize GNN models for faster inference."""
    
    def __init__(
        self,
        quantization_type: str = "dynamic"  # "dynamic", "static", "qat"
    ):
        """Initialize quantizer.
        
        Args:
            quantization_type: Type of quantization ("dynamic", "static", "qat")
        """
        self.quantization_type = quantization_type
        
        if not HAS_TORCH:
            logger.warning("PyTorch not available, quantization disabled")
    
    def quantize_model(
        self,
        model: nn.Module,
        example_input: Optional[Any] = None
    ) -> nn.Module:
        """Quantize a model.
        
        Args:
            model: Model to quantize
            example_input: Example input for static quantization
        
        Returns:
            Quantized model
        """
        if not HAS_TORCH:
            return model
        
        try:
            model.eval()
            
            if quantization is None:
                logger.warning("Quantization not available, returning original model")
                return model
            
            if self.quantization_type == "dynamic":
                # Dynamic quantization (no calibration needed)
                quantized_model = quantization.quantize_dynamic(
                    model,
                    {nn.Linear, nn.Conv1d, nn.Conv2d},
                    dtype=torch.qint8
                )
                logger.info("Model quantized using dynamic quantization")
                return quantized_model
            
            elif self.quantization_type == "static":
                # Static quantization (requires calibration)
                if example_input is None:
                    logger.warning("Example input required for static quantization, using dynamic")
                    return self.quantize_model(model, None)  # Fallback to dynamic
                
                # Prepare model
                model.qconfig = quantization.get_default_qconfig('fbgemm')
                quantization.prepare(model, inplace=True)
                
                # Calibrate (run with example inputs)
                # Note: In practice, you'd run calibration on a dataset
                with torch.no_grad():
                    _ = model(example_input)
                
                # Convert to quantized
                quantized_model = quantization.convert(model, inplace=False)
                logger.info("Model quantized using static quantization")
                return quantized_model
            
            elif self.quantization_type == "qat":
                # Quantization-Aware Training (requires training)
                logger.warning("QAT requires training, using dynamic quantization")
                return self.quantize_model(model, None)  # Fallback to dynamic
            
            else:
                logger.warning(f"Unknown quantization type: {self.quantization_type}")
                return model
        
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model
    
    def save_quantized_model(
        self,
        model: nn.Module,
        model_path: str
    ):
        """Save quantized model.
        
        Args:
            model: Quantized model
            model_path: Path to save model
        """
        try:
            torch.save(model.state_dict(), model_path)
            logger.info(f"Quantized model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save quantized model: {e}")
    
    def load_quantized_model(
        self,
        model: nn.Module,
        model_path: str
    ) -> nn.Module:
        """Load quantized model.
        
        Args:
            model: Base model architecture
            model_path: Path to quantized model
        
        Returns:
            Loaded quantized model
        """
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            logger.info(f"Quantized model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load quantized model: {e}")
            return model


class GNNModelPruner:
    """Prune GNN models for smaller size."""
    
    def __init__(
        self,
        pruning_method: str = "magnitude"  # "magnitude", "structured", "unstructured"
    ):
        """Initialize pruner.
        
        Args:
            pruning_method: Pruning method
        """
        self.pruning_method = pruning_method
        
        if not HAS_TORCH:
            logger.warning("PyTorch not available, pruning disabled")
    
    def prune_model(
        self,
        model: nn.Module,
        pruning_amount: float = 0.2  # 20% of weights
    ) -> nn.Module:
        """Prune a model.
        
        Args:
            model: Model to prune
            pruning_amount: Fraction of weights to prune (0.0 to 1.0)
        
        Returns:
            Pruned model
        """
        if not HAS_TORCH:
            return model
        
        try:
            if self.pruning_method == "magnitude":
                # Magnitude-based pruning
                from torch.nn.utils import prune
                
                # Prune all linear layers
                for module in model.modules():
                    if isinstance(module, nn.Linear):
                        prune.l1_unstructured(
                            module,
                            name='weight',
                            amount=pruning_amount
                        )
                        prune.remove(module, 'weight')  # Make pruning permanent
                
                logger.info(f"Model pruned using magnitude-based method ({pruning_amount*100:.1f}%)")
                return model
            
            elif self.pruning_method == "structured":
                # Structured pruning (removes entire channels/filters)
                from torch.nn.utils import prune
                
                for module in model.modules():
                    if isinstance(module, nn.Linear):
                        prune.ln_structured(
                            module,
                            name='weight',
                            amount=pruning_amount,
                            n=2,
                            dim=0
                        )
                        prune.remove(module, 'weight')
                
                logger.info(f"Model pruned using structured method ({pruning_amount*100:.1f}%)")
                return model
            
            elif self.pruning_method == "unstructured":
                # Unstructured pruning (individual weights)
                from torch.nn.utils import prune
                
                for module in model.modules():
                    if isinstance(module, nn.Linear):
                        prune.random_unstructured(
                            module,
                            name='weight',
                            amount=pruning_amount
                        )
                        prune.remove(module, 'weight')
                
                logger.info(f"Model pruned using unstructured method ({pruning_amount*100:.1f}%)")
                return model
            
            else:
                logger.warning(f"Unknown pruning method: {self.pruning_method}")
                return model
        
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            return model
    
    def get_pruning_stats(
        self,
        model: nn.Module
    ) -> Dict[str, Any]:
        """Get pruning statistics.
        
        Args:
            model: Model to analyze
        
        Returns:
            Dictionary with pruning stats
        """
        total_params = 0
        pruned_params = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                weight = module.weight
                total_params += weight.numel()
                pruned_params += (weight == 0).sum().item()
        
        pruning_ratio = pruned_params / total_params if total_params > 0 else 0.0
        
        return {
            "total_params": total_params,
            "pruned_params": pruned_params,
            "remaining_params": total_params - pruned_params,
            "pruning_ratio": pruning_ratio,
            "compression_ratio": 1.0 / (1.0 - pruning_ratio) if pruning_ratio < 1.0 else float('inf')
        }


class GNNInferenceOptimizer:
    """Optimize GNN inference speed."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        use_jit: bool = True,
        use_torch_compile: bool = False
    ):
        """Initialize inference optimizer.
        
        Args:
            device: Device to use
            use_jit: Whether to use TorchScript JIT
            use_torch_compile: Whether to use torch.compile (PyTorch 2.0+)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_jit = use_jit and HAS_TORCH
        self.use_torch_compile = use_torch_compile and HAS_TORCH
        
        # Check torch.compile availability
        if self.use_torch_compile:
            try:
                # Test if torch.compile is available
                _ = torch.compile
            except AttributeError:
                logger.warning("torch.compile not available (requires PyTorch 2.0+)")
                self.use_torch_compile = False
    
    def optimize_for_inference(
        self,
        model: nn.Module,
        example_input: Optional[Any] = None
    ) -> nn.Module:
        """Optimize model for inference.
        
        Args:
            model: Model to optimize
            example_input: Example input for JIT compilation
        
        Returns:
            Optimized model
        """
        if not HAS_TORCH:
            return model
        
        try:
            model.eval()
            model = model.to(self.device)
            
            # Use torch.compile if available
            if self.use_torch_compile:
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info("Model optimized using torch.compile")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}, trying JIT")
                    self.use_jit = True
            
            # Use TorchScript JIT if available
            if self.use_jit and example_input is not None:
                try:
                    with torch.no_grad():
                        traced_model = torch.jit.trace(model, example_input)
                        traced_model = traced_model.to(self.device)
                        logger.info("Model optimized using TorchScript JIT")
                        return traced_model
                except Exception as e:
                    logger.warning(f"JIT compilation failed: {e}")
            
            return model
        
        except Exception as e:
            logger.error(f"Inference optimization failed: {e}")
            return model
    
    def benchmark_inference(
        self,
        model: nn.Module,
        example_input: Any,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """Benchmark inference speed.
        
        Args:
            model: Model to benchmark
            example_input: Example input
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
        
        Returns:
            Dictionary with benchmark results
        """
        if not HAS_TORCH:
            return {"error": "PyTorch not available"}
        
        try:
            model.eval()
            model = model.to(self.device)
            example_input = example_input.to(self.device) if isinstance(example_input, torch.Tensor) else example_input
            
            # Warmup
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = model(example_input)
            
            # Synchronize if CUDA
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            # Benchmark
            import time
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(example_input)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / num_runs
            throughput = num_runs / total_time
            
            return {
                "total_time_seconds": total_time,
                "avg_time_ms": avg_time * 1000,
                "throughput_inferences_per_sec": throughput,
                "device": self.device,
                "num_runs": num_runs
            }
        
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return {"error": str(e)}


class GNNModelOptimizer:
    """Unified model optimizer combining quantization, pruning, and inference optimization."""
    
    def __init__(
        self,
        quantize: bool = False,
        prune: bool = False,
        optimize_inference: bool = True,
        quantization_type: str = "dynamic",
        pruning_method: str = "magnitude",
        pruning_amount: float = 0.2
    ):
        """Initialize unified optimizer.
        
        Args:
            quantize: Whether to quantize
            prune: Whether to prune
            optimize_inference: Whether to optimize for inference
            quantization_type: Quantization type
            pruning_method: Pruning method
            pruning_amount: Pruning amount
        """
        self.quantize = quantize
        self.prune = prune
        self.optimize_inference = optimize_inference
        
        if quantize:
            self.quantizer = GNNModelQuantizer(quantization_type=quantization_type)
        else:
            self.quantizer = None
        
        if prune:
            self.pruner = GNNModelPruner(pruning_method=pruning_method)
        else:
            self.pruner = None
        
        if optimize_inference:
            self.inference_optimizer = GNNInferenceOptimizer()
        else:
            self.inference_optimizer = None
    
    def optimize_model(
        self,
        model: nn.Module,
        example_input: Optional[Any] = None
    ) -> nn.Module:
        """Apply all optimizations to model.
        
        Args:
            model: Model to optimize
            example_input: Example input for quantization/JIT
        
        Returns:
            Optimized model
        """
        optimized_model = model
        
        # Prune first (before quantization)
        if self.prune and self.pruner:
            optimized_model = self.pruner.prune_model(optimized_model)
        
        # Quantize
        if self.quantize and self.quantizer:
            optimized_model = self.quantizer.quantize_model(optimized_model, example_input)
        
        # Optimize for inference
        if self.optimize_inference and self.inference_optimizer:
            optimized_model = self.inference_optimizer.optimize_for_inference(optimized_model, example_input)
        
        return optimized_model
    
    def get_optimization_stats(
        self,
        original_model: nn.Module,
        optimized_model: nn.Module
    ) -> Dict[str, Any]:
        """Get optimization statistics.
        
        Args:
            original_model: Original model
            optimized_model: Optimized model
        
        Returns:
            Dictionary with optimization stats
        """
        stats = {
            "quantized": self.quantize,
            "pruned": self.prune,
            "inference_optimized": self.optimize_inference
        }
        
        # Model size comparison
        try:
            import tempfile
            import os
            
            # Save models to temp files
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f1:
                torch.save(original_model.state_dict(), f1.name)
                original_size = os.path.getsize(f1.name)
                os.unlink(f1.name)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f2:
                torch.save(optimized_model.state_dict(), f2.name)
                optimized_size = os.path.getsize(f2.name)
                os.unlink(f2.name)
            
            stats["original_size_mb"] = original_size / (1024 * 1024)
            stats["optimized_size_mb"] = optimized_size / (1024 * 1024)
            stats["size_reduction_ratio"] = 1.0 - (optimized_size / original_size) if original_size > 0 else 0.0
        except Exception as e:
            logger.warning(f"Failed to compute size stats: {e}")
        
        # Pruning stats
        if self.prune and self.pruner:
            pruning_stats = self.pruner.get_pruning_stats(optimized_model)
            stats["pruning_stats"] = pruning_stats
        
        return stats

