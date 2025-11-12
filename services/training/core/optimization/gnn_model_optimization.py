"""Model Optimization for GNN.

This module provides model quantization, pruning, and inference optimization.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.quantization as quantization
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    onnx = None
    ort = None

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
                # Prepare model for QAT by inserting fake quantization layers
                try:
                    model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
                    quantization.prepare_qat(model, inplace=True)
                    logger.info("Model prepared for Quantization-Aware Training (QAT)")
                    logger.warning("Model must be trained before conversion to quantized model")
                    return model
                except Exception as e:
                    logger.warning(f"QAT preparation failed: {e}, using dynamic quantization")
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
    
    def convert_qat_to_quantized(
        self,
        qat_model: nn.Module
    ) -> nn.Module:
        """Convert QAT model to quantized model after training.
        
        Args:
            qat_model: Model that has been trained with QAT
        
        Returns:
            Fully quantized model ready for inference
        """
        if not HAS_TORCH:
            return qat_model
        
        try:
            qat_model.eval()
            
            # Convert QAT model to quantized model
            quantized_model = quantization.convert(qat_model, inplace=False)
            logger.info("QAT model converted to quantized model")
            return quantized_model
        
        except Exception as e:
            logger.error(f"Failed to convert QAT model: {e}")
            return qat_model
    
    def train_qat_model(
        self,
        qat_model: nn.Module,
        train_loader: Any,
        optimizer: Any,
        criterion: Any,
        num_epochs: int = 5,
        device: Optional[str] = None
    ) -> nn.Module:
        """Train a QAT model.
        
        Args:
            qat_model: Model prepared for QAT
            train_loader: Training data loader
            optimizer: Optimizer for training
            criterion: Loss function
            num_epochs: Number of training epochs
            device: Device to use
        
        Returns:
            Trained QAT model
        """
        if not HAS_TORCH:
            return qat_model
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        qat_model.train()
        qat_model = qat_model.to(device)
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs = batch.x
                    targets = batch.y if hasattr(batch, 'y') else None
                    edge_index = batch.edge_index if hasattr(batch, 'edge_index') else None
                
                inputs = inputs.to(device)
                if targets is not None:
                    targets = targets.to(device)
                
                # Forward pass
                if edge_index is not None:
                    edge_index = edge_index.to(device)
                    outputs = qat_model(inputs, edge_index)
                else:
                    outputs = qat_model(inputs)
                
                if targets is not None:
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"QAT Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        qat_model.eval()
        logger.info("QAT training completed")
        return qat_model


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
    
    def compute_sensitivity(
        self,
        model: nn.Module,
        criterion: Any,
        data_loader: Any,
        device: Optional[str] = None
    ) -> Dict[str, float]:
        """Compute sensitivity of each layer to pruning.
        
        Args:
            model: Model to analyze
            criterion: Loss function
            data_loader: Data loader for evaluation
            device: Device to use
        
        Returns:
            Dictionary mapping layer names to sensitivity scores
        """
        if not HAS_TORCH:
            return {}
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model.eval()
        model = model.to(device)
        
        # Get baseline loss
        baseline_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch.x, batch.y
                
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs, batch.edge_index.to(device) if hasattr(batch, 'edge_index') else None)
                loss = criterion(outputs, targets)
                baseline_loss += loss.item()
                num_batches += 1
        
        baseline_loss /= num_batches if num_batches > 0 else 1.0
        
        # Compute sensitivity for each layer
        sensitivities = {}
        from torch.nn.utils import prune
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Temporarily prune 10% and measure impact
                temp_module = module
                original_weight = temp_module.weight.data.clone()
                
                # Prune 10%
                prune.l1_unstructured(temp_module, name='weight', amount=0.1)
                
                # Measure loss increase
                perturbed_loss = 0.0
                num_batches = 0
                with torch.no_grad():
                    for batch in data_loader:
                        if isinstance(batch, (list, tuple)):
                            inputs, targets = batch[0], batch[1]
                        else:
                            inputs, targets = batch.x, batch.y
                        
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        
                        outputs = model(inputs, batch.edge_index.to(device) if hasattr(batch, 'edge_index') else None)
                        loss = criterion(outputs, targets)
                        perturbed_loss += loss.item()
                        num_batches += 1
                
                perturbed_loss /= num_batches if num_batches > 0 else 1.0
                
                # Sensitivity = relative loss increase
                sensitivity = (perturbed_loss - baseline_loss) / baseline_loss if baseline_loss > 0 else 0.0
                sensitivities[name] = sensitivity
                
                # Restore original weights
                temp_module.weight.data = original_weight
                prune.remove(temp_module, 'weight')
        
        return sensitivities
    
    def iterative_prune(
        self,
        model: nn.Module,
        target_pruning_ratio: float,
        criterion: Optional[Any] = None,
        data_loader: Optional[Any] = None,
        fine_tune_fn: Optional[Any] = None,
        step_size: float = 0.1,
        min_accuracy_drop: float = 0.05,
        device: Optional[str] = None
    ) -> Tuple[nn.Module, List[Dict[str, Any]]]:
        """Iteratively prune model with sensitivity analysis.
        
        Args:
            model: Model to prune
            target_pruning_ratio: Target pruning ratio (0.0 to 1.0)
            criterion: Loss function for sensitivity analysis
            data_loader: Data loader for evaluation
            fine_tune_fn: Function to fine-tune model after pruning (optional)
            step_size: Pruning step size per iteration (default: 0.1 = 10%)
            min_accuracy_drop: Maximum acceptable accuracy drop (default: 0.05 = 5%)
            device: Device to use
        
        Returns:
            Tuple of (pruned_model, pruning_history)
        """
        if not HAS_TORCH:
            return model, []
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = model.to(device)
        current_pruning_ratio = 0.0
        pruning_history = []
        
        from torch.nn.utils import prune
        
        while current_pruning_ratio < target_pruning_ratio:
            # Compute sensitivity if data available
            sensitivities = {}
            if criterion is not None and data_loader is not None:
                sensitivities = self.compute_sensitivity(model, criterion, data_loader, device)
            
            # Determine pruning amount for this step
            remaining_to_prune = target_pruning_ratio - current_pruning_ratio
            step_pruning = min(step_size, remaining_to_prune)
            
            # Prune based on sensitivity (prune less sensitive layers more)
            if sensitivities:
                # Sort layers by sensitivity (ascending = less sensitive first)
                sorted_layers = sorted(sensitivities.items(), key=lambda x: x[1])
                
                # Prune less sensitive layers more aggressively
                for layer_name, sensitivity in sorted_layers:
                    if current_pruning_ratio >= target_pruning_ratio:
                        break
                    
                    # Adjust pruning amount based on sensitivity
                    layer_pruning = step_pruning * (1.0 - sensitivity)  # Less sensitive = prune more
                    layer_pruning = min(layer_pruning, remaining_to_prune)
                    
                    # Find and prune the layer
                    for name, module in model.named_modules():
                        if name == layer_name and isinstance(module, nn.Linear):
                            prune.l1_unstructured(module, name='weight', amount=layer_pruning)
                            prune.remove(module, 'weight')
                            break
            else:
                # No sensitivity data, use uniform pruning
                for module in model.modules():
                    if isinstance(module, nn.Linear):
                        prune.l1_unstructured(module, name='weight', amount=step_pruning)
                        prune.remove(module, 'weight')
            
            # Update current pruning ratio
            stats = self.get_pruning_stats(model)
            current_pruning_ratio = stats["pruning_ratio"]
            
            # Fine-tune if function provided
            if fine_tune_fn is not None:
                model = fine_tune_fn(model)
            
            # Record pruning step
            pruning_history.append({
                "pruning_ratio": current_pruning_ratio,
                "step": len(pruning_history) + 1,
                "stats": stats
            })
            
            logger.info(f"Iterative pruning step {len(pruning_history)}: {current_pruning_ratio*100:.1f}% pruned")
            
            # Check if we've reached target
            if current_pruning_ratio >= target_pruning_ratio:
                break
        
        logger.info(f"Iterative pruning completed: {current_pruning_ratio*100:.1f}% pruned")
        return model, pruning_history


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
    
    def export_to_onnx(
        self,
        model: nn.Module,
        output_path: str,
        example_input: Optional[Any] = None,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 11,
        verbose: bool = False
    ) -> bool:
        """Export GNN model to ONNX format.
        
        Args:
            model: PyTorch model to export
            output_path: Path to save ONNX model
            example_input: Example input for tracing (required for GNN models)
            input_names: Names for input tensors (default: ["x", "edge_index"])
            output_names: Names for output tensors (default: ["output"])
            dynamic_axes: Dictionary specifying dynamic axes (e.g., {"x": {0: "batch_size"}})
            opset_version: ONNX opset version (default: 11)
            verbose: Whether to print verbose output
        
        Returns:
            True if export successful, False otherwise
        """
        if not HAS_TORCH:
            logger.error("PyTorch not available, cannot export to ONNX")
            return False
        
        if not HAS_ONNX:
            logger.warning("ONNX not available, install with: pip install onnx onnxruntime")
            return False
        
        try:
            model.eval()
            
            # Default input names for GNN models
            if input_names is None:
                input_names = ["x", "edge_index"]
            if output_names is None:
                output_names = ["output"]
            
            # Create example input if not provided
            if example_input is None:
                # Try to detect GNN model structure
                if hasattr(model, 'num_node_features'):
                    num_features = model.num_node_features
                else:
                    num_features = 64  # Default
                
                num_nodes = 10
                num_edges = 20
                
                example_x = torch.randn(num_nodes, num_features)
                example_edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
                example_input = (example_x, example_edge_index)
            
            # Default dynamic axes for variable graph sizes
            if dynamic_axes is None:
                dynamic_axes = {
                    "x": {0: "num_nodes"},
                    "edge_index": {1: "num_edges"},
                    "output": {0: "num_nodes"}
                }
            
            # Export to ONNX
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    example_input,
                    output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    verbose=verbose
                )
            
            logger.info(f"Model exported to ONNX: {output_path}")
            
            # Verify ONNX model
            if HAS_ONNX:
                try:
                    onnx_model = onnx.load(output_path)
                    onnx.checker.check_model(onnx_model)
                    logger.info("ONNX model verification passed")
                except Exception as e:
                    logger.warning(f"ONNX model verification failed: {e}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")
            return False
    
    def load_onnx_model(
        self,
        onnx_path: str,
        providers: Optional[List[str]] = None
    ) -> Optional[Any]:
        """Load ONNX model for inference.
        
        Args:
            onnx_path: Path to ONNX model
            providers: Execution providers (default: ["CPUExecutionProvider"])
        
        Returns:
            ONNX Runtime InferenceSession or None if failed
        """
        if not HAS_ONNX:
            logger.error("ONNX Runtime not available")
            return None
        
        try:
            if providers is None:
                providers = ["CPUExecutionProvider"]
                if HAS_TORCH and torch.cuda.is_available():
                    providers.insert(0, "CUDAExecutionProvider")
            
            session = ort.InferenceSession(onnx_path, providers=providers)
            logger.info(f"ONNX model loaded: {onnx_path}")
            return session
        
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return None

