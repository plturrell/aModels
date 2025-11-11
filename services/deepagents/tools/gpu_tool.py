"""Tools for GPU orchestration and allocation via DeepAgents."""

import os
from typing import Optional, Dict, Any, List
import httpx
from langchain_core.tools import tool

GPU_ORCHESTRATOR_URL = os.getenv("GPU_ORCHESTRATOR_URL", "http://gpu-orchestrator:8088")
_client = httpx.Client(timeout=30.0)


@tool
def allocate_gpu(
    service_name: str,
    workload_type: str = "inference",
    required_gpus: int = 1,
    min_memory_mb: Optional[int] = None,
    priority: int = 5,
    workload_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Allocate GPU resources from the GPU orchestrator.
    
    This tool requests GPU allocation for a service based on workload requirements.
    The orchestrator will analyze the workload and allocate appropriate GPUs.
    
    Args:
        service_name: Name of the service requesting GPUs (e.g., "training", "localai", "sap-rpt")
        workload_type: Type of workload - "training", "inference", "embedding", "ocr", "graph-processing", "agent-workflow", "unified-workflow", or "generic"
        required_gpus: Number of GPUs required (default: 1)
        min_memory_mb: Minimum GPU memory required in MB (optional)
        priority: Priority level (1-10, higher = more important, default: 5)
        workload_data: Optional dictionary with additional workload metadata (e.g., {"model_size": "large", "batch_size": 32})
    
    Returns:
        String containing allocation ID and GPU IDs, or error message
    
    Examples:
        - Allocate 1 GPU for inference: service_name="localai", workload_type="inference", required_gpus=1
        - Allocate 2 GPUs for training: service_name="training", workload_type="training", required_gpus=2, min_memory_mb=8192
        - Allocate GPU for OCR: service_name="deepseek-ocr", workload_type="ocr", required_gpus=1
    """
    try:
        endpoint = f"{GPU_ORCHESTRATOR_URL}/gpu/allocate"
        
        payload: Dict[str, Any] = {
            "service_name": service_name,
            "workload_type": workload_type,
            "workload_data": workload_data or {},
        }
        
        # Add optional fields if provided
        if required_gpus > 0:
            payload["workload_data"]["required_gpus"] = required_gpus
        if min_memory_mb:
            payload["workload_data"]["min_memory_mb"] = min_memory_mb
        if priority:
            payload["workload_data"]["priority"] = priority
        
        response = _client.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if "id" in result and "gpu_ids" in result:
            allocation_id = result["id"]
            gpu_ids = result["gpu_ids"]
            return f"GPU allocation successful. Allocation ID: {allocation_id}, GPU IDs: {gpu_ids}"
        else:
            return f"GPU allocation response: {result}"
    
    except httpx.HTTPStatusError as e:
        return f"Error allocating GPU: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error allocating GPU: {str(e)}"


@tool
def release_gpu(allocation_id: Optional[str] = None, service_name: Optional[str] = None) -> str:
    """Release GPU resources back to the orchestrator.
    
    This tool releases previously allocated GPUs, making them available for other services.
    
    Args:
        allocation_id: Allocation ID to release (if provided, releases specific allocation)
        service_name: Service name to release all allocations for (if allocation_id not provided)
    
    Returns:
        String confirmation of release or error message
    
    Examples:
        - Release specific allocation: allocation_id="training-1234567890"
        - Release all for a service: service_name="training"
    """
    try:
        if allocation_id:
            endpoint = f"{GPU_ORCHESTRATOR_URL}/gpu/release"
            payload = {"allocation_id": allocation_id}
        elif service_name:
            endpoint = f"{GPU_ORCHESTRATOR_URL}/gpu/release"
            payload = {"service_name": service_name}
        else:
            return "Error: Must provide either allocation_id or service_name"
        
        response = _client.post(endpoint, json=payload)
        response.raise_for_status()
        
        return f"GPU resources released successfully for {allocation_id or service_name}"
    
    except httpx.HTTPStatusError as e:
        return f"Error releasing GPU: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error releasing GPU: {str(e)}"


@tool
def query_gpu_status() -> str:
    """Query the current status of all GPUs in the system.
    
    Returns information about GPU availability, utilization, memory usage, and allocations.
    
    Returns:
        String containing GPU status information
    
    Examples:
        - Check GPU status: query_gpu_status()
    """
    try:
        endpoint = f"{GPU_ORCHESTRATOR_URL}/gpu/status"
        
        response = _client.get(endpoint)
        response.raise_for_status()
        
        result = response.json()
        
        # Format result for readability
        if isinstance(result, dict):
            gpus = result.get("gpus", [])
            allocations = result.get("allocations", [])
            
            status_lines = ["GPU Status:"]
            status_lines.append(f"Total GPUs: {len(gpus)}")
            
            for gpu in gpus:
                gpu_id = gpu.get("id", "unknown")
                name = gpu.get("name", "unknown")
                allocated = gpu.get("allocated", False)
                allocated_to = gpu.get("allocated_to", "none")
                memory_used = gpu.get("memory_used", 0)
                memory_total = gpu.get("memory_total", 0)
                utilization = gpu.get("utilization", 0)
                
                status_lines.append(
                    f"  GPU {gpu_id} ({name}): "
                    f"Allocated={allocated}, To={allocated_to}, "
                    f"Memory={memory_used}/{memory_total}MB, "
                    f"Utilization={utilization:.1f}%"
                )
            
            if allocations:
                status_lines.append(f"\nActive Allocations: {len(allocations)}")
                for alloc in allocations:
                    alloc_id = alloc.get("id", "unknown")
                    service = alloc.get("service_name", "unknown")
                    gpu_ids = alloc.get("gpu_ids", [])
                    status_lines.append(f"  {alloc_id}: Service={service}, GPUs={gpu_ids}")
            
            return "\n".join(status_lines)
        
        return str(result)
    
    except httpx.HTTPStatusError as e:
        return f"Error querying GPU status: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error querying GPU status: {str(e)}"


@tool
def analyze_workload(
    workload_type: str,
    workload_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Analyze a workload to determine GPU requirements.
    
    This tool uses the GPU orchestrator's workload analyzer to determine
    how many GPUs and how much memory a workload will need.
    
    Args:
        workload_type: Type of workload - "training", "inference", "embedding", "ocr", "graph-processing", "agent-workflow", "unified-workflow", or "generic"
        workload_data: Optional dictionary with workload metadata (e.g., {"model_size": "large", "batch_size": 32, "image_count": 100})
    
    Returns:
        String containing GPU requirements analysis
    
    Examples:
        - Analyze training workload: workload_type="training", workload_data={"model_size": 10.5}
        - Analyze OCR workload: workload_type="ocr", workload_data={"image_count": 50}
        - Analyze inference workload: workload_type="inference", workload_data={"batch_size": 16}
    """
    try:
        endpoint = f"{GPU_ORCHESTRATOR_URL}/gpu/workload"
        
        payload = {
            "workload_type": workload_type,
            "workload_data": workload_data or {},
        }
        
        response = _client.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if isinstance(result, dict):
            required_gpus = result.get("required_gpus", 0)
            min_memory_mb = result.get("min_memory_mb", 0)
            priority = result.get("priority", 5)
            max_utilization = result.get("max_utilization", 0.9)
            
            return (
                f"Workload Analysis:\n"
                f"  Required GPUs: {required_gpus}\n"
                f"  Minimum Memory: {min_memory_mb} MB\n"
                f"  Priority: {priority}/10\n"
                f"  Max Utilization: {max_utilization*100:.1f}%"
            )
        
        return str(result)
    
    except httpx.HTTPStatusError as e:
        return f"Error analyzing workload: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error analyzing workload: {str(e)}"

