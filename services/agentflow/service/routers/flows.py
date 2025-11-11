from __future__ import annotations

import asyncio
import os
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from ..dependencies import get_catalog, get_langflow_client, get_registry_service
from ..schemas import FlowInfo, FlowRunRequestSchema, FlowSyncRequest, FlowSyncResponse
from ..services import FlowCatalog, FlowRegistryService
from ..services.langflow import FlowImportRequest, LangflowClient, RunFlowRequest

router = APIRouter(prefix="/flows", tags=["flows"])


def _mapping_to_info(mapping) -> FlowInfo:
    return FlowInfo(
        local_id=mapping.local_id,
        remote_id=mapping.remote_id,
        name=mapping.name,
        description=mapping.description,
        project_id=mapping.project_id,
        folder_path=mapping.folder_path,
        updated_at=mapping.updated_at,
        synced_at=mapping.synced_at,
    )


@router.get("", response_model=list[FlowInfo])
async def list_flows(
    catalog: FlowCatalog = Depends(get_catalog),
    registry: FlowRegistryService = Depends(get_registry_service),
) -> list[FlowInfo]:
    specs = await run_in_threadpool(catalog.list)
    results: list[FlowInfo] = []
    for spec in specs:
        mapping = await registry.get(spec.id)
        if mapping:
            results.append(_mapping_to_info(mapping))
        else:
            try:
                folder = str(spec.path.relative_to(catalog.root).parent)
            except ValueError:
                folder = None
            results.append(
                FlowInfo(
                    local_id=spec.id,
                    name=spec.name,
                    description=spec.description,
                    project_id=spec.raw.get("project_id"),
                    folder_path=folder,
                )
            )
    return results


@router.post("/{flow_id:path}/sync", response_model=FlowSyncResponse)
async def sync_flow(
    flow_id: str,
    payload: FlowSyncRequest,
    catalog: FlowCatalog = Depends(get_catalog),
    langflow: LangflowClient = Depends(get_langflow_client),
    registry: FlowRegistryService = Depends(get_registry_service),
) -> FlowSyncResponse:
    try:
        spec = await run_in_threadpool(lambda: catalog.get(flow_id))
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    existing_mapping = await registry.get(flow_id)

    import_request = FlowImportRequest(
        flow=spec.raw,
        force=payload.force,
        project_id=payload.project_id,
        folder_path=payload.folder_path,
        remote_id=existing_mapping.remote_id if existing_mapping else None,
    )
    remote_record = await langflow.import_flow(import_request)
    mapping = await registry.upsert_from_flow(spec=spec, remote=remote_record)
    info = _mapping_to_info(mapping)
    return FlowSyncResponse(**info.dict(), status="synced")


@router.get("/{flow_id:path}", response_model=FlowInfo)
async def get_flow(
    flow_id: str,
    catalog: FlowCatalog = Depends(get_catalog),
    registry: FlowRegistryService = Depends(get_registry_service),
) -> FlowInfo:
    mapping = await registry.get(flow_id)
    if mapping:
        return _mapping_to_info(mapping)
    try:
        spec = await run_in_threadpool(lambda: catalog.get(flow_id))
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    try:
        folder = str(spec.path.relative_to(catalog.root).parent)
    except ValueError:
        folder = None
    return FlowInfo(
        local_id=spec.id,
        name=spec.name,
        description=spec.description,
        project_id=spec.raw.get("project_id"),
        folder_path=folder,
    )


def _has_llm_nodes(flow_spec: dict) -> bool:
    """Check if flow contains LLM nodes that would benefit from GPU allocation (Priority 4)."""
    if not flow_spec:
        return False
    
    # Check for common LLM node types in LangFlow
    llm_node_types = [
        "LLMChain", "ChatOpenAI", "OpenAI", "ChatModel", "LLM",
        "ChatLocalAI", "LocalAI", "ChatAnthropic", "Anthropic",
        "ChatVertexAI", "VertexAI", "ChatCohere", "Cohere"
    ]
    
    # Check nodes in flow
    nodes = flow_spec.get("nodes", [])
    if isinstance(nodes, list):
        for node in nodes:
            if isinstance(node, dict):
                node_type = node.get("type", "") or node.get("data", {}).get("type", "")
                if any(llm_type in node_type for llm_type in llm_node_types):
                    return True
    
    # Check for LLM-related tags or metadata
    tags = flow_spec.get("tags", [])
    if isinstance(tags, list):
        llm_tags = ["llm", "localai", "openai", "chat", "inference"]
        if any(tag.lower() in llm_tags for tag in tags if isinstance(tag, str)):
            return True
    
    return False


async def _request_gpu_allocation(
    flow_id: str,
    workflow_id: Optional[str] = None,
    workflow_priority: Optional[int] = None,
) -> Optional[str]:
    """Request GPU allocation for LLM-intensive flow (Priority 4)."""
    gpu_orchestrator_url = os.getenv("GPU_ORCHESTRATOR_URL", "http://gpu-orchestrator:8086")
    if not gpu_orchestrator_url:
        return None
    
    workload_data = {
        "flow_id": flow_id,
        "workload_type": "inference",
        "required_gpus": 1,
        "min_memory_mb": 4096,
        "priority": workflow_priority or 6,  # Default medium-high priority
    }
    
    if workflow_id:
        workload_data["workflow_id"] = workflow_id
    
    request_body = {
        "service_name": "agentflow",
        "workload_type": "inference",
        "workload_data": workload_data,
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{gpu_orchestrator_url}/gpu/allocate",
                json=request_body,
            )
            if resp.status_code == 200:
                result = resp.json()
                return result.get("id")
    except Exception:
        pass  # Non-fatal - continue without GPU
    
    return None


@router.post("/{flow_id:path}/run")
async def run_flow(
    flow_id: str,
    payload: FlowRunRequestSchema,
    catalog: FlowCatalog = Depends(get_catalog),
    langflow: LangflowClient = Depends(get_langflow_client),
    registry: FlowRegistryService = Depends(get_registry_service),
    request: Request = None,
) -> dict:
    try:
        spec = await run_in_threadpool(lambda: catalog.get(flow_id))
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    # Priority 4: Check if flow needs GPU allocation and request it
    gpu_allocation_id = None
    if _has_llm_nodes(spec.raw):
        # Extract workflow context from headers if available
        workflow_id = None
        workflow_priority = None
        if request:
            workflow_id = request.headers.get("X-Workflow-ID")
            priority_str = request.headers.get("X-Workflow-Priority")
            if priority_str:
                try:
                    workflow_priority = int(priority_str)
                except ValueError:
                    pass
        
        gpu_allocation_id = await _request_gpu_allocation(
            flow_id=flow_id,
            workflow_id=workflow_id,
            workflow_priority=workflow_priority,
        )
        if gpu_allocation_id:
            # Store allocation ID for cleanup after flow execution
            # Note: In production, use proper async context manager or background task
            pass  # GPU will be released by orchestrator timeout or explicit release

    mapping = None
    if payload.ensure:
        existing_mapping = await registry.get(spec.id)
        import_request = FlowImportRequest(
            flow=spec.raw,
            force=True,
            project_id=spec.raw.get("project_id"),
            folder_path=None,
            remote_id=existing_mapping.remote_id if existing_mapping else None,
        )
        remote_record = await langflow.import_flow(import_request)
        mapping = await registry.upsert_from_flow(spec=spec, remote=remote_record)
    else:
        mapping = await registry.get(spec.id)
        if mapping is None or not mapping.remote_id:
            raise HTTPException(
                status_code=status.HTTP_412_PRECONDITION_FAILED,
                detail="Flow is not synchronised with Langflow. Run with ensure=true or sync first.",
            )

    target_id = mapping.remote_id or spec.id
    run_request = RunFlowRequest(
        input_value=payload.input_value,
        inputs=payload.inputs,
        chat_history=payload.chat_history,
        session_id=payload.session_id,
        tweaks=payload.tweaks,
        stream=payload.stream,
    )
    result = await langflow.run_flow(target_id, run_request)
    
    # Priority 4: Release GPU allocation after flow execution
    if gpu_allocation_id:
        try:
            gpu_orchestrator_url = os.getenv("GPU_ORCHESTRATOR_URL", "http://gpu-orchestrator:8086")
            if gpu_orchestrator_url:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(
                        f"{gpu_orchestrator_url}/gpu/release",
                        json={"allocation_id": gpu_allocation_id, "service_name": "agentflow"},
                    )
        except Exception:
            pass  # Non-fatal - GPU will be released by orchestrator timeout

    # Optional DeepAgents analysis (if enabled)
    from ..deepagents import analyze_flow_execution

    deepagents_analysis = await analyze_flow_execution(
        flow_id=flow_id,
        flow_result=result,
        input_value=payload.input_value,
        inputs=payload.inputs,
    )

    response = {
        "local_id": spec.id,
        "remote_id": target_id,
        "result": result,
    }

    # Add DeepAgents analysis if available
    if deepagents_analysis:
        response["deepagents_analysis"] = deepagents_analysis

    return response
