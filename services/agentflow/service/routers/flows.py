from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
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


@router.post("/{flow_id:path}/run")
async def run_flow(
    flow_id: str,
    payload: FlowRunRequestSchema,
    catalog: FlowCatalog = Depends(get_catalog),
    langflow: LangflowClient = Depends(get_langflow_client),
    registry: FlowRegistryService = Depends(get_registry_service),
) -> dict:
    try:
        spec = await run_in_threadpool(lambda: catalog.get(flow_id))
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

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
    return {
        "local_id": spec.id,
        "remote_id": target_id,
        "result": result,
    }
