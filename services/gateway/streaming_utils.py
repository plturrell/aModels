"""
Streaming utilities for LLM responses.
"""

from typing import AsyncGenerator, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


async def stream_orchestration_response(
    response,
    format_type: str = "text"
) -> AsyncGenerator[str, None]:
    """
    Stream orchestration service response.
    
    Args:
        response: HTTP response from orchestration service
        format_type: Format type ("text", "json", "sse")
        
    Yields:
        Chunks of response data
    """
    if format_type == "sse":
        # Server-Sent Events format
        async for chunk in response.aiter_bytes():
            if chunk:
                yield f"data: {chunk.decode('utf-8')}\n\n"
        yield "data: [DONE]\n\n"
    elif format_type == "json":
        # JSON streaming
        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            try:
                # Try to parse complete JSON objects
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.strip():
                        yield json.dumps({"chunk": line.strip()}) + "\n"
            except:
                pass
    else:
        # Plain text streaming
        async for chunk in response.aiter_text():
            yield chunk


def format_streaming_chunk(chunk: str, chunk_type: str = "text") -> str:
    """
    Format a streaming chunk for SSE.
    
    Args:
        chunk: Chunk content
        chunk_type: Type of chunk ("text", "metadata", "done")
        
    Returns:
        Formatted SSE chunk
    """
    data = {
        "type": chunk_type,
        "content": chunk
    }
    return f"data: {json.dumps(data)}\n\n"


async def stream_narrative_generation(
    orchestration_response,
    query: str
) -> AsyncGenerator[str, None]:
    """
    Stream narrative generation with metadata.
    
    Args:
        orchestration_response: Response from orchestration service
        query: Original search query
        
    Yields:
        SSE-formatted chunks
    """
    # Send initial metadata
    yield format_streaming_chunk(
        json.dumps({
            "query": query,
            "status": "generating",
            "type": "narrative"
        }),
        "metadata"
    )
    
    # Stream the narrative text
    buffer = ""
    async for chunk in orchestration_response.aiter_text():
        buffer += chunk
        # Yield complete sentences or paragraphs
        if "\n" in buffer or len(buffer) > 100:
            parts = buffer.split("\n", 1)
            if len(parts) == 2:
                yield format_streaming_chunk(parts[0] + "\n", "text")
                buffer = parts[1]
            else:
                yield format_streaming_chunk(buffer, "text")
                buffer = ""
    
    # Yield remaining buffer
    if buffer:
        yield format_streaming_chunk(buffer, "text")
    
    # Send completion signal
    yield format_streaming_chunk("", "done")


async def stream_dashboard_generation(
    orchestration_response,
    query: str
) -> AsyncGenerator[str, None]:
    """
    Stream dashboard generation with metadata.
    
    Args:
        orchestration_response: Response from orchestration service
        query: Original search query
        
    Yields:
        SSE-formatted chunks
    """
    # Send initial metadata
    yield format_streaming_chunk(
        json.dumps({
            "query": query,
            "status": "generating",
            "type": "dashboard"
        }),
        "metadata"
    )
    
    # Stream the dashboard JSON
    buffer = ""
    async for chunk in orchestration_response.aiter_text():
        buffer += chunk
        # Try to parse and yield complete JSON objects
        try:
            if buffer.strip().startswith("{") and buffer.strip().endswith("}"):
                parsed = json.loads(buffer.strip())
                yield format_streaming_chunk(json.dumps(parsed), "json")
                buffer = ""
        except:
            pass
    
    # Send completion signal
    yield format_streaming_chunk("", "done")

