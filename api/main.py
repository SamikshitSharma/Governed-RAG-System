from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from api import ingest
from config import RAW_DOCUMENTS_DIR, ensure_directories
from models import DecisionType, RAGResponse
from query import run_query
from vector_store import VectorStore


load_dotenv()
ensure_directories()

_vector_store_ready_task: asyncio.Task[None] | None = None


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _log_json(payload: dict[str, object]) -> None:
    print(_json_dumps(payload), file=sys.stdout, flush=True)


def _initialize_vector_store() -> None:
    vector_store = VectorStore()
    vector_stats = vector_store.get_collection_stats()
    _log_json(
        {
            "timestamp": _utc_timestamp(),
            "event": "vector_store_ready",
            "backend": vector_stats["backend"],
            "collection_name": vector_stats["collection_name"],
            "total_chunks": vector_stats["total_chunks"],
            "storage_target": str(vector_store.storage_target()),
        }
    )


async def _wait_for_vector_store_ready() -> None:
    if _vector_store_ready_task is not None:
        await asyncio.shield(_vector_store_ready_task)


def _chunk_text(text: str, chunk_size: int = 320) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return [""]

    paragraphs = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    if not paragraphs:
        paragraphs = [normalized]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        remainder = paragraph
        while len(remainder) > chunk_size:
            split_at = remainder.rfind(" ", 0, chunk_size)
            if split_at <= 0:
                split_at = chunk_size
            chunks.append(remainder[:split_at].strip())
            remainder = remainder[split_at:].strip()
        current = remainder

    if current:
        chunks.append(current)

    return chunks or [normalized]


def _format_source_label(source_trace: dict[str, object]) -> str:
    document_name = str(source_trace.get("document_name") or "Unknown document")
    parts = []
    clause_type = source_trace.get("clause_type")
    page_number = source_trace.get("page_number")
    if clause_type:
        parts.append(str(clause_type).replace("_", " "))
    if page_number:
        parts.append(f"p. {page_number}")
    return f"{document_name} ({', '.join(parts)})" if parts else document_name


def _refusal_payload(response: RAGResponse) -> dict[str, object]:
    return {
        "type": "refusal",
        "reason": response.governance_decision.refusal_message or response.response_text,
        "risk": response.governance_decision.risk_level.value,
        "trust_score": response.trust_score,
        "faithfulness_score": response.faithfulness_score,
        "refusal_reason": (
            response.governance_decision.refusal_reason.value
            if response.governance_decision.refusal_reason
            else None
        ),
        "retrieval_metadata": response.source_traces,
    }


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _vector_store_ready_task
    ensure_directories()
    _log_json(
        {
            "timestamp": _utc_timestamp(),
            "event": "startup",
            "service": "governed-rag-api",
        }
    )
    _vector_store_ready_task = asyncio.create_task(run_in_threadpool(_initialize_vector_store))
    try:
        yield
    finally:
        if _vector_store_ready_task is not None and not _vector_store_ready_task.done():
            _vector_store_ready_task.cancel()
    _log_json(
        {
            "timestamp": _utc_timestamp(),
            "event": "shutdown",
            "service": "governed-rag-api",
        }
    )


app = FastAPI(
    title="Governed Cloud Hosted RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(ingest.router)


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(..., min_length=1)
    user_role: str = Field(..., min_length=1)


class StatusResponse(BaseModel):
    status: str


class DocumentsResponse(BaseModel):
    documents: list[str]


class VectorStatsResponse(BaseModel):
    backend: str
    collection_name: str
    total_chunks: int
    storage_target: str


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        _log_json(
            {
                "timestamp": _utc_timestamp(),
                "event": "request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else None,
                "status_code": 500,
                "duration_ms": duration_ms,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        response = JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "request_id": request_id,
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    _log_json(
        {
            "timestamp": _utc_timestamp(),
            "event": "request",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else None,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        }
    )
    return response


async def _stream_query_events(request: Request, payload: QueryRequest) -> AsyncIterator[dict[str, str]]:
    request_id = request.state.request_id

    yield {
        "event": "message",
        "data": _json_dumps(
            {
                "type": "status",
                "status": "processing",
                "request_id": request_id,
            }
        ),
    }

    try:
        await _wait_for_vector_store_ready()
        response = await run_in_threadpool(
            run_query,
            payload.question,
            "api",
            payload.user_role,
        )
    except Exception as exc:
        _log_json(
            {
                "timestamp": _utc_timestamp(),
                "event": "query_failed",
                "request_id": request_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        yield {
            "event": "message",
            "data": _json_dumps(
                {
                    "type": "error",
                    "message": "Query execution failed.",
                    "request_id": request_id,
                }
            ),
        }
        return

    _log_json(
        {
            "timestamp": _utc_timestamp(),
            "event": "query_completed",
            "request_id": request_id,
            "query_id": response.query_id,
            "final_decision": response.final_decision.value,
            "risk": response.governance_decision.risk_level.value,
            "latency_ms": round(response.total_latency_ms, 2),
        }
    )

    if response.final_decision == DecisionType.REFUSE:
        yield {
            "event": "message",
            "data": _json_dumps(_refusal_payload(response)),
        }
        return

    for index, chunk in enumerate(_chunk_text(response.response_text)):
        if await request.is_disconnected():
            return
        yield {
            "event": "message",
            "data": _json_dumps(
                {
                    "type": "chunk",
                    "index": index,
                    "content": chunk,
                    "query_id": response.query_id,
                }
            ),
        }
        await asyncio.sleep(0)

    if await request.is_disconnected():
        return

    yield {
        "event": "message",
        "data": _json_dumps(
            {
                "type": "complete",
                "query_id": response.query_id,
                "decision": response.final_decision.value,
                "risk": response.governance_decision.risk_level.value,
                "sources": [_format_source_label(source) for source in response.source_traces],
                "trust_score": response.trust_score,
                "faithfulness_score": response.faithfulness_score,
                "refusal_reason": None,
                "retrieval_metadata": response.source_traces,
            }
        ),
    }


@app.post("/query")
async def query_endpoint(payload: QueryRequest, request: Request) -> EventSourceResponse:
    return EventSourceResponse(
        _stream_query_events(request, payload),
        ping=15,
        headers={
            "Cache-Control": "no-cache",
            "X-Request-ID": request.state.request_id,
        },
    )


@app.get("/health", response_model=StatusResponse)
async def health_endpoint() -> StatusResponse:
    return StatusResponse(status="ok")


@app.get("/documents", response_model=DocumentsResponse)
async def documents_endpoint() -> DocumentsResponse:
    ensure_directories()
    documents = sorted(path.name for path in RAW_DOCUMENTS_DIR.iterdir() if path.is_file())
    return DocumentsResponse(documents=documents)


@app.get("/vector-stats", response_model=VectorStatsResponse)
async def vector_stats_endpoint() -> VectorStatsResponse:
    await _wait_for_vector_store_ready()
    vector_store = VectorStore()
    stats = vector_store.get_collection_stats()
    return VectorStatsResponse(
        backend=str(stats["backend"]),
        collection_name=str(stats["collection_name"]),
        total_chunks=int(stats["total_chunks"]),
        storage_target=str(vector_store.storage_target()),
    )
