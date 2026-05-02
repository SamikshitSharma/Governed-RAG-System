from __future__ import annotations

import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from ingest import ingest_document_file
from vector_store import VectorStore

router = APIRouter(tags=["ingest"])
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt"}


class IngestUrlRequest(BaseModel):
    url: str = Field(..., min_length=1)


class IngestResponse(BaseModel):
    status: str
    doc_id: str
    chunks_created: int
    batch_id: str


class DeleteIngestResponse(BaseModel):
    status: str
    doc_id: str


def _guess_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Use PDF, DOCX, or TXT.",
        )
    return suffix


def _name_from_url(source_url: str) -> str:
    parsed_path = urlparse(source_url).path
    return Path(parsed_path).name or "downloaded-document.pdf"


def _download_url(source_url: str) -> tuple[bytes, str]:
    try:
        request = UrlRequest(source_url, headers={"User-Agent": "governed-rag-ingest/1.0"})
        with urlopen(request, timeout=60) as response:
            file_bytes = response.read()
            final_url = response.geturl()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to fetch URL: {source_url}",
        ) from exc

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The downloaded document is empty.",
        )

    return file_bytes, final_url


def _persist_temp_document(file_bytes: bytes, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(file_bytes)
        return Path(handle.name)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: Request) -> IngestResponse:
    temp_path: Path | None = None
    try:
        content_type = request.headers.get("content-type", "").lower()
        source_url = ""
        source_path = ""
        document_name = ""
        profile = "generic"

        if "application/json" in content_type:
            payload = IngestUrlRequest(**await request.json())
            file_bytes, source_url = _download_url(payload.url)
            document_name = _name_from_url(source_url)
            source_path = source_url
            suffix = _guess_suffix(document_name)
        else:
            form = await request.form()
            upload = form.get("file")
            if upload is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Provide a file upload or a JSON body with a url field.",
                )
            document_name = str(form.get("document_name") or getattr(upload, "filename", "") or "")
            source_path = str(form.get("source_path") or document_name)
            suffix = _guess_suffix(document_name)
            file_bytes = await upload.read()
            if not file_bytes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file is empty.",
                )

        if suffix == ".pdf":
            profile = "nda"

        temp_path = _persist_temp_document(file_bytes, suffix)
        summary = ingest_document_file(
            temp_path,
            clear_existing=False,
            profile=profile,
            document_type="NDA" if profile == "nda" else None,
            sensitivity_level="high" if profile == "nda" else None,
            source_url=source_url,
            document_name=document_name,
            source_path_override=source_path,
        )
        return IngestResponse(
            status="success",
            doc_id=str(summary["doc_id"]),
            chunks_created=int(summary["chunk_count"]),
            batch_id=str(summary["batch_id"]),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document ingestion failed.",
        ) from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


@router.delete("/ingest/{doc_id}", response_model=DeleteIngestResponse)
async def delete_document(doc_id: str) -> DeleteIngestResponse:
    VectorStore().delete_document(doc_id)
    return DeleteIngestResponse(status="deleted", doc_id=doc_id)
