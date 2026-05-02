from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from config import (
    RAW_DOCUMENTS_DIR,
    VECTOR_DB_DIR,
    ensure_directories,
    get_roles_for_authority,
    normalize_authority_label,
    normalize_authority_type,
)
from embeddings import EmbeddingGenerator
from models import DocumentChunk, DocumentMetadata
from vector_store import VectorStore

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".docx"}
NDA_SUFFIXES = {".pdf"}
CLAUSE_PATTERNS = {
    "confidentiality": [
        r"\bconfidential(?:ity)?\b",
        r"\bnon[- ]disclosure\b",
        r"\bevaluation material\b",
    ],
    "termination": [
        r"\btermination\b",
        r"\bterm of (?:this )?agreement\b",
        r"\bremain in force\b",
        r"\bsurviv(?:e|al)\b",
    ],
    "liability": [
        r"\bliabilit(?:y|ies)\b",
        r"\bdamages\b",
        r"\bindemnif(?:y|ication)\b",
        r"\bconsequential\b",
    ],
    "obligations": [
        r"\bobligation(?:s)?\b",
        r"\bshall not disclose\b",
        r"\bshall keep\b",
        r"\buse .* solely\b",
        r"\brecipient shall\b",
        r"\buse the evaluation material solely\b",
    ],
    "exceptions": [
        r"\bdoes not include\b",
        r"\bshall not include\b",
        r"\bpublic domain\b",
        r"\bindependently developed\b",
        r"\balready in .* possession\b",
    ],
    "breach": [
        r"\bbreach\b",
        r"\bviolat(?:e|ion)\b",
        r"\bdefault\b",
        r"\bfail(?:ure)?\b",
    ],
    "remedies": [
        r"\bremed(?:y|ies)\b",
        r"\binjunctive relief\b",
        r"\bspecific performance\b",
        r"\bequitable relief\b",
    ],
}
HEADING_PATTERN = re.compile(
    r"^\s*(?:section\s+\d+|[ivxlcdm]+\.\s+|\d+[.)]\s+|[A-Z][A-Z\s/&,-]{5,})",
    re.IGNORECASE,
)


@dataclass
class ExtractedPage:
    page_number: int
    text: str


@dataclass
class SectionChunk:
    text: str
    page_number: int
    clause_type: str | None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 900,
        sentence_overlap: int = 1,
        profile: str = "generic",
        batch_id: str | None = None,
        default_document_type: str | None = None,
        default_sensitivity_level: str | None = None,
    ):
        self.chunk_size = chunk_size
        self.sentence_overlap = sentence_overlap
        self.profile = profile
        self.batch_id = batch_id or str(uuid.uuid4())
        self.default_document_type = default_document_type
        self.default_sensitivity_level = default_sensitivity_level

    def process_document(
        self,
        file_path: Path | str,
        *,
        source_url: str = "",
        document_name: str | None = None,
        source_path_override: str | None = None,
    ) -> List[DocumentChunk]:
        file_path = Path(file_path)
        source_document_name = document_name or file_path.name
        source_path = source_path_override or str(file_path)
        pages = self.load_document_pages(file_path)
        if not pages:
            raise ValueError("No extractable text was found in the document.")

        content = "\n\n".join(page.text for page in pages)
        base_metadata = self.extract_metadata(
            content,
            pages,
            file_path,
            source_url=source_url,
            document_name=source_document_name,
        )
        sections = self.extract_sections(pages)

        document_chunks: List[DocumentChunk] = []
        for section in sections:
            for chunk_text in self.chunk_text(section.text):
                clause_type = section.clause_type or self.detect_clause_type(chunk_text)
                chunk_index = len(document_chunks)
                chunk_id = f"{base_metadata['doc_id']}_chunk_{chunk_index:04d}"
                metadata = DocumentMetadata(
                    doc_id=base_metadata["doc_id"],
                    chunk_id=chunk_id,
                    authority=base_metadata["authority"],
                    source_file=source_document_name,
                    source_type=file_path.suffix.lower().lstrip("."),
                    chunk_index=chunk_index,
                    char_count=len(chunk_text),
                    department=base_metadata.get("department"),
                    version=base_metadata.get("version", "1.0"),
                    title=base_metadata.get("title"),
                    author=base_metadata.get("author"),
                    authority_type=base_metadata.get("authority_type", "memo"),
                    classification=base_metadata.get("classification"),
                    allowed_roles=base_metadata.get("allowed_roles", []),
                    tags=[tag for tag in [base_metadata.get("document_type"), clause_type] if tag],
                    extra={
                        "owner": base_metadata.get("owner"),
                        "document_path": source_path,
                        "document_name": source_document_name,
                        "document_type": base_metadata.get("document_type"),
                        "source_path": source_path,
                        "source_url": source_url,
                        "sensitivity_level": base_metadata.get("sensitivity_level"),
                        "clause_type": clause_type,
                        "batch_id": self.batch_id,
                        "doc_version": base_metadata.get("doc_version", base_metadata.get("version", "1.0")),
                        "file_size_bytes": file_path.stat().st_size,
                        "page_number": section.page_number,
                    },
                )
                document_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        metadata=metadata,
                    )
                )

        if not document_chunks:
            raise ValueError("Document did not produce any valid chunks.")
        return document_chunks

    def load_document_pages(self, file_path: Path) -> List[ExtractedPage]:
        suffix = file_path.suffix.lower()
        if suffix in {".md", ".txt"}:
            text = _normalize_text(file_path.read_text(encoding="utf-8", errors="ignore"))
            return [ExtractedPage(page_number=1, text=text)] if text else []
        if suffix == ".docx":
            try:
                from docx import Document
            except Exception as exc:
                raise RuntimeError("python-docx is required for DOCX ingestion.") from exc
            document = Document(str(file_path))
            text = _normalize_text("\n".join(paragraph.text for paragraph in document.paragraphs))
            return [ExtractedPage(page_number=1, text=text)] if text else []
        if suffix == ".pdf":
            pages = self._load_pdf_pages_with_unstructured(file_path)
            if not pages:
                pages = self._load_pdf_pages_with_pypdf2(file_path)
            pages = self._strip_repeated_headers_and_footers(pages)
            return [page for page in pages if page.text]
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def extract_metadata(
        self,
        content: str,
        pages: Sequence[ExtractedPage],
        file_path: Path,
        *,
        source_url: str,
        document_name: str | None = None,
    ) -> Dict[str, object]:
        source_name_path = Path(document_name or file_path.name)
        document_type = self.default_document_type or self._detect_document_type(content, source_name_path)
        sensitivity_level = self.default_sensitivity_level or ("high" if document_type == "NDA" else "medium")
        title = self._extract_title(content, source_name_path)
        version = self._extract_version(content, file_path)
        doc_id = self._default_doc_id(source_name_path)

        if document_type == "NDA":
            authority = "legal"
            authority_type = "memo"
            owner = "Legal"
            classification = "HIGH"
        else:
            markdown_fields = self._extract_markdown_fields(content)
            plain_header = self._extract_plain_header(content)
            if markdown_fields:
                classification = markdown_fields.get("classification")
                authority = normalize_authority_label(
                    classification=classification,
                    title=title,
                )
                authority_type = normalize_authority_type(markdown_fields.get("authority"))
                owner = markdown_fields.get("owner")
                doc_id = markdown_fields.get("document id") or doc_id
                version = markdown_fields.get("version", version)
            else:
                classification = plain_header.get("classification")
                authority = normalize_authority_label(
                    value=plain_header.get("authority"),
                    classification=classification,
                    title=title,
                )
                authority_type = "memo"
                owner = plain_header.get("owner")
                doc_id = plain_header.get("doc_id") or doc_id
                version = plain_header.get("version", version)

        return {
            "doc_id": doc_id,
            "title": title,
            "authority": authority,
            "authority_type": authority_type,
            "classification": classification,
            "version": version,
            "doc_version": version,
            "owner": owner,
            "author": owner,
            "department": owner or ("Legal" if document_type == "NDA" else None),
            "allowed_roles": get_roles_for_authority(authority),
            "document_type": document_type,
            "sensitivity_level": sensitivity_level,
            "source_url": source_url,
            "page_count": len(pages),
        }

    def extract_sections(self, pages: Sequence[ExtractedPage]) -> List[SectionChunk]:
        sections: List[SectionChunk] = []
        current_parts: List[str] = []
        current_page_number: int | None = None
        current_clause_type: str | None = None

        for page in pages:
            for paragraph in self._split_into_paragraphs(page.text):
                clause_type = self.detect_clause_type(paragraph)
                starts_new_section = self._starts_new_section(
                    paragraph,
                    current_parts,
                    clause_type,
                    current_clause_type,
                )

                if starts_new_section and current_parts:
                    sections.append(
                        SectionChunk(
                            text="\n\n".join(current_parts).strip(),
                            page_number=current_page_number or page.page_number,
                            clause_type=current_clause_type or self.detect_clause_type(" ".join(current_parts)),
                        )
                    )
                    current_parts = []
                    current_clause_type = None
                    current_page_number = None

                if current_page_number is None:
                    current_page_number = page.page_number
                current_parts.append(paragraph)
                if clause_type and not current_clause_type:
                    current_clause_type = clause_type

        if current_parts:
            sections.append(
                SectionChunk(
                    text="\n\n".join(current_parts).strip(),
                    page_number=current_page_number or 1,
                    clause_type=current_clause_type or self.detect_clause_type(" ".join(current_parts)),
                )
            )

        return [section for section in sections if len(section.text) >= 60]

    def chunk_text(self, text: str) -> List[str]:
        normalized = _normalize_text(text)
        if not normalized:
            return []

        if len(normalized) <= self.chunk_size:
            return [normalized]

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?;])\s+|\n+", normalized)
            if sentence.strip()
        ]
        if not sentences:
            return [normalized]

        chunks: List[str] = []
        start = 0
        while start < len(sentences):
            current_sentences: List[str] = []
            current_length = 0
            index = start

            while index < len(sentences):
                sentence = sentences[index]
                proposed_length = current_length + len(sentence) + (1 if current_sentences else 0)
                if current_sentences and proposed_length > self.chunk_size:
                    break
                current_sentences.append(sentence)
                current_length = proposed_length
                index += 1

            chunk = _normalize_text(" ".join(current_sentences))
            if len(chunk) >= 50:
                chunks.append(chunk)

            if index >= len(sentences):
                break
            start = max(index - self.sentence_overlap, start + 1)

        return chunks or [normalized]

    def detect_clause_type(self, text: str) -> str | None:
        lowered = text.lower()
        best_match: tuple[int, str] | None = None
        for clause_type, patterns in CLAUSE_PATTERNS.items():
            score = sum(1 for pattern in patterns if re.search(pattern, lowered))
            if score == 0:
                continue
            if best_match is None or score > best_match[0]:
                best_match = (score, clause_type)
        return best_match[1] if best_match else None

    def _load_pdf_pages_with_unstructured(self, file_path: Path) -> List[ExtractedPage]:
        try:
            from unstructured.partition.pdf import partition_pdf
        except Exception:
            return []

        try:
            elements = partition_pdf(filename=str(file_path))
        except Exception:
            return []

        pages: Dict[int, List[str]] = {}
        for element in elements:
            category = str(getattr(element, "category", "") or "").strip().lower()
            if category in {"header", "footer"}:
                continue

            raw_text = getattr(element, "text", None) or str(element)
            text = _normalize_text(raw_text)
            if not text:
                continue

            metadata = getattr(element, "metadata", None)
            page_number = getattr(metadata, "page_number", None) if metadata is not None else None
            if not isinstance(page_number, int) or page_number < 1:
                page_number = 1
            pages.setdefault(page_number, []).append(text)

        return [
            ExtractedPage(page_number=page_number, text=_normalize_text("\n\n".join(parts)))
            for page_number, parts in sorted(pages.items())
            if parts
        ]

    def _load_pdf_pages_with_pypdf2(self, file_path: Path) -> List[ExtractedPage]:
        try:
            import PyPDF2
        except Exception as exc:
            raise RuntimeError("PyPDF2 is required for PDF ingestion.") from exc

        pages: List[ExtractedPage] = []
        with file_path.open("rb") as handle:
            reader = PyPDF2.PdfReader(handle)
            for page_number, page in enumerate(reader.pages, start=1):
                text = _normalize_text(page.extract_text() or "")
                if text:
                    pages.append(ExtractedPage(page_number=page_number, text=text))
        return pages

    def _strip_repeated_headers_and_footers(self, pages: Sequence[ExtractedPage]) -> List[ExtractedPage]:
        if len(pages) <= 1:
            return list(pages)

        counts: Counter[str] = Counter()
        for page in pages:
            unique_lines = {
                self._normalize_line(line)
                for line in page.text.splitlines()
                if self._normalize_line(line)
            }
            counts.update(
                line
                for line in unique_lines
                if len(line) <= 120
            )

        removable_lines = {
            line
            for line, count in counts.items()
            if count >= max(2, len(pages) // 4)
            and (
                re.fullmatch(r"(?:page\s+)?\d+(?:\s+of\s+\d+)?", line, re.IGNORECASE)
                or line.isupper()
                or len(line.split()) <= 12
                or any(token in line.lower() for token in ["exhibit", ".htm", "page", "confidentiality agreement"])
            )
        }

        cleaned_pages: List[ExtractedPage] = []
        for page in pages:
            kept_lines = []
            for raw_line in page.text.splitlines():
                normalized_line = self._normalize_line(raw_line)
                if not normalized_line:
                    continue
                if normalized_line in removable_lines:
                    continue
                if re.fullmatch(r"(?:page\s+)?\d+(?:\s+of\s+\d+)?", normalized_line, re.IGNORECASE):
                    continue
                kept_lines.append(raw_line.strip())

            cleaned_pages.append(
                ExtractedPage(
                    page_number=page.page_number,
                    text=_normalize_text("\n".join(kept_lines)),
                )
            )
        return cleaned_pages

    def _split_into_paragraphs(self, text: str) -> List[str]:
        normalized = text
        normalized = re.sub(r"(?m)(?=^\s*\d+[.)]\s+)", "\n", normalized)
        normalized = re.sub(r"(?m)(?=^\s*section\s+\d+)", "\n", normalized, flags=re.IGNORECASE)
        parts = re.split(r"\n\s*\n+", normalized)
        paragraphs = [_normalize_text(part) for part in parts if _normalize_text(part)]
        return paragraphs

    def _starts_new_section(
        self,
        paragraph: str,
        current_parts: Sequence[str],
        next_clause_type: str | None,
        current_clause_type: str | None,
    ) -> bool:
        if not current_parts:
            return True
        if HEADING_PATTERN.match(paragraph):
            return True
        current_length = len(" ".join(current_parts))
        if next_clause_type and current_clause_type and next_clause_type != current_clause_type and current_length >= 240:
            return True
        if current_length >= self.chunk_size * 1.5:
            return True
        return False

    def _extract_title(self, content: str, file_path: Path) -> str:
        for line in content.splitlines():
            stripped = line.strip().strip("#").strip()
            if len(stripped) < 4:
                continue
            if re.fullmatch(r"(?:page\s+)?\d+(?:\s+of\s+\d+)?", stripped, re.IGNORECASE):
                continue
            return stripped[:180]
        return file_path.stem.replace("_", " ").title()

    def _extract_markdown_fields(self, content: str) -> Dict[str, str]:
        fields: Dict[str, str] = {}
        for line in content.splitlines()[:20]:
            match = re.search(r"\*\*(.+?)\*\*:\s*(.+)", line)
            if match:
                fields[match.group(1).strip().lower()] = match.group(2).strip()
        return fields

    def _extract_plain_header(self, content: str) -> Dict[str, str]:
        header = "\n".join(content.splitlines()[:12])
        authority_match = re.search(r"Authority:\s*([A-Za-z ]+)", header, re.IGNORECASE)
        version_match = re.search(r"Version\s+([0-9]+(?:\.[0-9]+)*)", header, re.IGNORECASE)
        classification = None
        if re.search(r"strictly confidential|highly confidential", header, re.IGNORECASE):
            classification = "CONFIDENTIAL"
        elif re.search(r"internal", header, re.IGNORECASE):
            classification = "INTERNAL"
        elif re.search(r"public", header, re.IGNORECASE):
            classification = "PUBLIC"
        return {
            "authority": authority_match.group(1).strip() if authority_match else "",
            "version": version_match.group(1).strip() if version_match else "1.0",
            "classification": classification or "",
        }

    def _extract_version(self, content: str, file_path: Path) -> str:
        match = re.search(r"\bversion\s+([0-9]+(?:\.[0-9]+)*)", content, re.IGNORECASE)
        if match:
            return match.group(1)
        return datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc).strftime("%Y%m%d%H%M%S")

    def _default_doc_id(self, file_path: Path) -> str:
        return re.sub(r"[^a-z0-9]+", "_", file_path.stem.lower()).strip("_")

    def _detect_document_type(self, content: str, file_path: Path) -> str:
        if self.profile == "nda":
            return "NDA"

        lowered = content.lower()
        if any(
            token in lowered
            for token in [
                "confidentiality agreement",
                "confidentiality and standstill agreement",
                "non-disclosure",
                "evaluation material",
                "transaction",
            ]
        ):
            return "NDA"
        return "KnowledgeBase"

    def _normalize_line(self, value: str) -> str:
        normalized = _normalize_text(value)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized


def _normalize_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"-\s*\n", "", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _persist_chunks(
    chunks: Sequence[DocumentChunk],
    *,
    clear_existing: bool,
) -> Dict[str, object]:
    embedder = EmbeddingGenerator()
    vector_store = VectorStore(embedder=embedder)

    if clear_existing:
        vector_store.clear_collection()

    embeddings = embedder.generate_embeddings_batch([chunk.text for chunk in chunks]) if chunks else []
    vector_store.add_documents(list(chunks), embeddings)
    stats = vector_store.get_collection_stats()

    return {
        "backend": stats["backend"],
        "collection_name": stats["collection_name"],
        "total_chunks": stats["total_chunks"],
        "vector_store_path": str(vector_store.storage_target()),
    }


def ingest_document_file(
    file_path: Path | str,
    *,
    clear_existing: bool = False,
    profile: str = "generic",
    document_type: str | None = None,
    sensitivity_level: str | None = None,
    batch_id: str | None = None,
    source_url: str = "",
    document_name: str | None = None,
    source_path_override: str | None = None,
) -> Dict[str, object]:
    ensure_directories()
    file_path = Path(file_path)
    batch_id = batch_id or str(uuid.uuid4())
    processor = DocumentProcessor(
        profile=profile,
        batch_id=batch_id,
        default_document_type=document_type,
        default_sensitivity_level=sensitivity_level,
    )

    chunks = processor.process_document(
        file_path,
        source_url=source_url,
        document_name=document_name,
        source_path_override=source_path_override,
    )
    storage_stats = _persist_chunks(chunks, clear_existing=clear_existing)
    doc_id = chunks[0].metadata.doc_id if chunks else processor._default_doc_id(file_path)
    source_document_name = document_name or file_path.name

    summary = {
        "timestamp": _utc_now_iso(),
        "batch_id": batch_id,
        "profile": profile,
        "processed_files": 1 if chunks else 0,
        "failed_files": 0,
        "skipped_files": 0,
        "chunk_count": len(chunks),
        "doc_id": doc_id,
        "document_name": source_document_name,
        "source_dir": str(Path(source_path_override).parent) if source_path_override else str(file_path.parent),
        **storage_stats,
    }
    _append_jsonl(VECTOR_DB_DIR / "ingest_log.jsonl", summary)
    return summary


def ingest_documents(
    source_dir: Path | str = RAW_DOCUMENTS_DIR,
    clear_existing: bool = True,
    *,
    profile: str = "generic",
    allowed_suffixes: Iterable[str] | None = None,
    document_type: str | None = None,
    sensitivity_level: str | None = None,
    batch_id: str | None = None,
) -> Dict[str, object]:
    ensure_directories()
    source_path = Path(source_dir)
    suffixes = {suffix.lower() for suffix in (allowed_suffixes or (NDA_SUFFIXES if profile == "nda" else SUPPORTED_SUFFIXES))}
    batch_id = batch_id or str(uuid.uuid4())

    processor = DocumentProcessor(
        profile=profile,
        batch_id=batch_id,
        default_document_type=document_type,
        default_sensitivity_level=sensitivity_level,
    )

    chunks: List[DocumentChunk] = []
    processed_files = 0
    skipped_files = 0
    failures: List[Dict[str, object]] = []

    for file_path in sorted(source_path.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in suffixes:
            continue

        try:
            document_chunks = processor.process_document(file_path)
            chunks.extend(document_chunks)
            processed_files += 1
        except Exception as exc:
            skipped_files += 1
            failure_payload = {
                "timestamp": _utc_now_iso(),
                "batch_id": batch_id,
                "file_name": file_path.name,
                "source_path": str(file_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure_payload)
            _append_jsonl(VECTOR_DB_DIR / "ingest_failures.jsonl", failure_payload)

    storage_stats = _persist_chunks(chunks, clear_existing=clear_existing)
    summary = {
        "timestamp": _utc_now_iso(),
        "batch_id": batch_id,
        "profile": profile,
        "source_dir": str(source_path),
        "processed_files": processed_files,
        "failed_files": len(failures),
        "skipped_files": skipped_files,
        "chunk_count": len(chunks),
        **storage_stats,
    }
    _append_jsonl(VECTOR_DB_DIR / "ingest_log.jsonl", summary)
    return summary


if __name__ == "__main__":
    print(ingest_documents())
