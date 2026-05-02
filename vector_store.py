from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from config import (
    BASE_DIR,
    COLLECTION_NAME,
    CONFIDENCE_WEIGHTS,
    TOP_K_RETRIEVAL,
    VECTOR_DB_DIR,
    ensure_directories,
    get_allowed_authorities,
)
from embeddings import EmbeddingGenerator
from models import DocumentChunk, DocumentMetadata, RetrievedDocument, RetrievalResult


class VectorStore:
    def __init__(
        self,
        persist_directory: Path | str = VECTOR_DB_DIR,
        embedder: EmbeddingGenerator | None = None,
    ):
        ensure_directories()
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.chroma_directory = self.persist_directory / "chroma"
        self.embedder = embedder or EmbeddingGenerator()
        self.collection_name = f"{COLLECTION_NAME}_{self.embedder.backend}"
        try:
            import chromadb
            from chromadb.config import Settings
        except Exception as exc:
            raise RuntimeError("ChromaDB is required for persistent vector storage.") from exc

        try:
            self.chroma_directory.mkdir(parents=True, exist_ok=True)
            self.backend = "chromadb"
            self.client = chromadb.PersistentClient(
                path=str(self.chroma_directory),
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._migrate_legacy_json_if_needed()
        except Exception as exc:
            raise RuntimeError(
                f"Unable to initialize persistent Chroma collection at {self.chroma_directory}."
            ) from exc

    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [self._serialize_metadata(chunk.metadata) for chunk in chunks]

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def retrieve(self, query: str, user_role: str, top_k: int = TOP_K_RETRIEVAL) -> RetrievalResult:
        start_time = time.time()
        query_embedding = self.embedder.generate_embedding(query)
        allowed_authorities = set(get_allowed_authorities(user_role))

        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(top_k * 5, top_k),
            include=["documents", "metadatas", "distances"],
        )
        total_candidates = self.collection.count()
        candidate_rows = []
        for document, metadata, distance in zip(
            raw.get("documents", [[]])[0],
            raw.get("metadatas", [[]])[0],
            raw.get("distances", [[]])[0],
        ):
            meta = self._deserialize_metadata(metadata or {})
            base_similarity = max(0.0, 1.0 - float(distance))
            candidate_rows.append(
                {
                    "text": document,
                    "metadata": meta,
                    "similarity": self._score_similarity(
                        query,
                        document,
                        base_similarity,
                        meta,
                    ),
                }
            )
        candidate_rows.sort(key=lambda row: row["similarity"], reverse=True)
        candidate_rows = candidate_rows[: max(top_k * 5, top_k)]

        authorized_rows = [
            row
            for row in candidate_rows
            if row["metadata"].authority in allowed_authorities
        ]
        filtered_count = len(authorized_rows)
        top_rows = authorized_rows[:top_k]

        documents: List[RetrievedDocument] = []
        for rank, row in enumerate(top_rows, start=1):
            documents.append(
                RetrievedDocument(
                    chunk_id=row["metadata"].chunk_id,
                    text=row["text"],
                    similarity_score=row["similarity"],
                    metadata=row["metadata"],
                    rank=rank,
                    is_authorized=True,
                )
            )

        avg_similarity = (
            sum(doc.similarity_score for doc in documents) / len(documents)
            if documents
            else 0.0
        )
        top1_similarity = documents[0].similarity_score if documents else 0.0
        coverage_score = self._calculate_coverage_score(query, [doc.text for doc in documents])
        confidence_score = min(
            1.0,
            (
                CONFIDENCE_WEIGHTS["avg_similarity"] * avg_similarity
                + CONFIDENCE_WEIGHTS["top1_similarity"] * top1_similarity
                + CONFIDENCE_WEIGHTS["coverage_score"] * coverage_score
            ),
        )

        return RetrievalResult(
            query=query,
            user_role=user_role,
            documents=documents,
            total_candidates=total_candidates,
            filtered_count=filtered_count,
            retrieved_count=len(documents),
            avg_similarity=avg_similarity,
            top1_similarity=top1_similarity,
            coverage_score=coverage_score,
            confidence_score=confidence_score,
            retrieval_time_ms=(time.time() - start_time) * 1000,
        )

    def get_collection_stats(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "collection_name": self.collection_name,
            "total_chunks": self.collection.count(),
        }

    def clear_collection(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})

    def storage_target(self) -> Path:
        return self.chroma_directory

    def _migrate_legacy_json_if_needed(self) -> None:
        marker_file = self.persist_directory / f"{self.collection_name}.json.migrated"
        legacy_file = self._find_legacy_json_file()
        if legacy_file is None:
            return

        try:
            records = json.loads(legacy_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Legacy vector file is not valid JSON: {legacy_file}") from exc

        valid_records = [
            record
            for record in records
            if record.get("chunk_id")
            and record.get("text")
            and record.get("embedding")
            and record.get("metadata")
        ]
        current_count = self.collection.count()
        if current_count >= len(valid_records):
            if not marker_file.exists():
                marker_file.write_text(
                    json.dumps(
                        {
                            "legacy_file": str(legacy_file),
                            "migrated_records": len(valid_records),
                            "collection_name": self.collection_name,
                            "existing_records": current_count,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            return

        for index in range(0, len(valid_records), 500):
            batch = valid_records[index : index + 500]
            self.collection.upsert(
                ids=[str(record["chunk_id"]) for record in batch],
                embeddings=[record["embedding"] for record in batch],
                documents=[str(record["text"]) for record in batch],
                metadatas=[
                    self._serialize_metadata(DocumentMetadata.from_storage_dict(record["metadata"]))
                    for record in batch
                ],
            )
        marker_file.write_text(
            json.dumps(
                {
                    "legacy_file": str(legacy_file),
                    "migrated_records": len(valid_records),
                    "collection_name": self.collection_name,
                    "existing_records": current_count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _find_legacy_json_file(self) -> Path | None:
        candidates = [
            self.persist_directory / f"{self.collection_name}.json",
            BASE_DIR / "vector_db" / f"{self.collection_name}.json",
        ]
        fallback_name = f"{COLLECTION_NAME}_local.json"
        if self.collection_name != f"{COLLECTION_NAME}_local":
            candidates.extend(
                [
                    self.persist_directory / fallback_name,
                    BASE_DIR / "vector_db" / fallback_name,
                ]
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _serialize_metadata(self, metadata: DocumentMetadata) -> Dict[str, Any]:
        serialized = {}
        for key, value in metadata.storage_dict().items():
            if isinstance(value, list):
                serialized[key] = json.dumps(value)
            elif value is None:
                serialized[key] = ""
            else:
                serialized[key] = value
        return serialized

    def _deserialize_metadata(self, metadata: Dict[str, Any]) -> DocumentMetadata:
        payload: Dict[str, Any] = {}
        for key, value in metadata.items():
            if key in {"allowed_roles", "tags"} and isinstance(value, str):
                try:
                    payload[key] = json.loads(value)
                except json.JSONDecodeError:
                    payload[key] = [item for item in value.split(",") if item]
            else:
                payload[key] = value
        return DocumentMetadata.from_storage_dict(payload)

    def _calculate_coverage_score(self, query: str, documents: List[str]) -> float:
        query_terms = set(self._query_terms(query))
        if not query_terms or not documents:
            return 0.0

        covered = set()
        for document in documents:
            lowered = document.lower()
            for token in query_terms:
                if token in lowered:
                    covered.add(token)
        return len(covered) / len(query_terms)

    def _score_similarity(
        self,
        query: str,
        document: str,
        base_similarity: float,
        metadata: DocumentMetadata,
    ) -> float:
        lexical_similarity = self._lexical_similarity(query, document)
        clause_bonus = self._clause_bonus(query, metadata)
        return min(
            1.0,
            max(
                0.0,
                (0.20 * max(0.0, base_similarity))
                + (0.75 * lexical_similarity)
                + clause_bonus,
            ),
        )

    def _lexical_similarity(self, query: str, document: str) -> float:
        query_terms = self._query_terms(query)
        if not query_terms:
            return 0.0

        document_lower = document.lower()
        overlap = sum(1 for token in query_terms if token in document_lower)
        score = overlap / len(query_terms)

        phrase_bonus = 0.0
        for first, second in zip(query_terms, query_terms[1:]):
            if f"{first} {second}" in document_lower:
                phrase_bonus += 0.1
        return min(1.0, score + min(0.3, phrase_bonus))

    def _query_terms(self, query: str) -> List[str]:
        stopwords = {
            "what",
            "which",
            "when",
            "where",
            "who",
            "why",
            "how",
            "the",
            "and",
            "for",
            "our",
            "your",
            "are",
            "is",
            "does",
            "about",
            "document",
            "documents",
            "mention",
            "mentions",
            "mentioned",
        }
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        return [
            self._normalize_query_token(token)
            for token in tokens
            if len(token) > 1 and token not in stopwords
        ]

    def _normalize_query_token(self, token: str) -> str:
        if len(token) > 5 and token.endswith("ies"):
            return f"{token[:-3]}y"
        if len(token) > 4 and token.endswith("es") and not token.endswith("ses"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token

    def _clause_bonus(self, query: str, metadata: DocumentMetadata) -> float:
        clause_type = str(metadata.extra.get("clause_type", "") or "").lower()
        if not clause_type:
            return 0.0

        lowered_query = query.lower()
        if clause_type in lowered_query:
            return 0.12

        query_terms = set(self._query_terms(query))
        mapped_terms = {
            "confidentiality": {"confidentiality", "confidential", "non-disclosure", "nda"},
            "termination": {"termination", "term", "survival"},
            "liability": {"liability", "damages", "indemnity", "exclusion"},
            "obligations": {"obligation", "obligations", "duty", "duties"},
            "exceptions": {"exception", "exceptions", "public", "independent"},
            "breach": {"breach", "violation", "default"},
            "remedies": {"remedy", "remedies", "injunction", "injunctive"},
        }
        if query_terms & mapped_terms.get(clause_type, set()):
            return 0.08
        return 0.0
