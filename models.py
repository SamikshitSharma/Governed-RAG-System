from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class DataModel:
    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


class DecisionType(str, Enum):
    ANSWER = "answer"
    REFUSE = "refuse"
    ANSWER_WITH_WARNING = "answer_with_warning"


class RefusalReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    NO_ACCESS = "no_access"
    HIGH_RISK = "high_risk"
    FAITHFULNESS_FAILURE = "faithfulness_failure"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DocumentMetadata(DataModel):
    doc_id: str
    chunk_id: str
    authority: str
    source_file: str
    source_type: str
    chunk_index: int
    char_count: int
    department: Optional[str] = None
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    title: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    authority_type: str = "memo"
    classification: Optional[str] = None
    allowed_roles: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def storage_dict(self) -> Dict[str, Any]:
        payload = {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "authority": self.authority,
            "source_file": self.source_file,
            "source_type": self.source_type,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "department": self.department,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "title": self.title,
            "author": self.author,
            "tags": list(self.tags),
            "authority_type": self.authority_type,
            "classification": self.classification,
            "allowed_roles": list(self.allowed_roles),
        }
        payload.update(self.extra)
        return payload

    @classmethod
    def from_storage_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        known_keys = {
            "doc_id",
            "chunk_id",
            "authority",
            "source_file",
            "source_type",
            "chunk_index",
            "char_count",
            "department",
            "version",
            "created_at",
            "updated_at",
            "title",
            "author",
            "tags",
            "authority_type",
            "classification",
            "allowed_roles",
        }
        extras = {key: value for key, value in data.items() if key not in known_keys}
        created_at = _parse_datetime(data.get("created_at"))
        updated_at = _parse_datetime(data.get("updated_at"))
        return cls(
            doc_id=str(data.get("doc_id", "")),
            chunk_id=str(data.get("chunk_id", "")),
            authority=str(data.get("authority", "public")),
            source_file=str(data.get("source_file", "")),
            source_type=str(data.get("source_type", "txt")),
            chunk_index=int(data.get("chunk_index", 0)),
            char_count=int(data.get("char_count", 0)),
            department=data.get("department"),
            version=str(data.get("version", "1.0")),
            created_at=created_at,
            updated_at=updated_at,
            title=data.get("title"),
            author=data.get("author"),
            tags=list(data.get("tags") or []),
            authority_type=str(data.get("authority_type", "memo")),
            classification=data.get("classification"),
            allowed_roles=list(data.get("allowed_roles") or []),
            extra=extras,
        )


@dataclass
class DocumentChunk(DataModel):
    chunk_id: str
    text: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None


@dataclass
class RetrievedDocument(DataModel):
    chunk_id: str
    text: str
    similarity_score: float
    metadata: DocumentMetadata
    rank: Optional[int] = None
    is_authorized: bool = True


@dataclass
class RetrievalResult(DataModel):
    query: str
    user_role: str
    documents: List[RetrievedDocument]
    total_candidates: int
    filtered_count: int
    retrieved_count: int
    avg_similarity: float
    top1_similarity: float
    coverage_score: float
    confidence_score: float
    retrieval_time_ms: float


@dataclass
class GovernanceDecision(DataModel):
    decision: DecisionType
    confidence_score: float
    risk_level: RiskLevel
    risk_score: float
    meets_similarity_threshold: bool
    meets_confidence_threshold: bool
    has_sufficient_documents: bool
    has_access_permission: bool
    refusal_reason: Optional[RefusalReason] = None
    refusal_message: Optional[str] = None
    suggested_actions: List[str] = field(default_factory=list)
    decision_time_ms: float = 0.0


@dataclass
class GeneratedAnswer(DataModel):
    answer_text: str
    model_name: str
    temperature: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cited_chunks: List[str] = field(default_factory=list)
    generation_time_ms: float = 0.0


@dataclass
class FaithfulnessScore(DataModel):
    is_faithful: bool
    confidence: float
    supported_claims: int
    total_claims: int
    unsupported_claims: List[str] = field(default_factory=list)
    judge_model: str = "heuristic"
    judge_reasoning: Optional[str] = None
    evaluation_time_ms: float = 0.0


@dataclass
class HallucinationRisk(DataModel):
    risk_score: float
    risk_level: RiskLevel
    citation_density: float
    semantic_variance: float
    factual_consistency: float
    risk_factors: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult(DataModel):
    faithfulness: FaithfulnessScore
    hallucination_risk: HallucinationRisk
    passes_evaluation: bool
    evaluation_message: Optional[str] = None


@dataclass
class RAGResponse(DataModel):
    query: str
    user_id: str
    user_role: str
    retrieval_result: RetrievalResult
    governance_decision: GovernanceDecision
    final_decision: DecisionType
    response_text: str
    generated_answer: Optional[GeneratedAnswer] = None
    evaluation_result: Optional[EvaluationResult] = None
    total_latency_ms: float = 0.0
    cost_usd: float = 0.0
    trust_score: float = 0.0
    faithfulness_score: float = 0.0
    source_traces: List[Dict[str, Any]] = field(default_factory=list)
    pipeline_stages: Dict[str, Any] = field(default_factory=dict)
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AuditLog(DataModel):
    query_id: str
    user_id: str
    user_role: str
    query: str
    decision: DecisionType
    confidence_score: float
    risk_score: float
    retrieved_chunk_ids: List[str]
    document_authorities: List[str]
    response_provided: bool
    latency_ms: float
    cost_usd: float
    refusal_reason: Optional[RefusalReason] = None
    log_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data_accessed: List[str] = field(default_factory=list)
    access_violations: List[str] = field(default_factory=list)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.utcnow()
