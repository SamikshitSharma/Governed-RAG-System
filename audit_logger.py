from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from config import LOGS_DIR, ensure_directories
from models import DecisionType, RAGResponse


class AuditLogger:
    def __init__(self, log_dir: Path | str = LOGS_DIR) -> None:
        ensure_directories()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        date_part = datetime.utcnow().strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"audit_{date_part}.jsonl"
        self.summary_file = self.log_dir / "summary.jsonl"

    def log_rag_response(self, response: RAGResponse) -> None:
        detailed_log = {
            "timestamp": response.timestamp.isoformat(),
            "query_id": response.query_id,
            "query": response.query,
            "user_id": response.user_id,
            "user_role": response.user_role,
            "retrieval": {
                "retrieved_count": response.retrieval_result.retrieved_count,
                "filtered_count": response.retrieval_result.filtered_count,
                "confidence_score": response.retrieval_result.confidence_score,
                "avg_similarity": response.retrieval_result.avg_similarity,
                "top1_similarity": response.retrieval_result.top1_similarity,
                "coverage_score": response.retrieval_result.coverage_score,
                "retrieval_time_ms": response.retrieval_result.retrieval_time_ms,
                "chunk_ids": [document.chunk_id for document in response.retrieval_result.documents],
                "authorities": [document.metadata.authority for document in response.retrieval_result.documents],
            },
            "governance": {
                "decision": response.governance_decision.decision.value,
                "confidence_score": response.governance_decision.confidence_score,
                "risk_level": response.governance_decision.risk_level.value,
                "risk_score": response.governance_decision.risk_score,
                "refusal_reason": (
                    response.governance_decision.refusal_reason.value
                    if response.governance_decision.refusal_reason
                    else None
                ),
                "decision_time_ms": response.governance_decision.decision_time_ms,
            },
            "scores": {
                "trust_score": response.trust_score,
                "faithfulness_score": response.faithfulness_score,
            },
            "source_traces": response.source_traces,
            "generation": None,
            "evaluation": None,
            "final_decision": response.final_decision.value,
            "response_text": response.response_text,
            "total_latency_ms": response.total_latency_ms,
            "cost_usd": response.cost_usd,
        }

        if response.generated_answer is not None:
            detailed_log["generation"] = {
                "model_name": response.generated_answer.model_name,
                "prompt_tokens": response.generated_answer.prompt_tokens,
                "completion_tokens": response.generated_answer.completion_tokens,
                "total_tokens": response.generated_answer.total_tokens,
                "cited_chunks": response.generated_answer.cited_chunks,
                "generation_time_ms": response.generated_answer.generation_time_ms,
            }

        if response.evaluation_result is not None:
            detailed_log["evaluation"] = {
                "is_faithful": response.evaluation_result.faithfulness.is_faithful,
                "confidence": response.evaluation_result.faithfulness.confidence,
                "risk_level": response.evaluation_result.hallucination_risk.risk_level.value,
                "risk_score": response.evaluation_result.hallucination_risk.risk_score,
                "passes_evaluation": response.evaluation_result.passes_evaluation,
            }

        self._append_jsonl(self.log_file, detailed_log)

        summary_log = {
            "timestamp": response.timestamp.isoformat(),
            "query_id": response.query_id,
            "user_role": response.user_role,
            "decision": response.final_decision.value,
            "confidence": response.governance_decision.confidence_score,
            "trust_score": response.trust_score,
            "faithfulness_score": response.faithfulness_score,
            "risk_score": response.governance_decision.risk_score,
            "latency_ms": response.total_latency_ms,
            "cost_usd": response.cost_usd,
            "refused": response.final_decision == DecisionType.REFUSE,
            "refusal_reason": (
                response.governance_decision.refusal_reason.value
                if response.governance_decision.refusal_reason
                else None
            ),
        }
        self._append_jsonl(self.summary_file, summary_log)

    def get_statistics(self) -> Dict[str, object]:
        if not self.summary_file.exists():
            return {"total_queries": 0, "refusal_rate": 0.0}

        entries: List[Dict[str, object]] = []
        with self.summary_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))

        total_queries = len(entries)
        refused_queries = sum(1 for entry in entries if entry.get("refused"))
        answered_queries = total_queries - refused_queries
        avg_confidence = (
            sum(float(entry.get("confidence", 0.0)) for entry in entries) / total_queries
            if total_queries
            else 0.0
        )
        avg_risk_score = (
            sum(float(entry.get("risk_score", 0.0)) for entry in entries) / total_queries
            if total_queries
            else 0.0
        )
        avg_latency_ms = (
            sum(float(entry.get("latency_ms", 0.0)) for entry in entries) / total_queries
            if total_queries
            else 0.0
        )

        return {
            "total_queries": total_queries,
            "answered_queries": answered_queries,
            "refused_queries": refused_queries,
            "refusal_rate": (refused_queries / total_queries) if total_queries else 0.0,
            "avg_confidence": avg_confidence,
            "avg_risk_score": avg_risk_score,
            "avg_latency_ms": avg_latency_ms,
        }

    def _append_jsonl(self, path: Path, payload: Dict[str, object]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
