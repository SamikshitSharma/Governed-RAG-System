from __future__ import annotations

import time

from answer_generator import AnswerGenerator
from audit_logger import AuditLogger
from config import TOP_K_RETRIEVAL, ensure_directories, get_refusal_message
from decision_gate import GovernanceLayer
from faithfulness_checker import FaithfulnessEvaluator
from models import DecisionType, RAGResponse, RefusalReason
from vector_store import VectorStore


def _build_source_traces(response_documents) -> list[dict[str, object]]:
    traces: list[dict[str, object]] = []
    for document in response_documents:
        traces.append(
            {
                "chunk_id": document.chunk_id,
                "document_name": document.metadata.extra.get("document_name") or document.metadata.source_file,
                "document_type": document.metadata.extra.get("document_type"),
                "source_path": document.metadata.extra.get("source_path") or document.metadata.extra.get("document_path"),
                "source_url": document.metadata.extra.get("source_url") or "",
                "page_number": document.metadata.extra.get("page_number"),
                "clause_type": document.metadata.extra.get("clause_type"),
                "similarity_score": round(document.similarity_score, 4),
                "authority": document.metadata.authority,
            }
        )
    return traces


def _calculate_trust_score(
    retrieval_confidence: float,
    faithfulness_score: float,
    citation_density: float,
) -> float:
    return round(
        max(
            0.0,
            min(
                1.0,
                (0.45 * retrieval_confidence)
                + (0.35 * faithfulness_score)
                + (0.20 * citation_density),
            ),
        ),
        4,
    )


def run_query(
    query: str,
    user_id: str,
    user_role: str,
    top_k: int | None = None,
) -> RAGResponse:
    ensure_directories()
    start_time = time.time()

    vector_store = VectorStore()
    governance_layer = GovernanceLayer()
    answer_generator = AnswerGenerator()
    faithfulness_evaluator = FaithfulnessEvaluator()
    audit_logger = AuditLogger()

    retrieval_result = vector_store.retrieve(query, user_role, top_k or TOP_K_RETRIEVAL)
    governance_decision = governance_layer.make_decision(retrieval_result, user_role)

    pipeline_stages = {
        "retrieval": {
            "retrieved_count": retrieval_result.retrieved_count,
            "confidence_score": retrieval_result.confidence_score,
        },
        "governance": {
            "decision": governance_decision.decision.value,
            "risk_level": governance_decision.risk_level.value,
        },
    }

    generated_answer = None
    evaluation_result = None
    final_decision = governance_decision.decision
    response_text = governance_decision.refusal_message or get_refusal_message("low_confidence")
    total_cost = 0.0
    faithfulness_score = 0.0
    trust_score = 0.0
    source_traces = _build_source_traces(retrieval_result.documents)

    if governance_decision.decision != DecisionType.REFUSE:
        generated_answer = answer_generator.generate(query, retrieval_result.documents)
        pipeline_stages["generation"] = {
            "model": generated_answer.model_name,
            "tokens": generated_answer.total_tokens,
            "cited_chunks": generated_answer.cited_chunks,
        }
        total_cost += answer_generator.calculate_cost(generated_answer)

        evaluation_result = faithfulness_evaluator.evaluate(generated_answer, retrieval_result.documents)
        faithfulness_score = evaluation_result.faithfulness.confidence
        trust_score = _calculate_trust_score(
            retrieval_result.confidence_score,
            faithfulness_score,
            evaluation_result.hallucination_risk.citation_density,
        )
        pipeline_stages["evaluation"] = {
            "is_faithful": evaluation_result.faithfulness.is_faithful,
            "confidence": evaluation_result.faithfulness.confidence,
            "passes": evaluation_result.passes_evaluation,
            "citation_density": evaluation_result.hallucination_risk.citation_density,
        }
        pipeline_stages["scores"] = {
            "trust_score": trust_score,
            "faithfulness_score": faithfulness_score,
        }

        if evaluation_result.passes_evaluation:
            response_text = generated_answer.answer_text
            if governance_decision.decision == DecisionType.ANSWER_WITH_WARNING:
                response_text = "Use caution: retrieval risk was elevated.\n\n" + response_text
        else:
            final_decision = DecisionType.REFUSE
            governance_decision.refusal_reason = RefusalReason.FAITHFULNESS_FAILURE
            governance_decision.refusal_message = get_refusal_message("faithfulness_failure")
            response_text = governance_decision.refusal_message
    else:
        trust_score = round(max(0.0, retrieval_result.confidence_score * 0.5), 4)

    pipeline_stages["sources"] = source_traces

    rag_response = RAGResponse(
        query=query,
        user_id=user_id,
        user_role=user_role,
        retrieval_result=retrieval_result,
        governance_decision=governance_decision,
        generated_answer=generated_answer,
        evaluation_result=evaluation_result,
        final_decision=final_decision,
        response_text=response_text,
        total_latency_ms=(time.time() - start_time) * 1000,
        cost_usd=total_cost,
        trust_score=trust_score,
        faithfulness_score=faithfulness_score,
        source_traces=source_traces,
        pipeline_stages=pipeline_stages,
    )
    audit_logger.log_rag_response(rag_response)
    return rag_response
