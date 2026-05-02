from __future__ import annotations

import time
from typing import List

from config import (
    CONFIDENCE_THRESHOLD,
    MIN_DOCUMENT_COUNT,
    MIN_SIMILARITY_THRESHOLD,
    RISK_THRESHOLDS,
    get_allowed_authorities,
    get_refusal_message,
)
from models import (
    DecisionType,
    GovernanceDecision,
    RefusalReason,
    RetrievalResult,
    RiskLevel,
)


class GovernanceLayer:
    def make_decision(self, retrieval: RetrievalResult, user_role: str) -> GovernanceDecision:
        start_time = time.time()

        allowed_authorities = set(get_allowed_authorities(user_role))
        meets_similarity = retrieval.top1_similarity >= MIN_SIMILARITY_THRESHOLD if retrieval.documents else False
        meets_confidence = retrieval.confidence_score >= CONFIDENCE_THRESHOLD
        has_sufficient_docs = retrieval.retrieved_count >= MIN_DOCUMENT_COUNT
        has_access = all(
            document.metadata.authority in allowed_authorities
            for document in retrieval.documents
        )

        risk_score = self._calculate_risk_score(retrieval)
        risk_level = self._risk_level(risk_score)

        refusal_reason = None
        refusal_message = None

        if retrieval.total_candidates > 0 and retrieval.filtered_count == 0:
            decision = DecisionType.REFUSE
            refusal_reason = RefusalReason.NO_ACCESS
        elif not has_sufficient_docs:
            decision = DecisionType.REFUSE
            refusal_reason = RefusalReason.INSUFFICIENT_CONTEXT
        elif not meets_similarity or not meets_confidence:
            decision = DecisionType.REFUSE
            refusal_reason = RefusalReason.LOW_CONFIDENCE
        elif not has_access:
            decision = DecisionType.REFUSE
            refusal_reason = RefusalReason.NO_ACCESS
        elif risk_level == RiskLevel.CRITICAL:
            decision = DecisionType.REFUSE
            refusal_reason = RefusalReason.HIGH_RISK
        elif risk_level == RiskLevel.HIGH:
            decision = DecisionType.ANSWER_WITH_WARNING
        else:
            decision = DecisionType.ANSWER

        if refusal_reason is not None:
            refusal_message = get_refusal_message(refusal_reason.value)

        return GovernanceDecision(
            decision=decision,
            confidence_score=retrieval.confidence_score,
            risk_level=risk_level,
            risk_score=risk_score,
            meets_similarity_threshold=meets_similarity,
            meets_confidence_threshold=meets_confidence,
            has_sufficient_documents=has_sufficient_docs,
            has_access_permission=has_access,
            refusal_reason=refusal_reason,
            refusal_message=refusal_message,
            suggested_actions=self._suggest_actions(refusal_reason, retrieval),
            decision_time_ms=(time.time() - start_time) * 1000,
        )

    def _calculate_risk_score(self, retrieval: RetrievalResult) -> float:
        confidence_risk = 1.0 - retrieval.confidence_score
        similarity_risk = 1.0 - retrieval.top1_similarity
        coverage_risk = 1.0 - retrieval.coverage_score
        quantity_risk = 1.0 if retrieval.retrieved_count == 0 else max(0.0, 1.0 - (retrieval.retrieved_count / 5.0))
        score = (
            0.35 * confidence_risk
            + 0.30 * similarity_risk
            + 0.20 * coverage_risk
            + 0.15 * quantity_risk
        )
        return max(0.0, min(1.0, score))

    def _risk_level(self, risk_score: float) -> RiskLevel:
        if risk_score >= RISK_THRESHOLDS["critical"]:
            return RiskLevel.CRITICAL
        if risk_score >= RISK_THRESHOLDS["high"]:
            return RiskLevel.HIGH
        if risk_score >= RISK_THRESHOLDS["medium"]:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _suggest_actions(
        self,
        refusal_reason: RefusalReason | None,
        retrieval: RetrievalResult,
    ) -> List[str]:
        if refusal_reason == RefusalReason.NO_ACCESS:
            return [
                "Try the query with a role that has access to the target documents.",
                "Request elevated access if this information is required for your work.",
            ]
        if refusal_reason == RefusalReason.INSUFFICIENT_CONTEXT:
            return [
                "Try a more specific question.",
                "Add more source documents to the knowledge base and ingest again.",
            ]
        if refusal_reason == RefusalReason.LOW_CONFIDENCE:
            suggestions = [
                "Use more precise terms from the source documents.",
                "Ask about a narrower topic.",
            ]
            if retrieval.filtered_count == 0:
                suggestions.append("Re-run ingestion if the document set changed.")
            return suggestions
        if refusal_reason == RefusalReason.HIGH_RISK:
            return ["Review the retrieved sources manually before answering."]
        return []

