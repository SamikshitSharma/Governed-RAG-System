from __future__ import annotations

import json
import os
import re
import time
from typing import List

from config import (
    FAITHFULNESS_PROMPT_TEMPLATE,
    JUDGE_MODEL,
    JUDGE_TEMPERATURE,
    RISK_THRESHOLDS,
    THRESHOLDS_CONFIG,
    get_refusal_message,
)
from models import (
    EvaluationResult,
    FaithfulnessScore,
    GeneratedAnswer,
    HallucinationRisk,
    RetrievedDocument,
    RiskLevel,
)


class FaithfulnessEvaluator:
    def __init__(self) -> None:
        self.model = JUDGE_MODEL
        self.temperature = JUDGE_TEMPERATURE
        self.client = None
        self.backend = "heuristic"

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
            except Exception:
                self.client = None
            else:
                self.client = OpenAI(api_key=api_key)
                self.backend = "openai"

    def evaluate(
        self,
        generated_answer: GeneratedAnswer,
        documents: List[RetrievedDocument],
    ) -> EvaluationResult:
        start_time = time.time()
        if self.backend == "openai" and self.client is not None:
            try:
                result = self._evaluate_with_openai(generated_answer, documents, start_time)
            except Exception:
                result = self._evaluate_heuristically(generated_answer, documents, start_time)
        else:
            result = self._evaluate_heuristically(generated_answer, documents, start_time)

        min_score = THRESHOLDS_CONFIG["faithfulness"]["min_faithfulness_score"]
        passes = result.faithfulness.is_faithful and result.faithfulness.confidence >= min_score
        result.passes_evaluation = passes
        if not passes:
            result.evaluation_message = get_refusal_message("faithfulness_failure")
        return result

    def _evaluate_with_openai(
        self,
        generated_answer: GeneratedAnswer,
        documents: List[RetrievedDocument],
        start_time: float,
    ) -> EvaluationResult:
        context = self._build_context(documents)
        prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(
            context=context[:6000],
            answer=generated_answer.answer_text[:3000],
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": "You are a strict RAG faithfulness judge."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(self._extract_json(content))

        claims = self._split_claims(generated_answer.answer_text)
        unsupported_claims = list(parsed.get("unsupported_claims") or [])
        total_claims = max(1, len(claims))
        supported_claims = max(0, total_claims - len(unsupported_claims))
        confidence = float(parsed.get("confidence", 0.0))
        is_faithful = bool(parsed.get("is_faithful", False))

        faithfulness = FaithfulnessScore(
            is_faithful=is_faithful,
            confidence=max(0.0, min(1.0, confidence)),
            supported_claims=supported_claims,
            total_claims=total_claims,
            unsupported_claims=unsupported_claims,
            judge_model=self.model,
            judge_reasoning=parsed.get("reasoning"),
            evaluation_time_ms=(time.time() - start_time) * 1000,
        )
        risk = self._compute_risk(generated_answer, documents, faithfulness)
        return EvaluationResult(
            faithfulness=faithfulness,
            hallucination_risk=risk,
            passes_evaluation=False,
        )

    def _evaluate_heuristically(
        self,
        generated_answer: GeneratedAnswer,
        documents: List[RetrievedDocument],
        start_time: float,
    ) -> EvaluationResult:
        claims = self._split_claims(generated_answer.answer_text)
        total_claims = max(1, len(claims))
        context_tokens = set(re.findall(r"[a-z0-9]+", self._build_context(documents).lower()))

        unsupported_claims = []
        supported_claims = 0
        for claim in claims:
            claim_tokens = {
                token
                for token in re.findall(r"[a-z0-9]+", claim.lower())
                if len(token) > 2
            }
            if not claim_tokens:
                supported_claims += 1
                continue
            overlap = len(claim_tokens & context_tokens) / max(1, len(claim_tokens))
            if overlap >= 0.3 or "[Document" in claim:
                supported_claims += 1
            else:
                unsupported_claims.append(claim)

        confidence = supported_claims / total_claims
        faithfulness = FaithfulnessScore(
            is_faithful=confidence >= THRESHOLDS_CONFIG["faithfulness"]["min_faithfulness_score"],
            confidence=confidence,
            supported_claims=supported_claims,
            total_claims=total_claims,
            unsupported_claims=unsupported_claims,
            judge_model="heuristic",
            judge_reasoning="Lexical overlap heuristic fallback.",
            evaluation_time_ms=(time.time() - start_time) * 1000,
        )
        risk = self._compute_risk(generated_answer, documents, faithfulness)
        return EvaluationResult(
            faithfulness=faithfulness,
            hallucination_risk=risk,
            passes_evaluation=False,
        )

    def _compute_risk(
        self,
        generated_answer: GeneratedAnswer,
        documents: List[RetrievedDocument],
        faithfulness: FaithfulnessScore,
    ) -> HallucinationRisk:
        claims = self._split_claims(generated_answer.answer_text)
        total_claims = max(1, len(claims))
        citation_density = min(1.0, len(re.findall(r"\[Document \d+\]", generated_answer.answer_text)) / total_claims)

        answer_tokens = set(re.findall(r"[a-z0-9]+", generated_answer.answer_text.lower()))
        context_tokens = set(re.findall(r"[a-z0-9]+", self._build_context(documents).lower()))
        overlap = len(answer_tokens & context_tokens) / max(1, len(answer_tokens)) if answer_tokens else 0.0
        semantic_variance = max(0.0, 1.0 - overlap)
        factual_consistency = faithfulness.confidence

        risk_score = min(
            1.0,
            0.45 * (1.0 - faithfulness.confidence)
            + 0.25 * max(0.0, 1.0 - citation_density)
            + 0.30 * semantic_variance,
        )

        risk_factors = []
        if citation_density < THRESHOLDS_CONFIG["hallucination_risk"]["min_citation_density"]:
            risk_factors.append("Low citation density")
        if semantic_variance > THRESHOLDS_CONFIG["hallucination_risk"]["max_semantic_variance"]:
            risk_factors.append("High divergence from retrieved context")
        if not faithfulness.is_faithful:
            risk_factors.append("Faithfulness check failed")

        return HallucinationRisk(
            risk_score=risk_score,
            risk_level=self._risk_level(risk_score),
            citation_density=citation_density,
            semantic_variance=semantic_variance,
            factual_consistency=factual_consistency,
            risk_factors=risk_factors,
        )

    def _split_claims(self, answer_text: str) -> List[str]:
        stripped = re.sub(r"\[Document \d+\]", "", answer_text)
        parts = re.split(r"(?<=[.!?])\s+|\n+", stripped)
        return [part.strip(" -") for part in parts if len(part.strip(" -")) > 10]

    def _build_context(self, documents: List[RetrievedDocument]) -> str:
        return "\n\n".join(document.text for document in documents)

    def _risk_level(self, risk_score: float) -> RiskLevel:
        if risk_score >= RISK_THRESHOLDS["critical"]:
            return RiskLevel.CRITICAL
        if risk_score >= RISK_THRESHOLDS["high"]:
            return RiskLevel.HIGH
        if risk_score >= RISK_THRESHOLDS["medium"]:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _extract_json(self, content: str) -> str:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        return match.group(0) if match else "{}"

