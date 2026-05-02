from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Tuple

from config import GENERATION_MODEL, GENERATION_PROMPT_TEMPLATE, GENERATION_TEMPERATURE
from models import GeneratedAnswer, RetrievedDocument


class AnswerGenerator:
    def __init__(self) -> None:
        self.model = GENERATION_MODEL
        self.temperature = GENERATION_TEMPERATURE
        self.client = None
        self.backend = "extractive"

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            try:
                from anthropic import Anthropic
            except Exception:
                self.client = None
            else:
                self.client = Anthropic(api_key=api_key)
                self.backend = "anthropic"

    def generate(
        self,
        query: str,
        documents: List[RetrievedDocument],
        max_tokens: int = 800,
    ) -> GeneratedAnswer:
        start_time = time.time()
        if self.backend == "anthropic" and self.client is not None:
            try:
                return self._generate_with_anthropic(query, documents, max_tokens, start_time)
            except Exception:
                pass
        return self._generate_extractive(query, documents, start_time)

    def calculate_cost(self, answer: GeneratedAnswer) -> float:
        if answer.model_name != self.model:
            return 0.0
        input_cost = (answer.prompt_tokens / 1_000_000) * 3.0
        output_cost = (answer.completion_tokens / 1_000_000) * 15.0
        return input_cost + output_cost

    def _generate_with_anthropic(
        self,
        query: str,
        documents: List[RetrievedDocument],
        max_tokens: int,
        start_time: float,
    ) -> GeneratedAnswer:
        context = self._format_context(documents)
        prompt = GENERATION_PROMPT_TEMPLATE.format(context=context, question=query)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        answer_text = response.content[0].text
        cited_chunks = self._extract_citations(answer_text, documents)
        if not cited_chunks:
            return self._generate_extractive(query, documents, start_time)
        prompt_tokens = int(getattr(response.usage, "input_tokens", 0))
        completion_tokens = int(getattr(response.usage, "output_tokens", 0))
        return GeneratedAnswer(
            answer_text=answer_text,
            model_name=self.model,
            temperature=self.temperature,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cited_chunks=cited_chunks,
            generation_time_ms=(time.time() - start_time) * 1000,
        )

    def _generate_extractive(
        self,
        query: str,
        documents: List[RetrievedDocument],
        start_time: float,
    ) -> GeneratedAnswer:
        if not documents:
            text = "I could not find any retrieved documents to answer from."
            return GeneratedAnswer(
                answer_text=text,
                model_name="extractive-fallback",
                temperature=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cited_chunks=[],
                generation_time_ms=(time.time() - start_time) * 1000,
            )

        ranked_passages = self._rank_passages(query, documents)
        chosen = ranked_passages[:2] if ranked_passages else []
        if not chosen:
            chosen = [(1.0, 0, documents[0].text.splitlines()[0].strip())]

        bullet_lines = []
        cited_chunks = []
        for _, index, line in chosen:
            document = documents[index]
            citation = f"[Document {index + 1}]"
            page_number = document.metadata.extra.get("page_number")
            clause_type = document.metadata.extra.get("clause_type")
            source_suffix = []
            if clause_type:
                source_suffix.append(str(clause_type).replace("_", " "))
            if page_number:
                source_suffix.append(f"p. {page_number}")
            source_hint = f" ({', '.join(source_suffix)})" if source_suffix else ""
            bullet_lines.append(f"- {line.strip()} {citation}{source_hint}")
            chunk_id = document.chunk_id
            if chunk_id not in cited_chunks:
                cited_chunks.append(chunk_id)

        answer_text = "Based on the retrieved evidence:\n" + "\n".join(bullet_lines)
        return GeneratedAnswer(
            answer_text=answer_text,
            model_name="extractive-fallback",
            temperature=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cited_chunks=cited_chunks,
            generation_time_ms=(time.time() - start_time) * 1000,
        )

    def _format_context(self, documents: List[RetrievedDocument]) -> str:
        parts = []
        for index, document in enumerate(documents, start=1):
            parts.append(f"[Document {index}]")
            parts.append(f"Source: {document.metadata.source_file}")
            parts.append(f"Authority: {document.metadata.authority}")
            parts.append(document.text)
            parts.append("")
        return "\n".join(parts)

    def _extract_citations(self, answer_text: str, documents: List[RetrievedDocument]) -> List[str]:
        cited = []
        for match in re.findall(r"\[Document (\d+)\]", answer_text):
            position = int(match) - 1
            if 0 <= position < len(documents):
                chunk_id = documents[position].chunk_id
                if chunk_id not in cited:
                    cited.append(chunk_id)
        return cited

    def _rank_passages(self, query: str, documents: List[RetrievedDocument]) -> List[Tuple[float, int, str]]:
        query_terms = {
            self._normalize_query_token(token)
            for token in re.findall(r"[a-z0-9]+", query.lower())
            if len(token) > 2
            and token
            not in {"what", "which", "does", "about", "say", "document", "documents", "mention", "mentions", "mentioned"}
        }
        ranked: List[Tuple[float, int, str]] = []

        for index, document in enumerate(documents):
            candidate = self._best_passage_excerpt(document.text, query_terms)
            if not candidate:
                continue

            candidate_terms = set(re.findall(r"[a-z0-9]+", candidate.lower()))
            overlap = len(query_terms & candidate_terms)
            clause_type = str(document.metadata.extra.get("clause_type", "") or "").lower()
            clause_bonus = 0.0
            if clause_type and clause_type in query.lower():
                clause_bonus += 0.4
            elif clause_type and any(term in query_terms for term in clause_type.split("_")):
                clause_bonus += 0.2

            title = str(document.metadata.title or "").lower()
            domain_bonus = 0.08 if any(token in title for token in ["confidential", "nondisclosure", "standstill"]) else 0.0
            score = document.similarity_score + (overlap / max(1, len(query_terms))) + clause_bonus + domain_bonus
            ranked.append((score, index, candidate))

        ranked.sort(key=lambda item: item[0], reverse=True)

        seen = set()
        deduped = []
        for item in ranked:
            key = item[2].lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _best_passage_excerpt(self, text: str, query_terms: set[str]) -> str:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
            if sentence.strip()
        ]
        if not sentences:
            return ""

        windows: List[Tuple[float, str]] = []
        for start in range(len(sentences)):
            for width in (1, 2, 3):
                window = sentences[start : start + width]
                if not window:
                    continue
                candidate = " ".join(window).strip()
                if len(candidate) < 60 or len(candidate) > 480:
                    continue
                if candidate.lower().startswith("source:"):
                    continue
                if candidate.endswith(("and", "or", "of", "to", "for", "the", "a", "an", ",")):
                    continue
                if candidate.count("(") != candidate.count(")"):
                    continue
                tokens = set(re.findall(r"[a-z0-9]+", candidate.lower()))
                overlap = len(tokens & query_terms)
                if overlap == 0:
                    continue
                fragment_penalty = 0.0
                if re.match(r"^[a-z]", candidate):
                    fragment_penalty += 0.2
                if candidate.endswith(("and", "or", "of", "to", "for", ",")):
                    fragment_penalty += 0.2
                score = overlap - fragment_penalty
                windows.append((score, candidate))

        if not windows:
            return ""

        windows.sort(key=lambda item: item[0], reverse=True)
        return windows[0][1]

    def _normalize_query_token(self, token: str) -> str:
        if len(token) > 5 and token.endswith("ies"):
            return f"{token[:-3]}y"
        if len(token) > 4 and token.endswith("es") and not token.endswith("ses"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token
