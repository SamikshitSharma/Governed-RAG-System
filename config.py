from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List


BASE_DIR = Path(__file__).resolve().parent
RAW_DOCUMENTS_DIR = BASE_DIR / "data" / "raw_documents"
DEFAULT_CHROMA_PATH = "/app/chroma" if BASE_DIR == Path("/app") else "./vector_db"
CHROMA_PATH = os.getenv("CHROMA_PATH", DEFAULT_CHROMA_PATH)
VECTOR_DB_DIR = Path(CHROMA_PATH).expanduser()
if not VECTOR_DB_DIR.is_absolute():
    VECTOR_DB_DIR = BASE_DIR / VECTOR_DB_DIR
LOGS_DIR = BASE_DIR / "logs"
COLLECTION_NAME = "governed_docs"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


ROLES_CONFIG = {
    "roles": {
        "public": {
            "description": "Public read-only access.",
            "allowed_authorities": ["public"],
            "clearance_level": 0,
        },
        "contractor": {
            "description": "Contractors with public-only access.",
            "allowed_authorities": ["public"],
            "clearance_level": 0,
        },
        "employee": {
            "description": "Employees with internal handbook access.",
            "allowed_authorities": ["public", "employee"],
            "clearance_level": 1,
        },
        "analyst": {
            "description": "Analysts with access to governed NDA and enterprise records.",
            "allowed_authorities": ["public", "employee", "management", "legal", "strategic"],
            "clearance_level": 3,
        },
        "engineering": {
            "description": "Engineering staff with technical document access.",
            "allowed_authorities": ["public", "employee", "engineering"],
            "clearance_level": 2,
        },
        "manager": {
            "description": "Managers with management document access.",
            "allowed_authorities": ["public", "employee", "engineering", "management"],
            "clearance_level": 3,
        },
        "executive": {
            "description": "Executives with executive, financial, and strategic access.",
            "allowed_authorities": [
                "public",
                "employee",
                "engineering",
                "management",
                "executive",
                "financial",
                "legal",
                "strategic",
            ],
            "clearance_level": 4,
        },
        "admin": {
            "description": "Administrative override for all authorities.",
            "allowed_authorities": ["*"],
            "clearance_level": 999,
        },
    },
    "authorities": {
        "public": {"description": "Public information", "risk_level": "low"},
        "employee": {"description": "Employee-only information", "risk_level": "medium"},
        "engineering": {"description": "Engineering information", "risk_level": "medium"},
        "management": {"description": "Management information", "risk_level": "high"},
        "legal": {"description": "Legal and NDA information", "risk_level": "high"},
        "executive": {"description": "Executive information", "risk_level": "critical"},
        "financial": {"description": "Financial information", "risk_level": "critical"},
        "strategic": {"description": "Strategic information", "risk_level": "critical"},
    },
}

THRESHOLDS_CONFIG = {
    "retrieval_confidence": {
        "min_similarity_threshold": 0.55,
        "min_document_count": 1,
        "top_k": 5,
        "weights": {
            "avg_similarity": 0.4,
            "top1_similarity": 0.4,
            "coverage_score": 0.2,
        },
        "min_confidence": 0.58,
    },
    "faithfulness": {
        "judge_model": "gpt-4o-mini",
        "min_faithfulness_score": 0.75,
        "failure_behavior": "refuse_with_reason",
        "judge_temperature": 0.0,
    },
    "hallucination_risk": {
        "min_citation_density": 0.4,
        "max_semantic_variance": 0.45,
        "thresholds": {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.92,
        },
    },
    "refusal": {
        "messages": {
            "low_confidence": (
                "I do not have enough grounded evidence to answer this reliably."
            ),
            "no_access": (
                "You do not have permission to access the information needed for this query."
            ),
            "high_risk": (
                "I cannot answer because the available evidence is too risky to trust."
            ),
            "faithfulness_failure": (
                "I generated a draft answer, but it did not pass the faithfulness check."
            ),
            "insufficient_context": (
                "I could not find enough relevant context to answer your question."
            ),
        }
    },
}


TOP_K_RETRIEVAL = THRESHOLDS_CONFIG["retrieval_confidence"]["top_k"]
MIN_DOCUMENT_COUNT = THRESHOLDS_CONFIG["retrieval_confidence"]["min_document_count"]
MIN_SIMILARITY_THRESHOLD = THRESHOLDS_CONFIG["retrieval_confidence"]["min_similarity_threshold"]
CONFIDENCE_THRESHOLD = THRESHOLDS_CONFIG["retrieval_confidence"]["min_confidence"]
CONFIDENCE_WEIGHTS = THRESHOLDS_CONFIG["retrieval_confidence"]["weights"]
RISK_THRESHOLDS = THRESHOLDS_CONFIG["hallucination_risk"]["thresholds"]
REFUSAL_MESSAGES = THRESHOLDS_CONFIG["refusal"]["messages"]

AUTHORITY_SCORES = {
    "final_policy": 1.0,
    "approved_guideline": 0.9,
    "draft_policy": 0.6,
    "memo": 0.4,
}

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local").strip().lower()
GENERATION_MODEL = "claude-3-5-sonnet-latest"
GENERATION_TEMPERATURE = 0.0
JUDGE_MODEL = THRESHOLDS_CONFIG["faithfulness"]["judge_model"]
JUDGE_TEMPERATURE = THRESHOLDS_CONFIG["faithfulness"]["judge_temperature"]

GENERATION_PROMPT_TEMPLATE = """You are a governed enterprise assistant.

Answer ONLY from the supplied context.
- Cite supporting evidence with [Document N].
- If the context is incomplete, say so clearly.
- Do not use outside knowledge.

Context:
{context}

Question: {question}
"""

FAITHFULNESS_PROMPT_TEMPLATE = """Evaluate whether the answer is fully supported by the context.

Return JSON with:
- is_faithful: true or false
- confidence: number from 0.0 to 1.0
- unsupported_claims: list of unsupported statements
- reasoning: short explanation

Context:
{context}

Answer:
{answer}
"""


def ensure_directories() -> None:
    RAW_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_role(role: str) -> str:
    normalized = (role or "public").strip().lower()
    if normalized in ROLES_CONFIG["roles"]:
        return normalized
    return "public"


def get_allowed_authorities(role: str) -> List[str]:
    normalized = normalize_role(role)
    allowed = ROLES_CONFIG["roles"][normalized]["allowed_authorities"]
    if "*" in allowed:
        return list(ROLES_CONFIG["authorities"].keys())
    return list(allowed)


def get_roles_for_authority(authority: str) -> List[str]:
    normalized_authority = normalize_authority_label(authority)
    allowed_roles: List[str] = []
    for role_name, role_data in ROLES_CONFIG["roles"].items():
        if "*" in role_data["allowed_authorities"] or normalized_authority in role_data["allowed_authorities"]:
            allowed_roles.append(role_name)
    return allowed_roles


def get_refusal_message(reason: str) -> str:
    return REFUSAL_MESSAGES.get(reason, "I cannot answer this request safely.")


def normalize_authority_type(value: str | None) -> str:
    text = (value or "").strip().lower()
    if "final policy" in text:
        return "final_policy"
    if "approved guideline" in text or "guideline" in text:
        return "approved_guideline"
    if "draft" in text:
        return "draft_policy"
    return "memo"


def normalize_authority_label(
    value: str | None = None,
    classification: str | None = None,
    title: str | None = None,
) -> str:
    text = " ".join(part for part in [value or "", classification or "", title or ""] if part).lower()
    if "public" in text:
        return "public"
    if any(token in text for token in ["financial", "finance", "revenue", "budget", "cfo"]):
        return "financial"
    if any(token in text for token in ["strategic", "strategy", "board"]):
        return "strategic"
    if any(token in text for token in ["executive", "c-suite"]):
        return "executive"
    if any(token in text for token in ["manager", "management", "director"]):
        return "management"
    if any(token in text for token in ["engineering", "technical", "deployment", "api"]):
        return "engineering"
    if any(token in text for token in ["employee", "internal", "handbook", "hr"]):
        return "employee"
    return "public"
