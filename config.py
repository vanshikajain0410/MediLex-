"""
MediLex India — Configuration & Guardrails.

Single source of truth for API keys, model settings, rate limits,
and safety guardrails. Every tunable lives here so nothing is
scattered across files.

Why a plain module instead of pydantic-settings or dynaconf?
  - Zero extra dependencies.
  - Easy to explain in an interview: "it's just a Python file
    that reads env vars with sensible defaults."
  - pydantic-settings would be a fine upgrade later, but YAGNI now.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in project root


# ── Gemini API ────────────────────────────────────────────────────────────────
# Why Gemini?  Free tier (15 RPM, 1 M tokens/min for gemini-2.0-flash).
# Swapping to another provider later means changing only this file + rag/generator.py.

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


# ── Retrieval ─────────────────────────────────────────────────────────────────

CHROMA_PATH: str = "chroma_db"
COLLECTION_NAME: str = "medilex_laws"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
TOP_K: int = 6  # chunks returned per query


# ── Rate Limiting (in-memory, per-IP) ─────────────────────────────────────────
# Why in-memory instead of Redis?  Single-process deployment on free tiers
# (Render, HF Spaces) doesn't need distributed state.  If you scale to
# multi-worker, swap to Redis or a DB-backed counter.

RATE_LIMIT_RPM: int = int(os.getenv("RATE_LIMIT_RPM", "10"))         # requests per minute per IP
RATE_LIMIT_DAILY: int = int(os.getenv("RATE_LIMIT_DAILY", "200"))    # requests per day (all IPs combined)


# ── Input Guardrails ──────────────────────────────────────────────────────────
# Prevent prompt injection and abuse via the free-text symptoms field.

MAX_SYMPTOMS_LENGTH: int = 1000   # characters — a few sentences is plenty
MIN_PATIENT_AGE: int = 0
MAX_PATIENT_AGE: int = 120


# ── Confidence Gating (used by generator to decide abstention) ────────────────
# If the best retrieval score is below this threshold, the system declines
# to generate a checklist and returns an "insufficient context" response.
# Tuned later via the evaluation harness; 0.0 disables gating for now.

CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.0"))


# ── CORS ──────────────────────────────────────────────────────────────────────
# Tighten for production; "*" is fine for local development.

CORS_ORIGINS: list[str] = os.getenv(
    "CORS_ORIGINS", "*"
).split(",")


# ── SQLite ────────────────────────────────────────────────────────────────────

DB_PATH: str = "medilex.db"
