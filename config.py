"""
MediLex India — Configuration & Guardrails.

Single source of truth for API keys, model settings, rate limits,
and safety guardrails. Every tunable lives here so nothing is
scattered across files.

LLM_PROVIDER controls which backend generator.py uses:
  "groq"   — Groq (recommended: 14,400 req/day free, no credit card)
  "gemini" — Google Gemini (free tier varies by region; try if Groq unavailable)

Swapping providers = one line change in .env.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── LLM Provider ──────────────────────────────────────────────────────────────
# "groq" or "gemini" — controls which generator backend is used.

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")


# ── Groq (recommended free tier) ─────────────────────────────────────────────
# Free tier: 30 RPM, 14,400 req/day, no credit card required.
# Get a key at: https://console.groq.com/keys

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ── Gemini (alternative) ──────────────────────────────────────────────────────
# Free tier availability varies by region and project.
# Get a key at: https://aistudio.google.com/apikey (create key from AI Studio,
# not Google Cloud Console — AI Studio projects have better free quota).

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ── Retrieval ─────────────────────────────────────────────────────────────────

CHROMA_PATH: str = "chroma_db"
COLLECTION_NAME: str = "medilex_laws"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
TOP_K: int = 6

# Cross-encoder for reranking the hybrid candidate set (retriever.py's
# retrieve_hybrid_reranked()). ms-marco-MiniLM-L-6-v2 is a small (~80 MB),
# CPU-fast, well-established passage reranker — trained on general web
# search relevance (MS MARCO), not Indian legal text specifically. It's a
# reasonable off-the-shelf starting point at this corpus size; if the
# eval harness shows it underperforming on legal queries, fine-tuning a
# reranker on domain query/chunk pairs is the natural next step, not a
# bigger off-the-shelf model.
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Not yet consumed anywhere — main.py still calls retriever.retrieve() (dense-only).
# MEASURED (scripts/benchmark.py --eval, 25 queries, ground_truth.json): the
# reranker net-HURTS on this corpus — Recall@5 0.863→0.835, MRR 0.980→0.907
# vs. plain Hybrid. ms-marco-MiniLM-L-6-v2 is trained on general web search
# relevance, not Indian statutory text, and it demotes several first-page-
# correct chunks (e.g. BNSS-07: MRR 1.00→0.33). Keep this False until either
# a domain-appropriate reranker is evaluated, or a fine-tuned one is trained
# on this corpus's query/chunk pairs — not "because it's built."
ENABLE_RERANKER: bool = os.getenv("ENABLE_RERANKER", "false").lower() == "true"


# ── Rate Limiting (in-memory, per-IP) ─────────────────────────────────────────

RATE_LIMIT_RPM: int = int(os.getenv("RATE_LIMIT_RPM", "10"))
RATE_LIMIT_DAILY: int = int(os.getenv("RATE_LIMIT_DAILY", "200"))


# ── Input Guardrails ──────────────────────────────────────────────────────────

MAX_SYMPTOMS_LENGTH: int = 1000
MIN_PATIENT_AGE: int = 0
MAX_PATIENT_AGE: int = 120


# ── Confidence Gating ────────────────────────────────────────────────────────
# Gates on the top retrieved chunk's score, checked in main.py's analyze_case().
# main.py calls retrieve_hybrid() — this is an RRF score (k=60). Max possible
# = 1/61 + 1/61 ≈ 0.0328 (rank #1 in both Dense and BM25).
#
# MEASURED (scripts/benchmark.py --eval + scripts/probe_confidence.py):
#   In-domain (25 real BENCHMARK queries):        0.0315 - 0.0328
#   Out-of-domain (4 probes: cake/code/food/weather): 0.0272 - 0.0301
# Real gap at (0.0301, 0.0315). Threshold set at 0.0310 — biased toward the
# conservative (higher) end of that gap, not the midpoint: for a legal-safety
# tool, an unnecessary "please rephrase" is a cheaper mistake than answering
# confidently on off-topic input.
#
# SCOPE LIMITATION — read before trusting this too far: this separates
# in-domain from out-of-domain queries. It does NOT separate correct from
# incorrect retrieval among in-domain queries. JJ-01 (scripts/benchmark.py
# BENCHMARK) is a confirmed retrieval miss in --eval, yet scored 0.0325 —
# solidly "high confidence." The gate catches garbage input; it doesn't catch
# a legitimate question that happened to retrieve the wrong section.
#
# Sample size is small (4 out-of-domain probes, 25 in-domain queries) — this
# is a real, evidenced improvement over an unverified guess, not a precisely
# tuned final value. Expand both probe sets over time rather than treating
# 0.0310 as settled.

CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.0310"))

# UI display bands (frontend.html's confidence badge color) — updated to match.
# MEDIUM now starts above the entire measured out-of-domain range (0.0301 max),
# not inside it as the old 0.0250 value did — that was showing off-topic
# queries as "medium confidence," which was actively misleading.
CONFIDENCE_BAND_HIGH: float = 0.0320    # majority cluster of well-retrieved queries
CONFIDENCE_BAND_MEDIUM: float = 0.0310  # matches CONFIDENCE_THRESHOLD — anything lower is "low" in the UI too


# ── CORS ──────────────────────────────────────────────────────────────────────

CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")


# ── SQLite ────────────────────────────────────────────────────────────────────

DB_PATH: str = "medilex.db"