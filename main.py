"""
MediLex India — FastAPI server.

Single entry point for the application.  Wires together:
  - RAG retrieval  (rag/retriever.py)
  - LLM generation (rag/generator.py)
  - SQLite logging (database.py)

Tech stack:
  FastAPI  — chosen because:
    - Automatic request validation via Pydantic (our CaseInput model
      rejects bad payloads before handler code runs).
    - Auto-generated /docs (Swagger) for testing without a frontend.
    - Async-capable but works fine synchronously (our I/O is SQLite
      writes + one HTTP call to Gemini per request).
    - Lighter than Django; more structured than raw Flask.

  Uvicorn  — ASGI server.  The standard way to run FastAPI.
    `python main.py` starts it on port 8000.

Run:
    python main.py
    # or: uvicorn main:app --reload
"""

import os
import time
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

import config
from schema import CaseInput
from database import init_db, log_session, log_protocol_result, log_error, get_stats
from rag.retriever import get_retriever
from rag.generator import generate_checklist, get_active_model_name


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MediLex India API",
    description="Medico-legal decision support for Indian clinicians.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Rate limiter (in-memory) ─────────────────────────────────────────────────
# Why in-memory instead of Redis / slowapi?
#   - Zero extra dependencies.
#   - Single-process deployment (Render free tier, HF Spaces) doesn't
#     need distributed state.
#   - Easy to explain: it's a dict of timestamps per IP.
#   - If you scale to multiple workers, swap to Redis.

_per_ip_timestamps: dict[str, list[float]] = defaultdict(list)
_daily_count: list[float] = []  # timestamps of all requests today


def _check_rate_limit(client_ip: str) -> str | None:
    """Returns an error message if rate limit exceeded, else None."""
    now = time.time()
    one_minute_ago = now - 60
    one_day_ago = now - 86400

    # Per-IP: max N requests per minute
    _per_ip_timestamps[client_ip] = [
        t for t in _per_ip_timestamps[client_ip] if t > one_minute_ago
    ]
    if len(_per_ip_timestamps[client_ip]) >= config.RATE_LIMIT_RPM:
        return (
            f"Rate limit exceeded: max {config.RATE_LIMIT_RPM} requests per minute. "
            "Please wait before retrying."
        )
    _per_ip_timestamps[client_ip].append(now)

    # Global daily cap (protects free-tier Gemini quota)
    global _daily_count
    _daily_count = [t for t in _daily_count if t > one_day_ago]
    if len(_daily_count) >= config.RATE_LIMIT_DAILY:
        return (
            f"Daily limit reached ({config.RATE_LIMIT_DAILY} requests/day). "
            "This is a demo deployment with limited API quota."
        )
    _daily_count.append(now)

    return None


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    init_db()
    retriever = get_retriever()  # warm up embeddings + ChromaDB (~5–10 s first time)
    retriever.warm_bm25_index()  # build BM25 now too, not on the first hybrid request
    print("✅ MediLex server ready.")


# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_case(data: CaseInput, request: Request):
    """
    Submit a patient case for medico-legal analysis.

    Flow:
      1. Rate-limit check
      2. Retrieve relevant statute chunks (hybrid: BM25 + dense, RRF-fused)
      2.5. Confidence gate — abstain if the best match is too weak (see config.CONFIDENCE_THRESHOLD)
      3. Generate structured checklist via Gemini
      4. Log session + result to SQLite
      5. Return checklist to frontend
    """
    # ── Rate limit ────────────────────────────────────────────────────────
    client_ip = request.client.host if request.client else "unknown"
    rate_error = _check_rate_limit(client_ip)
    if rate_error:
        raise HTTPException(status_code=429, detail=rate_error)

    session_id = None
    start_time = time.time()

    try:
        # STEP 1 — Retrieve statute chunks (hybrid: BM25 + dense, fused via RRF)
        # MEASURED (scripts/benchmark.py --eval): hybrid beats dense-only
        # Recall@5 0.816→0.863, matching MRR. Reranker measured net-negative
        # on this corpus (0.863→0.835 Recall@5) — gated behind
        # config.ENABLE_RERANKER (default False) rather than hardcoded off,
        # so re-enabling it later (a validated/fine-tuned reranker) is an
        # env change, not a code change.
        retriever = get_retriever()
        if config.ENABLE_RERANKER:
            rag_result = retriever.retrieve_hybrid_reranked(
                case_type=data.case_type,
                symptoms=data.symptoms,
                patient_age=data.patient_age,
            )
        else:
            rag_result = retriever.retrieve_hybrid(
                case_type=data.case_type,
                symptoms=data.symptoms,
                patient_age=data.patient_age,
            )

        # STEP 2 — Log session metadata (no PII)
        context = {
            "case_reference_number": f"AUTO-{int(time.time())}",
            "patient_age": data.patient_age,
            "sex_at_birth": data.gender,
            "injury_types": [data.case_type],
            "sexual_offense_suspected": "sexual" in data.case_type.lower(),
            "pregnancy_confirmed": False,
            "is_minor": rag_result["is_minor"],
            "hospital_type": "government",
        }
        session_id = log_session(
            context=context,
            laws_retrieved=rag_result["laws_retrieved"],
        )

        # STEP 2.5 — Confidence gate
        # Uses the top retrieved chunk's score. Prefers rerank_score if present
        # (only set when retrieve_hybrid_reranked() is in use) over relevance_score
        # (dense cosine similarity, ~0-1) — these are DIFFERENT SCALES. If this
        # endpoint is ever switched to hybrid/reranked retrieval, CONFIDENCE_THRESHOLD
        # must be recalibrated: a cross-encoder logit and a cosine similarity are not
        # interchangeable numbers. Default threshold is 0.0 (disabled, everything
        # passes) until a real value is set from evidence — see config.py.
        top_chunk = rag_result["chunks"][0] if rag_result["chunks"] else None
        if top_chunk is None:
            confidence_score = 0.0
        elif "rerank_score" in top_chunk:
            confidence_score = top_chunk["rerank_score"]
        else:
            confidence_score = top_chunk["relevance_score"]

        if confidence_score < config.CONFIDENCE_THRESHOLD:
            abstain_reason = (
                f"Best matching legal context scored {confidence_score:.3f}, "
                f"below the configured confidence threshold ({config.CONFIDENCE_THRESHOLD}). "
                "Returning without a generated checklist rather than answering from "
                "weak legal grounding — please rephrase with more specific case details."
            )
            log_protocol_result(
                session_id=session_id,
                protocol={"abstained": True, "reason": abstain_reason, "top_score": confidence_score},
                ai_model="none (abstained)",
                response_time_ms=int((time.time() - start_time) * 1000),
            )
            return {
                "session_id": session_id,
                "is_minor": rag_result["is_minor"],
                "laws_retrieved": rag_result["laws_retrieved"],
                "checklist": None,
                "abstained": True,
                "abstain_reason": abstain_reason,
                "confidence_score": confidence_score,
            }

        # STEP 3 — Generate checklist (provider dispatched by config.LLM_PROVIDER)
        checklist = generate_checklist(
            patient_age=data.patient_age,
            gender=data.gender,
            case_type=data.case_type,
            symptoms=data.symptoms,
            rag_context=rag_result["combined_context"],
        )

        # STEP 4 — Log result
        response_ms = int((time.time() - start_time) * 1000)
        log_protocol_result(
            session_id=session_id,
            protocol=checklist,
            ai_model=get_active_model_name(),
            response_time_ms=response_ms,
        )

        # STEP 5 — Return to frontend
        return {
            "session_id": session_id,
            "is_minor": rag_result["is_minor"],
            "laws_retrieved": rag_result["laws_retrieved"],
            "checklist": checklist,
            "abstained": False,
            "confidence_score": confidence_score,
        }

    except HTTPException:
        raise  # re-raise rate limit errors as-is
    except Exception as e:
        log_error(
            error_message=str(e),
            endpoint="/api/analyze",
            session_id=session_id,
        )
        raise HTTPException(status_code=500, detail=str(e))


# ── Stats endpoint ────────────────────────────────────────────────────────────

@app.get("/api/stats")
def stats():
    """Aggregate statistics across all logged sessions."""
    return get_stats()


# ── Health check ──────────────────────────────────────────────────────────────

# ── Frontend ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """
    Serves the demo UI directly, so a deployed instance (Docker/HF Spaces/
    Render) shows the actual app at its URL, not just Swagger docs at /docs.
    Local dev can still open frontend.html directly as a file — see the
    protocol-aware API_BASE in frontend.html for why both paths work.
    """
    return FileResponse("frontend.html")


@app.get("/health")
def health():
    """Simple health check for deployment probes."""
    retriever = get_retriever()
    return {
        "status": "ok",
        "chunks_loaded": retriever.collection.count(),
        "model": get_active_model_name(),
    }


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)