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

import time
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import config
from schema import CaseInput
from database import init_db, log_session, log_protocol_result, log_error, get_stats
from rag.retriever import get_retriever
from rag.generator import generate_checklist


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
    get_retriever()  # warm up embeddings + ChromaDB (~5–10 s first time)
    print("✅ MediLex server ready.")


# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_case(data: CaseInput, request: Request):
    """
    Submit a patient case for medico-legal analysis.

    Flow:
      1. Rate-limit check
      2. Retrieve relevant statute chunks from ChromaDB
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
        # STEP 1 — Retrieve statute chunks
        retriever = get_retriever()
        rag_result = retriever.retrieve(
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

        # STEP 3 — Generate checklist via Gemini
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
            ai_model=config.GEMINI_MODEL,
            response_time_ms=response_ms,
        )

        # STEP 5 — Return to frontend
        return {
            "session_id": session_id,
            "is_minor": rag_result["is_minor"],
            "laws_retrieved": rag_result["laws_retrieved"],
            "checklist": checklist,
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

@app.get("/health")
def health():
    """Simple health check for deployment probes."""
    retriever = get_retriever()
    return {
        "status": "ok",
        "chunks_loaded": retriever.collection.count(),
        "model": config.GEMINI_MODEL,
    }


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
