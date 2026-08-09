# MediLex India 🏥⚖️

**AI-powered medico-legal decision support for Indian doctors.**

MediLex helps clinicians navigate their legal obligations in real time. Given a patient's case details, it retrieves relevant Indian statutes and generates a structured checklist of legal duties, medical actions, documentation requirements, and notification obligations — with every item tagged as either a specific statute citation or a general clinical best practice, so the grounding is explicit rather than implied.

---

## Architecture

```
frontend.html (browser)
      │  POST /api/analyze
      ▼
main.py (FastAPI + rate limiter + confidence gate)
      │
      ├─ 1. rag/retriever.py — Hybrid retrieval
      │      Dense (MiniLM + ChromaDB) and sparse (BM25) run independently,
      │      fused via Reciprocal Rank Fusion (RRF). Measured to beat dense-
      │      only on Recall@5 (see Evaluation below). An optional cross-
      │      encoder reranker exists but is OFF by default — measured to
      │      hurt on this corpus, not just untested.
      │
      ├─ 2. Confidence gate (main.py)
      │      Top retrieved chunk's RRF score checked against
      │      CONFIDENCE_THRESHOLD. Below threshold → abstain, no LLM call,
      │      session still logged. Currently disabled (0.0) pending
      │      out-of-domain verification — see scripts/probe_confidence.py.
      │
      ├─ 3. database.py
      │      Anonymised session metadata → SQLite (including abstained sessions)
      │
      ├─ 4. rag/generator.py
      │      Case facts + statute chunks → Groq or Gemini → tagged checklist.
      │      Every item is prefixed [Act Section] or [Clinical Best Practice] —
      │      this split is what makes faithfulness evaluation meaningful
      │      (see Evaluation below).
      │
      └─ 5. Response → frontend renders checklist with citation-tag chips,
             a confidence badge, and an explicit low-confidence state
```

## Tech Stack

| Layer          | Technology                                                    | Why This Choice                                                                                                                                                                                                             |
| -------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Backend**    | FastAPI                                                       | Auto-validates requests via Pydantic, auto-generates Swagger docs, async-capable, lighter than Django                                                                                                                       |
| **LLM**        | Groq (default) or Gemini                                      | `LLM_PROVIDER` env var switches providers with no code change. Groq's free tier (14,400 req/day) is generous for a demo; Gemini is the fallback.                                                                            |
| **Retrieval**  | Hybrid: ChromaDB (dense) + rank_bm25 (sparse), fused via RRF  | Dense embeddings miss exact section numbers and acronyms (`Section 106`, `MLC`); BM25 misses semantic paraphrase (`unlawful killing` ≈ `culpable homicide`). Measured Recall@5 improvement over dense-only: see Evaluation. |
| **Reranking**  | Cross-encoder (`ms-marco-MiniLM-L-6-v2`), disabled by default | Measured net-negative on this corpus (Recall@5 0.863→0.835) — trained on general web search relevance, not Indian statutory text. Kept as an opt-in, evaluated code path, not deleted.                                      |
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2)                       | Lightweight (80 MB), fast CPU inference (<100ms/query), solid English semantic quality                                                                                                                                      |
| **Vector DB**  | ChromaDB                                                      | Embedded (no server), persists to local SQLite, simple API, good for <500 chunks                                                                                                                                            |
| **Logging**    | SQLite                                                        | Zero deployment overhead, file-based, portable, sufficient for demo-scale traffic                                                                                                                                           |
| **Frontend**   | Vanilla HTML/CSS/JS                                           | Single file, no build step, no framework lock-in, works by opening in a browser                                                                                                                                             |
| **Config**     | python-dotenv                                                 | Simple .env file for secrets, no heavyweight settings framework needed                                                                                                                                                      |

## Project Structure

```
medilex/
├── main.py                    # FastAPI server — single entry point
├── config.py                  # Settings, rate limits, guardrails, confidence bands (all tunables here)
├── schema.py                  # Pydantic request/response models
├── database.py                # SQLite session/error logging
├── rag/
│   ├── retriever.py           # Dense + BM25 hybrid retriever, RRF fusion, optional reranker
│   └── generator.py           # Groq/Gemini dispatch + tagged-checklist prompt engineering
├── scripts/
│   ├── build_database.py      # Embed statutes into ChromaDB (run once, or at Docker build time)
│   ├── benchmark.py           # 25-query retrieval benchmark: law-level pass/fail, dense-vs-hybrid
│   │                          #   ablation, and chunk-level Recall@5/MRR eval
│   ├── ragas_eval.py          # Custom faithfulness evaluator (LLM-as-judge via Groq) — see Evaluation
│   ├── probe_confidence.py    # Out-of-domain confidence threshold verification
│   ├── ground_truth.json      # Human-reviewed chunk-level ground truth for --eval (25 queries)
│   └── test_api.py            # Smoke test for the running server
├── data/
│   └── raw/                   # Statute text files (6 Indian laws) — committed, used to rebuild chroma_db/
├── chroma_db/                 # Vector store (124 chunks) — gitignored, rebuilt from data/raw/
├── frontend.html              # Single-page UI — citation tags, confidence badge, abstention state
├── Dockerfile                 # Rebuilds chroma_db/ at image-build time (see Deployment)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Supported Case Types

| Case Type               | Key Laws Retrieved            |
| ----------------------- | ----------------------------- |
| `pocso`                 | POCSO 2012, JJ Act 2015       |
| `sexual_assault`        | BNS 64/65/70, POCSO, BNSS 184 |
| `acid_burn`             | BNS 124                       |
| `pregnancy_termination` | MTP Act 1971 (amended 2021)   |
| `road_accident`         | BNSS inquest, BNS 106         |
| `death`                 | BNSS 176/177/181              |
| `physical_assault`      | BNS 114/115/117/118           |
| `domestic_violence`     | BNS 85/86                     |
| `poisoning`             | BNS 118, BNSS                 |

## Evaluation

Two separate evaluations, measuring different things — retrieval quality and generation faithfulness are not the same question and shouldn't be conflated.

### Retrieval: Recall@5 / MRR ablation (25 queries, real chunk-level ground truth)

```
Method            Recall@5    MRR
Dense             0.816       0.980
BM25              0.730       0.853
Hybrid            0.863       0.980
Hybrid+Reranker   0.835       0.907
```

Hybrid (BM25 + dense, RRF-fused) beats dense-only on Recall@5 — modest but real and consistent, not noise. BM25 alone underperforms dense, which is exactly why fusion rather than a BM25 swap: the win comes from combining signals, not replacing one with the other. The reranker measurably _hurts_ — reproduce with `python scripts/benchmark.py --eval`; see `--dump-candidates` / `ground_truth.json` for how chunk-level ground truth was built (candidates are surfaced for human review, nothing is auto-labeled).

### Generation: Faithfulness (custom LLM-as-judge, Groq)

The `ragas` package (v0.3.9 and v0.4.3) currently fails to import in a clean environment — verified directly, not assumed; it pulls a `langchain_community` path that package has since removed. `scripts/ragas_eval.py` reimplements RAGAS's own faithfulness method directly: decompose the generated checklist into atomic claims, ask an LLM judge (Groq) whether each is supported by the retrieved context, score = supported/total. Verification is tag-aware — `[Clinical Best Practice]` claims aren't checked against legal context, since they're not claiming statutory grounding; `[Act Section]` claims are.

Smoke-tested on 3 of the 25 benchmark queries: **mean faithfulness 0.949** (11/13, 27/27, 12/12 claims supported). This is a pilot result, not yet run across the full 25 — treat it as promising, not conclusive, until it is. Run `python scripts/ragas_eval.py` for the full set (uses the same Groq quota as production; ~3 calls/query).

### Confidence threshold calibration

`CONFIDENCE_THRESHOLD` (`0.0310`) was set from measured data, not guessed: the top chunk's RRF score across 25 real in-domain queries falls in `0.0315–0.0328`; four deliberately out-of-domain probes (cake recipes, programming languages, restaurants, weather — see `scripts/probe_confidence.py`) scored `0.0272–0.0301`. The threshold sits in the gap, biased toward the conservative side — for a legal-safety tool, an unnecessary "please rephrase" costs less than a confident answer built on irrelevant context.

**Scope limitation, stated plainly:** this separates in-domain from out-of-domain queries. It does not separate correct from incorrect retrieval among in-domain queries — JJ-01 (a confirmed retrieval miss in the `--eval` run) still scores `0.0325`, comfortably "high confidence." The gate catches garbage input, not subtly wrong retrieval on a legitimate question. Sample size is small (4 out-of-domain probes) — a real, evidenced improvement over an unverified guess, not a precisely tuned final value.

## Getting Started

### Prerequisites

- Python 3.10+
- A free Groq API key ([get one here](https://console.groq.com/keys)) — default provider. Gemini is supported as an alternative (set `LLM_PROVIDER=gemini`).

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/medilex.git
cd medilex
pip install -r requirements.txt
```

### Configure API Key

```bash
cp .env.example .env
# Edit .env:
#   LLM_PROVIDER=groq
#   GROQ_API_KEY=your_key_here
# (or LLM_PROVIDER=gemini + GEMINI_API_KEY, if you prefer)
```

### Run the Server

```bash
python main.py
# Server starts at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### Open the Frontend

Open `frontend.html` in your browser. It connects to `http://localhost:8000` by default.

### (Optional) Rebuild the Vector Database

Only needed if you modify the statute files in `data/raw/`:

```bash
python scripts/build_database.py
```

### Run the Retrieval Benchmark

```bash
python scripts/benchmark.py                    # law-level pass/fail (25 queries)
python scripts/benchmark.py --ablation          # dense vs hybrid, side by side
python scripts/benchmark.py --dump-candidates   # step 1 of building ground_truth.json
python scripts/benchmark.py --eval              # Recall@5/MRR, 4-way ablation (needs ground_truth.json)
```

## Deployment

**Live:** https://YOUR-RENDER-URL.onrender.com _(replace with actual URL)_

Deployed on [Render](https://render.com) (Docker runtime). `Dockerfile` builds a self-contained image — `chroma_db/` is gitignored, so it's rebuilt from `data/raw/*.txt` at **image build time**, not expected to already exist. Port is read from the `PORT` env var at runtime, which Render injects automatically — no manual port config needed on the platform. The same image is portable to Hugging Face Spaces (Docker SDK) or Railway without modification, though HF Spaces' Docker tier is no longer free, which is why Render is the current target.

### Render setup

1. New Web Service → connect this repo → runtime: **Docker** (auto-detected from `Dockerfile`).
2. Set environment variables in the Render dashboard: `GROQ_API_KEY`, `LLM_PROVIDER=groq` (or `gemini` + `GEMINI_API_KEY`).
3. No `PORT` variable needs to be set manually — Render provides it at runtime.

**Note:** on Render's free tier, the service spins down after inactivity — expect a cold-start delay (10–50s) on the first request after idling. `medilex.db` (session logging) does not persist across restarts/redeploys on this tier — acceptable for a demo; use a paid tier with a persistent disk if session history needs to survive.

### Local Docker (for testing before pushing)

```bash
docker build -t medilex .
docker run -p 8000:7860 -e PORT=7860 -e GROQ_API_KEY=your_key -e LLM_PROVIDER=groq medilex
```

`medilex.db` (session logging) is not persisted across container restarts on a typical free-tier deploy — acceptable for a demo; mount a volume if session history needs to survive restarts.

## API Reference

### `POST /api/analyze`

Submit a case for medico-legal analysis.

```json
{
  "patient_age": 15,
  "gender": "female",
  "case_type": "pocso",
  "symptoms": "bruising, distress, disclosed assault by relative"
}
```

**Response:**

```json
{
  "session_id": 1,
  "is_minor": true,
  "laws_retrieved": ["POCSO", "JJ_ACT", "BNS"],
  "checklist": {
    "legal_obligations": ["[POCSO Section 19] File report within 24 hours"],
    "medical_actions": ["[BNSS Section 184] Conduct examination per protocol"],
    "documentation": [
      "[Clinical Best Practice] MLC form",
      "[Clinical Best Practice] Injury certificate"
    ],
    "whom_to_inform": [
      "[POCSO Section 19] Special Juvenile Police Unit",
      "[Clinical Best Practice] CWC within 24 hours"
    ]
  },
  "abstained": false,
  "confidence_score": 0.0325
}
```

If retrieval confidence is below `CONFIDENCE_THRESHOLD`, `checklist` is `null`, `abstained` is `true`, and `abstain_reason` explains why — no LLM call is made.

### `GET /api/stats`

Aggregate session statistics.

### `GET /health`

Health check (deployment probes). Reports the actually-active model (`get_active_model_name()`), not a hardcoded value.

## Guardrails & Rate Limiting

| Protection         | Default                                                                                     | Configurable via                                                   |
| ------------------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Per-IP rate limit  | 10 req/min                                                                                  | `RATE_LIMIT_RPM` in .env                                           |
| Daily global cap   | 200 req/day                                                                                 | `RATE_LIMIT_DAILY` in .env                                         |
| Input length limit | 1000 chars (symptoms field)                                                                 | `MAX_SYMPTOMS_LENGTH` in config.py                                 |
| Age bounds         | 0–120                                                                                       | `MIN/MAX_PATIENT_AGE` in config.py                                 |
| Confidence gating  | Enabled, `0.0310` — verified against 25 in-domain + 4 out-of-domain probes (see Evaluation) | `CONFIDENCE_THRESHOLD` in .env — see `scripts/probe_confidence.py` |
| Reranker           | Disabled — measured net-negative on this corpus                                             | `ENABLE_RERANKER` in .env                                          |

## Privacy

- No PII is stored. Sessions are identified by auto-generated reference numbers.
- All data stays local in `medilex.db`.
- The only external call is to the configured LLM provider (Groq or Gemini) — case type, age, gender, symptoms, and retrieved legal text are sent. The faithfulness evaluator additionally sends generated checklists and context to Groq as a judge, using the same key/quota.
- CORS is set to `*` for local development. Tighten `CORS_ORIGINS` in `.env` for production.

## Disclaimer

MediLex is a **decision support tool** and does not replace legal advice or clinical judgment. Doctors should verify obligations with their hospital's legal team and the relevant statutory authorities.
