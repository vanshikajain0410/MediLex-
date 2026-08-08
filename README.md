# MediLex India 🏥⚖️

**AI-powered medico-legal decision support for Indian doctors.**

MediLex helps clinicians navigate their legal obligations in real time. Given a patient's case details, it retrieves relevant Indian statutes and generates a structured checklist of legal duties, medical actions, documentation requirements, and notification obligations.

---

## Architecture

```
frontend.html (browser)
      │  POST /api/analyze
      ▼
main.py (FastAPI + rate limiter)
      │
      ├─ 1. rag/retriever.py
      │      Keyword-expanded query → ChromaDB → top-6 statute chunks
      │
      ├─ 2. database.py
      │      Anonymised session metadata → SQLite
      │
      ├─ 3. rag/generator.py
      │      Case facts + statute chunks → Gemini → structured checklist
      │
      └─ 4. Response → frontend renders 4-category checklist
```

## Tech Stack

| Layer | Technology | Why This Choice |
|---|---|---|
| **Backend** | FastAPI | Auto-validates requests via Pydantic, auto-generates Swagger docs, async-capable, lighter than Django |
| **LLM** | Google Gemini (gemini-2.0-flash) | Free tier (15 RPM, 1M tokens/min), fast inference, JSON output support |
| **Vector DB** | ChromaDB | Embedded (no server), persists to local SQLite, simple API, good for <500 chunks |
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) | Lightweight (80 MB), fast CPU inference (<100ms/query), solid English semantic quality |
| **Logging** | SQLite | Zero deployment overhead, file-based, portable, sufficient for demo-scale traffic |
| **Frontend** | Vanilla HTML/CSS/JS | Single file, no build step, no framework lock-in, works by opening in a browser |
| **Config** | python-dotenv | Simple .env file for secrets, no heavyweight settings framework needed |

## Project Structure

```
medilex/
├── main.py                  # FastAPI server — single entry point
├── config.py                # Settings, rate limits, guardrails (all tunables here)
├── schema.py                # Pydantic request/response models
├── database.py              # SQLite session/error logging
├── rag/
│   ├── retriever.py         # ChromaDB + MiniLM dense retriever
│   └── generator.py         # Gemini LLM wrapper + prompt engineering
├── scripts/
│   ├── build_database.py    # Embed statutes into ChromaDB (run once)
│   ├── benchmark.py         # 25-query retrieval evaluation
│   └── test_api.py          # Smoke test for the running server
├── data/
│   └── raw/                 # Statute text files (6 Indian laws)
├── chroma_db/               # Pre-built vector store (124 chunks)
├── frontend.html            # Single-page UI
├── requirements.txt
├── .env.example             # Template for environment variables
├── .gitignore
└── README.md
```

**Every file has a single responsibility.  No file is dead code.**

## Supported Case Types

| Case Type | Key Laws Retrieved |
|---|---|
| `pocso` | POCSO 2012, JJ Act 2015 |
| `sexual_assault` | BNS 64/65/70, POCSO, BNSS 184 |
| `acid_burn` | BNS 124 |
| `pregnancy_termination` | MTP Act 1971 (amended 2021) |
| `road_accident` | BNSS inquest, BNS 106 |
| `death` | BNSS 176/177/181 |
| `physical_assault` | BNS 114/115/117/118 |
| `domestic_violence` | BNS 85/86 |
| `poisoning` | BNS 118, BNSS |

## Getting Started

### Prerequisites

- Python 3.10+
- A free Gemini API key ([get one here](https://aistudio.google.com/apikey))

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/medilex.git
cd medilex
pip install -r requirements.txt
```

### Configure API Key

```bash
cp .env.example .env
# Edit .env and paste your GEMINI_API_KEY
```

### Run the Server

```bash
python main.py
# Server starts at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### Open the Frontend

Open `frontend.html` in your browser.  It connects to `http://localhost:8000` by default.

### (Optional) Rebuild the Vector Database

Only needed if you modify the statute files in `data/raw/`:

```bash
python scripts/build_database.py
```

### Run the Retrieval Benchmark

```bash
python scripts/benchmark.py
```

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
    "legal_obligations": ["File POCSO report within 24 hours (Section 19)"],
    "medical_actions": ["Conduct examination per BNSS Section 184"],
    "documentation": ["MLC form", "Injury certificate"],
    "whom_to_inform": ["Special Juvenile Police Unit", "CWC within 24 hours"]
  }
}
```

### `GET /api/stats`

Aggregate session statistics.

### `GET /health`

Health check (deployment probes).

## Guardrails & Rate Limiting

| Protection | Default | Configurable via |
|---|---|---|
| Per-IP rate limit | 10 req/min | `RATE_LIMIT_RPM` in .env |
| Daily global cap | 200 req/day | `RATE_LIMIT_DAILY` in .env |
| Input length limit | 1000 chars (symptoms field) | `MAX_SYMPTOMS_LENGTH` in config.py |
| Age bounds | 0–120 | `MIN/MAX_PATIENT_AGE` in config.py |
| Confidence gating | Disabled (threshold=0.0) | `CONFIDENCE_THRESHOLD` in .env |

## Privacy

- No PII is stored.  Sessions are identified by auto-generated reference numbers.
- All data stays local in `medilex.db`.
- The only external call is to the Gemini API (case type, age, gender, symptoms, and retrieved legal text are sent).
- CORS is set to `*` for local development.  Tighten `CORS_ORIGINS` in `.env` for production.

## Disclaimer

MediLex is a **decision support tool** and does not replace legal advice or clinical judgment.  Doctors should verify obligations with their hospital's legal team and the relevant statutory authorities.
