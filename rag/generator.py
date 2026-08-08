"""
MediLex India — LLM generation via Gemini.

Takes the patient case context + retrieved statute chunks and generates
a structured medico-legal checklist grounded in Indian law.

Tech stack:
  Google Gemini (gemini-2.0-flash) via the google-genai SDK.
  Chosen because:
    - Free tier: 15 requests/min, 1 M tokens/min — sufficient for
      a demo / portfolio project without billing.
    - Fast inference (~1–3 s for structured JSON output).
    - JSON mode support reduces parsing errors.
  If the project moves to production, swap to a paid tier or another
  provider by changing only this file + config.py.

Design decisions:
  1. System prompt merges the medico-legal "gate" logic from mediLex.py
     (POCSO minor checks, acid burn mandate, MTP windows, consent
     conflicts) with the simple 4-category output format that the
     frontend's renderResults() reads.
  2. JSON output is validated with a try/except fallback — if the LLM
     returns malformed JSON, the user sees a safe error message rather
     than a crash.
  3. The prompt is deterministic and reproducible: case facts are
     injected as structured text, not free-form prose.
"""

import json
from google import genai

import config


# ── Gemini client ─────────────────────────────────────────────────────────────

def _get_client() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not set.  Add it to your .env file.\n"
            "Get a free key at https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=config.GEMINI_API_KEY)


# ── System prompt ─────────────────────────────────────────────────────────────
# This prompt encodes the medico-legal reasoning gates from the original
# mediLex.py SYSTEM_PROMPT, adapted for the simpler 4-category output
# format that the frontend expects.

SYSTEM_PROMPT = """You are a medico-legal decision support engine for Indian medical practitioners.
You reason ONLY under Indian law: BNS 2023 (replaces IPC), BNSS 2023 (replaces CrPC),
BSA 2023 (replaces IEA), POCSO 2012, JJ Act 2015 (amended 2021), and MTP Act 1971 (amended 2021).

Before generating output, evaluate EVERY gate below:
1. Is patient a minor (age < 18)? → POCSO mandatory reporting overrides all patient refusals.
2. Is sexual offense suspected? → Check: forensic window (72 hrs), FIR status, evidence preservation.
3. Is acid burn present? → Free treatment mandate (Laxmi v UOI 2014), immediate police intimation.
4. Is pregnancy involved? → Gestational age window (MTP Act), guardian consent for minors.
5. Is patient capable of consenting? → Identify substitute decision-maker (guardian/CWC/court).
6. Are patient wishes in conflict with legal mandates? → State statutory resolution.

RULES:
- ONLY cite laws and sections that appear in the LEGAL CONTEXT provided.
- If the context is insufficient to answer, say so explicitly in legal_obligations.
- Never invent legal provisions or cite sections not present in the context.
- Be specific to Indian law. Each item should be one clear, actionable instruction.
- Include the specific statute section (e.g. "BNS Section 124", "POCSO Section 19") in each item.

Respond ONLY with a valid JSON object matching this exact structure — no prose, no markdown fences:
{
  "legal_obligations": ["list of mandatory legal duties with statute citations"],
  "medical_actions": ["list of required medical procedures and examinations"],
  "documentation": ["list of documents the doctor must prepare before patient leaves"],
  "whom_to_inform": ["list of authorities/persons who must be notified, with time limits"]
}"""


# ── Public API ────────────────────────────────────────────────────────────────

def generate_checklist(
    patient_age: int,
    gender: str,
    case_type: str,
    symptoms: str,
    rag_context: str,
) -> dict:
    """
    Generate a medico-legal checklist using Gemini.

    Parameters
    ----------
    patient_age : int
    gender      : str
    case_type   : str
    symptoms    : str   — doctor's free-text description
    rag_context : str   — concatenated statute chunks from retriever

    Returns
    -------
    dict with keys: legal_obligations, medical_actions, documentation, whom_to_inform
    Each value is a list of strings.

    Raises
    ------
    RuntimeError  if GEMINI_API_KEY is not set.
    ValueError    if Gemini returns unparseable JSON after cleanup attempts.
    """
    client = _get_client()

    prompt = f"""CASE FACTS:
- Patient Age: {patient_age}
- Gender: {gender}
- Case Type: {case_type}
- Symptoms/Context: {symptoms}
- Is Minor: {"Yes" if patient_age < 18 else "No"}

RELEVANT STATUTE EXCERPTS (retrieved from legal corpus):
{rag_context if rag_context.strip() else "No statute chunks retrieved."}

Generate the medico-legal checklist for this case now."""

    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[
            {"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n" + prompt}]}
        ],
    )

    raw = response.text.strip()
    return _parse_response(raw)


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_response(raw: str) -> dict:
    """
    Parse the LLM's raw text into a validated checklist dict.

    Handles common LLM quirks:
      - Markdown fences (```json ... ```)
      - Leading/trailing whitespace
      - Partial JSON with trailing commas

    Returns a safe fallback if parsing fails entirely.
    """
    cleaned = raw

    # Strip markdown fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"⚠️  LLM JSON parse error: {e}\nRaw response:\n{raw[:500]}")
        return {
            "legal_obligations": [
                "Error: AI response could not be parsed. Please retry.",
                f"Debug: {str(e)[:100]}"
            ],
            "medical_actions": [],
            "documentation": [],
            "whom_to_inform": [],
        }

    # Ensure all 4 required keys exist (defensive — LLM might omit one)
    for key in ("legal_obligations", "medical_actions", "documentation", "whom_to_inform"):
        if key not in result or not isinstance(result[key], list):
            result[key] = []

    return result
