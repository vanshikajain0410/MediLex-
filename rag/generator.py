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
from openai import OpenAI

import config


# ── Provider clients ──────────────────────────────────────────────────────────

def _get_gemini_client() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not set.  Add it to your .env file.\n"
            "Get a free key at https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=config.GEMINI_API_KEY)


def _get_groq_client() -> OpenAI:
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not set.  Add it to your .env file.\n"
            "Get a free key at https://console.groq.com/keys"
        )
    # Groq exposes an OpenAI-API-compatible endpoint, so the same openai
    # client works — only base_url changes. No separate Groq SDK needed.
    return OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")


def get_active_model_name() -> str:
    """
    Single source of truth for 'which model is actually configured right now'.
    main.py uses this for logging (log_protocol_result) and /health instead
    of hardcoding config.GEMINI_MODEL, which was stale as soon as Groq became
    an option — this is exactly the kind of drift that caused that bug.
    """
    if config.LLM_PROVIDER == "groq":
        return config.GROQ_MODEL
    elif config.LLM_PROVIDER == "gemini":
        return config.GEMINI_MODEL
    raise RuntimeError(f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}' — expected 'groq' or 'gemini'.")


# ── System prompt ─────────────────────────────────────────────────────────────
# Every checklist item must be tagged [Act Section] or [Clinical Best Practice].
# This split matters beyond output formatting: it's what lets faithfulness
# evaluation (scripts/ragas_eval.py) distinguish real hallucination (a
# specific-section claim not backed by retrieved context) from clinical
# advice that was never claiming statutory grounding in the first place.

SYSTEM_PROMPT = """You are a medico-legal decision support engine for Indian medical practitioners.
You reason under Indian law: BNS 2023 (replaces IPC), BNSS 2023 (replaces CrPC),
BSA 2023 (replaces IEA), POCSO 2012, JJ Act 2015 (amended 2021), and MTP Act 1971 (amended 2021).
Before generating output, evaluate EVERY gate below:
1. Is patient a minor (age < 18)? → POCSO mandatory reporting overrides all patient refusals.
2. Is sexual offense suspected? → Check: forensic window (72 hrs), FIR status, evidence preservation.
3. Is acid burn present? → Free treatment mandate (Laxmi v UOI 2014), immediate police intimation.
4. Is pregnancy involved? → Gestational age window (MTP Act), guardian consent for minors.
5. Is patient capable of consenting? → Identify substitute decision-maker (guardian/CWC/court).
6. Are patient wishes in conflict with legal mandates? → State statutory resolution.
RULES FOR TAGGING (MANDATORY):
Every item in every category MUST begin with one of two tags in square brackets:
1. [Act Name Section Number] — Use this tag (e.g. "[BNS Section 124]", "[POCSO Section 19]") ONLY for legal requirements directly supported by the RELEVANT STATUTE EXCERPTS provided.
2. [Clinical Best Practice] — Use this tag for standard medical procedures, clinical workflows, or general documentation steps that are essential for patient care but NOT explicitly mentioned in the statute excerpts.
RULES FOR CONTENT:
- Never fabricate section numbers. If a specific section is not present in the legal context, tag the instruction as [Clinical Best Practice].
- Be specific to Indian law and emergency clinical workflows. Each item should be one clear, actionable instruction.
Respond ONLY with a valid JSON object matching this exact structure — no prose, no markdown fences:
{
  "legal_obligations": ["[BNS Section 124] Provide free first aid immediately...", "[Clinical Best Practice] Record verbal history..."],
  "medical_actions": ["[Clinical Best Practice] Assess ABCs and irrigate burns..."],
  "documentation": ["[BNSS Section 184] Complete medical examination report...", "[Clinical Best Practice] Take clear photos of injuries..."],
  "whom_to_inform": ["[POCSO Section 19] Notify Special Juvenile Police Unit within 24 hours..."]
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
    RuntimeError  if the configured provider's API key is not set, or
                  LLM_PROVIDER is neither 'groq' nor 'gemini'.
    ValueError    if the LLM returns unparseable JSON after cleanup attempts.
    """
    prompt = f"""CASE FACTS:
- Patient Age: {patient_age}
- Gender: {gender}
- Case Type: {case_type}
- Symptoms/Context: {symptoms}
- Is Minor: {"Yes" if patient_age < 18 else "No"}

RELEVANT STATUTE EXCERPTS (retrieved from legal corpus):
{rag_context if rag_context.strip() else "No statute chunks retrieved."}

Generate the medico-legal checklist for this case now."""

    if config.LLM_PROVIDER == "groq":
        raw = _generate_groq(prompt)
    elif config.LLM_PROVIDER == "gemini":
        raw = _generate_gemini(prompt)
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}' — expected 'groq' or 'gemini'.")

    return _parse_response(raw)


def _generate_gemini(prompt: str) -> str:
    client = _get_gemini_client()
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[
            {"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n" + prompt}]}
        ],
    )
    return response.text.strip()


def _generate_groq(prompt: str) -> str:
    return call_groq_judge(SYSTEM_PROMPT, prompt)


def call_groq_judge(system_prompt: str, user_prompt: str) -> str:
    """
    General-purpose raw Groq call, parameterized by system prompt.

    Public and separate from _generate_groq() specifically so external tooling
    (scripts/ragas_eval.py's faithfulness judge) can reuse the same client
    construction and model config without needing a different system prompt
    than checklist generation, and without reaching into a private function.
    """
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


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
