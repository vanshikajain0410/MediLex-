"""
MediLex India — Faithfulness Evaluation (custom, RAGAS-style).

WHY THIS ISN'T `import ragas`:
Verified directly (not assumed) in three clean virtualenvs: `ragas==0.4.3`
and `ragas==0.3.9` both fail on import. They pull in
`langchain_community.chat_models.vertexai.ChatVertexAI`, a path
langchain-community has already removed as part of its own deprecation
("being sunset"). Pinning an older langchain-community to work around it
then collides with langchain-openai's langchain-core version requirement.
This is a current upstream dependency conflict, not something fixable from
this project's requirements.txt.

Faithfulness itself doesn't need the package. RAGAS's own method is:
  1. Decompose the generated answer into atomic claims (LLM call).
  2. For each claim, ask an LLM judge whether it's supported by the
     retrieved context (LLM call).
  3. faithfulness = supported_claims / total_claims.
This file reimplements exactly that, transparently, using Groq as the judge
— the SAME provider/key/quota already powering production generation, so
this doesn't introduce a new vendor or a hidden second consumer of quota.

SCOPE: faithfulness only. Context precision/recall are already covered, more
precisely, by `scripts/benchmark.py --eval`'s chunk-level Recall@5/MRR against
real ground truth — re-deriving that through an LLM judge would be a noisier
approximation of something already measured exactly.

COST: 3 Groq calls per query (1 generation + 1 claim extraction + 1
verification). Default run is the 25 scripts/benchmark.py BENCHMARK queries
(75 calls) — use --limit for a smaller smoke test first. Uses the SAME
GROQ_API_KEY/quota as production; this is not a separate free allowance.

Retrieval/generation use each BENCHMARK entry's own case_type/patient_age/
gender (see scripts/benchmark.py) via the same retrieve_hybrid()/
generate_checklist() calls production's main.py makes — not fixed values
reused across every query.

Faithfulness verification is tag-aware, matching generator.py's SYSTEM_PROMPT
tagging rule: claims tagged "[Act Section]" are checked against retrieved
context; claims tagged "[Clinical Best Practice]" are not, since they aren't
claiming statutory grounding in the first place.

Usage:
    python scripts/ragas_eval.py --limit 3          # smoke test first
    python scripts/ragas_eval.py                    # full 25-query pilot
    python scripts/ragas_eval.py --quiet --delay 2  # quieter, slower (rate-limit safety)
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # scripts/ dir

import config
from rag.generator import generate_checklist, call_groq_judge
from benchmark import BENCHMARK, retriever  # reuse the canonical 25 queries + warmed retriever


# ── Judge prompts ─────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = (
    "You are a precise, skeptical auditor. You follow instructions exactly "
    "and respond only in the exact JSON format requested — no prose, no "
    "markdown fences, no commentary outside the JSON."
)

CLAIM_EXTRACTION_PROMPT_TEMPLATE = """Break the following medico-legal checklist into a list of short, atomic, independently-verifiable claims. Each item in the checklist begins with a bracketed tag — either an Act/Section tag like "[BNS Section 124]" or "[Clinical Best Practice]". PRESERVE that exact leading tag as the start of each extracted claim; do not remove, alter, or infer a tag. If one checklist item contains multiple facts, repeat its original tag on each atomic claim you split it into. Do not add claims that aren't in the checklist.

CHECKLIST:
{answer_text}

Respond ONLY with a JSON array of strings. Example: ["[BNS Section 124] claim one", "[Clinical Best Practice] claim two"]"""

CLAIM_VERIFICATION_PROMPT_TEMPLATE = """You are auditing an AI system for legal hallucination. Each claim below starts with a bracketed tag applied by the system itself — either a specific Act/Section (e.g. "[BNS Section 124]") or "[Clinical Best Practice]".

Apply DIFFERENT rules depending on the tag:
- If tagged with a specific Act/Section: mark supported=true ONLY IF that exact Act and Section appears in the CONTEXT below. Mark false if the section is absent from the context, even if the legal point might be true as a general matter of law elsewhere — the system claimed context-grounding it doesn't have.
- If tagged "[Clinical Best Practice]": mark supported=true automatically. These claims aren't asserting statutory grounding, so they shouldn't be checked against the legal context at all. Only mark one of these false if it's tagged as Clinical Best Practice but is clearly not a real medical/procedural step (i.e. the tag itself looks misapplied).

CONTEXT:
{context}

CLAIMS (numbered):
{numbered_claims}

Respond ONLY with a JSON array of objects, same order as the claims, each shaped exactly like:
{{"claim": "<claim text>", "supported": true, "reason": "<one short sentence>"}}"""


# ── Parsing / formatting helpers ────────────────────────────────────────────

def _parse_json_lenient(raw: str):
    """Same defensive parsing generator.py's _parse_response uses — strip
    markdown fences, then json.loads. Returns None on failure rather than
    raising, since one bad judge call shouldn't crash the whole eval run."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = [l for l in cleaned.split("\n") if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        return None


def flatten_checklist(checklist: dict) -> str:
    """Turns the 4-category checklist dict into flat text for claim extraction."""
    parts = []
    for key in ("legal_obligations", "medical_actions", "documentation", "whom_to_inform"):
        items = checklist.get(key, [])
        if items:
            label = key.replace("_", " ").title()
            parts.append(f"{label}:\n" + "\n".join(f"- {item}" for item in items))
    return "\n\n".join(parts)


# ── Judge steps ───────────────────────────────────────────────────────────────

def extract_claims(answer_text: str) -> list[str] | None:
    raw = call_groq_judge(
        JUDGE_SYSTEM_PROMPT,
        CLAIM_EXTRACTION_PROMPT_TEMPLATE.format(answer_text=answer_text),
    )
    claims = _parse_json_lenient(raw)
    if not isinstance(claims, list):
        return None
    return [c for c in claims if isinstance(c, str) and c.strip()]


def verify_claims(context: str, claims: list[str]) -> list[dict] | None:
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    raw = call_groq_judge(
        JUDGE_SYSTEM_PROMPT,
        CLAIM_VERIFICATION_PROMPT_TEMPLATE.format(context=context, numbered_claims=numbered),
    )
    verdicts = _parse_json_lenient(raw)
    if not isinstance(verdicts, list) or len(verdicts) != len(claims):
        # Length mismatch means the judge dropped/added/reordered claims —
        # can't reliably attribute verdicts to claims, so treat as a failure
        # rather than silently miscounting.
        return None
    return verdicts


# ── Main eval loop ────────────────────────────────────────────────────────────

def run_faithfulness_eval(limit: int | None = None, verbose: bool = True, delay: float = 1.0) -> dict:
    """
    Per query: retrieve (hybrid — same path production main.py uses) ->
    generate (real generate_checklist(), whichever provider is configured) ->
    extract claims -> verify against context -> faithfulness = supported/total.
    """
    queries = BENCHMARK[:limit] if limit else BENCHMARK
    results = []

    print("=" * 70)
    print(f"  FAITHFULNESS EVAL (custom, Groq judge) — {len(queries)} queries")
    print(f"  Generation provider: {config.LLM_PROVIDER}  |  Judge: groq ({config.GROQ_MODEL})")
    print("=" * 70)

    for test in queries:
        query = test["query"]

        # Retrieval — retrieve_hybrid(), the SAME call main.py's production
        # endpoint makes (case_type expansion included), not the raw/unexpanded
        # path. combined_context comes back built-in — no need to reassemble it.
        rag_result = retriever.retrieve_hybrid(
            case_type=test["case_type"],
            symptoms=query,
            patient_age=test["patient_age"],
            n_results=config.TOP_K,
        )
        context = rag_result["combined_context"]

        # Generation — real generate_checklist(), same function main.py calls,
        # with per-query intake pulled from BENCHMARK (see benchmark.py) instead
        # of one constant reused across all queries.
        checklist = generate_checklist(
            patient_age=test["patient_age"],
            gender=test["gender"],
            case_type=test["case_type"],
            symptoms=query,
            rag_context=context,
        )
        answer_text = flatten_checklist(checklist)

        if not answer_text.strip():
            if verbose:
                print(f"\n[{test['id']}] SKIPPED — empty checklist (generation likely failed)")
            results.append({"id": test["id"], "faithfulness": None, "claims": []})
            continue

        claims = extract_claims(answer_text)
        if not claims:
            if verbose:
                print(f"\n[{test['id']}] SKIPPED — no claims extracted (judge parse failure)")
            results.append({"id": test["id"], "faithfulness": None, "claims": []})
            continue

        verdicts = verify_claims(context, claims)
        if not verdicts:
            if verbose:
                print(f"\n[{test['id']}] SKIPPED — claim verification failed or count mismatch")
            results.append({"id": test["id"], "faithfulness": None, "claims": claims})
            continue

        supported = sum(1 for v in verdicts if isinstance(v, dict) and v.get("supported") is True)
        total = len(verdicts)
        faithfulness = supported / total if total else None

        results.append({"id": test["id"], "faithfulness": faithfulness, "claims": verdicts})

        if verbose:
            print(f"\n[{test['id']}] {query[:70]}")
            print(f"  Faithfulness: {faithfulness:.2f}  ({supported}/{total} claims supported)")
            for v in verdicts:
                if isinstance(v, dict) and v.get("supported") is False:
                    print(f"    ⚠ UNSUPPORTED: {v.get('claim', '?')[:80]}")
                    print(f"      reason: {v.get('reason', '?')[:100]}")

        time.sleep(delay)  # 3 calls/query — stay under Groq free-tier RPM

    scored = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    mean_faithfulness = sum(scored) / len(scored) if scored else 0.0
    skipped = len(results) - len(scored)

    print("\n" + "=" * 70)
    print(f"  RESULTS  ({len(scored)}/{len(results)} queries scored, {skipped} skipped)")
    print("=" * 70)
    print(f"  Mean Faithfulness: {mean_faithfulness:.3f}")
    if skipped:
        print(f"  Skipped queries had a judge parse failure — check --quiet=False output above.")

    return {"mean_faithfulness": mean_faithfulness, "results": results, "skipped": skipped}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom RAGAS-style faithfulness evaluation using Groq as judge.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N benchmark queries (smoke test)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to sleep between queries (rate-limit safety)")
    args = parser.parse_args()

    if config.LLM_PROVIDER != "groq":
        print(
            f"NOTE: LLM_PROVIDER is '{config.LLM_PROVIDER}', not 'groq'. "
            f"generate_checklist() will use {config.LLM_PROVIDER} for the answers being "
            f"judged, but the judge itself always calls Groq (call_groq_judge). That's a "
            f"deliberate choice per this project's current setup, just flagging it in case "
            f"it's not what you expected.\n"
        )

    run_faithfulness_eval(limit=args.limit, verbose=not args.quiet, delay=args.delay)
