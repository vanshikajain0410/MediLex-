"""
MediLex India — Out-of-domain confidence probe.

The missing verification step before treating CONFIDENCE_THRESHOLD (candidate:
0.0250, see config.py) as validated rather than extrapolated: do queries with
nothing to do with Indian medico-legal law actually score below it, on the
same retrieve_hybrid() RRF score the real gate uses?

Retrieval-only, no LLM calls — fast and free to run.

Usage:
    python scripts/probe_confidence.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from rag.retriever import get_retriever

PROBES = [
    "How do I bake a chocolate cake?",
    "What's the best programming language for web development?",
    "Recommend a good restaurant in Mumbai",
    "What's tomorrow's weather forecast?",
]

CANDIDATE_THRESHOLD = 0.0250
IN_DOMAIN_RANGE = (0.0315, 0.0328)  # measured across the 25 BENCHMARK queries


def main():
    retriever = get_retriever()
    print("=" * 70)
    print("  OUT-OF-DOMAIN CONFIDENCE PROBE")
    print(f"  In-domain range (25 BENCHMARK queries): {IN_DOMAIN_RANGE[0]}–{IN_DOMAIN_RANGE[1]}")
    print(f"  Candidate CONFIDENCE_THRESHOLD: {CANDIDATE_THRESHOLD}")
    print("=" * 70)

    all_below = True
    for q in PROBES:
        result = retriever.retrieve_raw_hybrid(q, n_results=config.TOP_K)
        top_score = result["chunks"][0]["relevance_score"] if result["chunks"] else 0.0
        top_law = result["chunks"][0]["law"] if result["chunks"] else "N/A"
        below = top_score < CANDIDATE_THRESHOLD
        all_below = all_below and below
        mark = "✅ below threshold" if below else "⚠️  NOT below threshold"
        print(f"\n[{q}]")
        print(f"  Top score: {top_score:.4f}  (top law: {top_law})  {mark}")

    print("\n" + "=" * 70)
    if all_below:
        print(f"  All probes scored below {CANDIDATE_THRESHOLD} — supports it as a")
        print(f"  garbage-input floor. Safe to set CONFIDENCE_THRESHOLD={CANDIDATE_THRESHOLD}.")
    else:
        print(f"  At least one probe did NOT score below {CANDIDATE_THRESHOLD} — the")
        print(f"  threshold wouldn't catch it. Consider raising it, or note that RRF")
        print(f"  top-1 score alone may not be a reliable out-of-domain signal here.")
    print("=" * 70)


if __name__ == "__main__":
    main()
