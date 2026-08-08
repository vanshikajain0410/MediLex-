"""
MediLex India — Retrieval Benchmark (25 medico-legal queries).

Run BEFORE plugging in the LLM.  Confirms the knowledge base returns
the correct statute sections for realistic doctor queries.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --quiet

Expected:
    >= 20/25 PASS  → proceed to LLM integration
    15–19/25       → review failed cases, check chunk coverage
    < 15/25        → re-chunk or re-embed; something is wrong
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import get_retriever

retriever = get_retriever()

BENCHMARK = [
    # ── BNS ──────────────────────────────────────────────────────────────────
    {"id": "BNS-01", "query": "What should a doctor do in a child sexual assault case — what law applies?",
     "expected_laws": ["POCSO", "BNS"]},
    {"id": "BNS-02", "query": "How do I classify an injury as grievous hurt in an MLC?",
     "expected_laws": ["BNS"]},
    {"id": "BNS-03", "query": "A patient has acid burns on face and neck — what is my legal duty?",
     "expected_laws": ["BNS"]},
    {"id": "BNS-04", "query": "Can a hospital refuse treatment to an acid attack victim?",
     "expected_laws": ["BNS"]},
    {"id": "BNS-05", "query": "Patient has multiple fractures from alleged domestic assault — what classification?",
     "expected_laws": ["BNS"]},
    {"id": "BNS-06", "query": "What is the legal definition of dowry death and when is it triggered?",
     "expected_laws": ["BNS"]},
    {"id": "BNS-07", "query": "A driver who caused a road accident and fled — what offence is this?",
     "expected_laws": ["BNS"]},
    {"id": "BNS-08", "query": "What does medical negligence causing death constitute under the new criminal law?",
     "expected_laws": ["BNS"]},
    # ── BNSS ─────────────────────────────────────────────────────────────────
    {"id": "BNSS-01", "query": "When is police intimation mandatory in a medico-legal case?",
     "expected_laws": ["BNSS"]},
    {"id": "BNSS-02", "query": "What are the steps for conducting a post-mortem in a suspicious death?",
     "expected_laws": ["BNSS"]},
    {"id": "BNSS-03", "query": "What is the procedure for medical examination of a rape victim?",
     "expected_laws": ["BNSS"]},
    {"id": "BNSS-04", "query": "Can police conduct a preliminary inquiry before filing FIR in a sexual assault case?",
     "expected_laws": ["BNSS"]},
    {"id": "BNSS-05", "query": "What are medico-legal obligations in a road traffic accident case?",
     "expected_laws": ["BNSS", "BNS"]},
    {"id": "BNSS-06", "query": "How should a doctor examine an arrested person who has injuries?",
     "expected_laws": ["BNSS"]},
    {"id": "BNSS-07", "query": "What documents must a post-mortem report contain?",
     "expected_laws": ["BNSS"]},
    {"id": "BNSS-08", "query": "What is Zero FIR and how does it affect the hospital?",
     "expected_laws": ["BNSS"]},
    # ── BSA ──────────────────────────────────────────────────────────────────
    {"id": "BSA-01", "query": "How should a doctor record a dying declaration?",
     "expected_laws": ["BSA"]},
    {"id": "BSA-02", "query": "Is DNA evidence admissible in court and what is the legal presumption?",
     "expected_laws": ["BSA"]},
    {"id": "BSA-03", "query": "What makes an MLC report admissible as evidence in court?",
     "expected_laws": ["BSA"]},
    {"id": "BSA-04", "query": "What are the duties of a doctor as an expert witness?",
     "expected_laws": ["BSA"]},
    {"id": "BSA-05", "query": "Can the previous sexual history of a rape victim be admitted as evidence?",
     "expected_laws": ["BSA", "BNSS"]},
    # ── POCSO / JJ Act ───────────────────────────────────────────────────────
    {"id": "POCSO-01", "query": "Who must report child sexual abuse to police and within what time?",
     "expected_laws": ["POCSO"]},
    {"id": "POCSO-02", "query": "Who can consent to treatment of a minor brought without a guardian?",
     "expected_laws": ["JJ_ACT", "POCSO"]},
    {"id": "JJ-01", "query": "What should a doctor do if a child in hospital appears to be neglected or abused?",
     "expected_laws": ["JJ_ACT"]},
    # ── MTP Act ──────────────────────────────────────────────────────────────
    {"id": "MTP-01", "query": "Can a rape survivor get abortion beyond 20 weeks of pregnancy?",
     "expected_laws": ["MTP_ACT"]},
]


def run_benchmark(verbose: bool = True) -> dict:
    passed = 0
    failed = []
    results_log = []

    print("=" * 70)
    print("  MEDILEX RETRIEVAL BENCHMARK — BNS / BNSS / BSA Edition")
    print(f"  {len(BENCHMARK)} queries | Top-6 chunks retrieved per query")
    print("=" * 70)

    for test in BENCHMARK:
        result = retriever.retrieve_raw(test["query"], n_results=6)
        retrieved_laws = result["laws_retrieved"]

        hit = any(law in retrieved_laws for law in test["expected_laws"])
        status = "✅ PASS" if hit else "❌ FAIL"
        if hit:
            passed += 1
        else:
            failed.append(test)

        top_score = result["chunks"][0]["relevance_score"] if result["chunks"] else 0.0
        top_law = result["chunks"][0]["law"] if result["chunks"] else "N/A"

        results_log.append({
            "id": test["id"],
            "status": status,
            "top_score": top_score,
            "top_law": top_law,
            "retrieved_laws": retrieved_laws,
        })

        if verbose:
            print(f"\n[{test['id']}] {status}")
            print(f"  Query   : {test['query'][:80]}")
            print(f"  Expected: {test['expected_laws']}")
            print(f"  Got     : {retrieved_laws}")
            print(f"  Top     : [{top_law}] score={top_score}")

    total = len(BENCHMARK)
    precision = round(passed / total * 100, 1)

    print("\n" + "=" * 70)
    print(f"  FINAL SCORE: {passed}/{total} ({precision}%)")
    print("=" * 70)

    if precision >= 80:
        print("  ✅ Knowledge base solid. Proceed to LLM integration.")
    elif precision >= 60:
        print("  ⚠️  Review failed cases. Check missing chunks.")
    else:
        print("  ❌ Re-chunk or re-embed. Knowledge base has gaps.")

    if failed:
        print(f"\n  Failed queries ({len(failed)}):")
        for f in failed:
            print(f"    [{f['id']}] {f['query'][:70]}")
            print(f"           Expected: {f['expected_laws']}")

    return {"passed": passed, "total": total, "precision_pct": precision, "results": results_log}


if __name__ == "__main__":
    verbose = "--quiet" not in sys.argv
    run_benchmark(verbose=verbose)
