"""
MediLex India — Retrieval Benchmark (25 medico-legal queries).

Run BEFORE plugging in the LLM.  Confirms the knowledge base returns
the correct statute sections for realistic doctor queries.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --quiet
    python scripts/benchmark.py --ablation          # dense vs hybrid, side by side
    python scripts/benchmark.py --dump-candidates   # step 1: build ground_truth.json
    python scripts/benchmark.py --eval              # step 2: Recall@5 / MRR, 4-way

Expected:
    >= 20/25 PASS  → proceed to LLM integration
    15–19/25       → review failed cases, check chunk coverage
    < 15/25        → re-chunk or re-embed; something is wrong

Ground truth for --eval:
    Recall@k and MRR need chunk-level ground truth (which exact chunk IDs
    are correct for a query), not just "which law". That's a domain
    judgment call this script won't make for you. Run --dump-candidates
    first, review the printed candidates, and save the ones you confirm
    relevant into scripts/ground_truth.json as {"QUERY_ID": ["chunk_id", ...]}.
    Queries missing from that file are skipped by --eval, not guessed.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import get_retriever

retriever = get_retriever()

GROUND_TRUTH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth.json")

BENCHMARK = [
    # ── BNS ──────────────────────────────────────────────────────────────────
    # case_type/patient_age/gender are illustrative per-query intake, not a
    # single "correct" demographic — these queries are generic legal questions,
    # not real patients. They exist as structured data (not hardcoded in the
    # eval loop) so retrieve_hybrid()'s case_type keyword expansion and
    # generate_checklist()'s CASE FACTS block get real per-query input instead
    # of one constant reused for all 25 queries. Edit freely per query.
    {"id": "BNS-01", "query": "What should a doctor do in a child sexual assault case — what law applies?",
     "expected_laws": ["POCSO", "BNS"], "case_type": "pocso", "patient_age": 10, "gender": "female"},
    {"id": "BNS-02", "query": "How do I classify an injury as grievous hurt in an MLC?",
     "expected_laws": ["BNS"], "case_type": "physical_assault", "patient_age": 35, "gender": "male"},
    {"id": "BNS-03", "query": "A patient has acid burns on face and neck — what is my legal duty?",
     "expected_laws": ["BNS"], "case_type": "acid_burn", "patient_age": 28, "gender": "female"},
    {"id": "BNS-04", "query": "Can a hospital refuse treatment to an acid attack victim?",
     "expected_laws": ["BNS"], "case_type": "acid_burn", "patient_age": 28, "gender": "female"},
    {"id": "BNS-05", "query": "Patient has multiple fractures from alleged domestic assault — what classification?",
     "expected_laws": ["BNS"], "case_type": "domestic_violence", "patient_age": 32, "gender": "female"},
    {"id": "BNS-06", "query": "What is the legal definition of dowry death and when is it triggered?",
     "expected_laws": ["BNS"], "case_type": "death", "patient_age": 26, "gender": "female"},
    {"id": "BNS-07", "query": "A driver who caused a road accident and fled — what offence is this?",
     "expected_laws": ["BNS"], "case_type": "road_accident", "patient_age": 40, "gender": "male"},
    {"id": "BNS-08", "query": "What does medical negligence causing death constitute under the new criminal law?",
     "expected_laws": ["BNS"], "case_type": "death", "patient_age": 50, "gender": "male"},
    # ── BNSS ─────────────────────────────────────────────────────────────────
    {"id": "BNSS-01", "query": "When is police intimation mandatory in a medico-legal case?",
     "expected_laws": ["BNSS"], "case_type": "mlc_documentation", "patient_age": 30, "gender": "male"},
    {"id": "BNSS-02", "query": "What are the steps for conducting a post-mortem in a suspicious death?",
     "expected_laws": ["BNSS"], "case_type": "death", "patient_age": 45, "gender": "male"},
    {"id": "BNSS-03", "query": "What is the procedure for medical examination of a rape victim?",
     "expected_laws": ["BNSS"], "case_type": "sexual_assault", "patient_age": 24, "gender": "female"},
    {"id": "BNSS-04", "query": "Can police conduct a preliminary inquiry before filing FIR in a sexual assault case?",
     "expected_laws": ["BNSS"], "case_type": "sexual_assault", "patient_age": 24, "gender": "female"},
    {"id": "BNSS-05", "query": "What are medico-legal obligations in a road traffic accident case?",
     "expected_laws": ["BNSS", "BNS"], "case_type": "road_accident", "patient_age": 35, "gender": "male"},
    {"id": "BNSS-06", "query": "How should a doctor examine an arrested person who has injuries?",
     "expected_laws": ["BNSS"], "case_type": "custodial_examination", "patient_age": 30, "gender": "male"},
    {"id": "BNSS-07", "query": "What documents must a post-mortem report contain?",
     "expected_laws": ["BNSS"], "case_type": "death", "patient_age": 45, "gender": "male"},
    {"id": "BNSS-08", "query": "What is Zero FIR and how does it affect the hospital?",
     "expected_laws": ["BNSS"], "case_type": "mlc_documentation", "patient_age": 30, "gender": "male"},
    # ── BSA ──────────────────────────────────────────────────────────────────
    {"id": "BSA-01", "query": "How should a doctor record a dying declaration?",
     "expected_laws": ["BSA"], "case_type": "dying_declaration", "patient_age": 50, "gender": "male"},
    {"id": "BSA-02", "query": "Is DNA evidence admissible in court and what is the legal presumption?",
     "expected_laws": ["BSA"], "case_type": "dna_evidence", "patient_age": 30, "gender": "male"},
    {"id": "BSA-03", "query": "What makes an MLC report admissible as evidence in court?",
     "expected_laws": ["BSA"], "case_type": "mlc_documentation", "patient_age": 30, "gender": "male"},
    {"id": "BSA-04", "query": "What are the duties of a doctor as an expert witness?",
     "expected_laws": ["BSA"], "case_type": "mlc_documentation", "patient_age": 40, "gender": "male"},
    {"id": "BSA-05", "query": "Can the previous sexual history of a rape victim be admitted as evidence?",
     "expected_laws": ["BSA", "BNSS"], "case_type": "sexual_assault", "patient_age": 24, "gender": "female"},
    # ── POCSO / JJ Act ───────────────────────────────────────────────────────
    {"id": "POCSO-01", "query": "Who must report child sexual abuse to police and within what time?",
     "expected_laws": ["POCSO"], "case_type": "pocso", "patient_age": 12, "gender": "female"},
    {"id": "POCSO-02", "query": "Who can consent to treatment of a minor brought without a guardian?",
     "expected_laws": ["JJ_ACT", "POCSO"], "case_type": "minor_consent", "patient_age": 15, "gender": "male"},
    {"id": "JJ-01", "query": "What should a doctor do if a child in hospital appears to be neglected or abused?",
     "expected_laws": ["JJ_ACT"], "case_type": "pocso", "patient_age": 8, "gender": "male"},
    # ── MTP Act ──────────────────────────────────────────────────────────────
    {"id": "MTP-01", "query": "Can a rape survivor get abortion beyond 20 weeks of pregnancy?",
     "expected_laws": ["MTP_ACT"], "case_type": "pregnancy_termination", "patient_age": 22, "gender": "female"},
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


def run_ablation(verbose: bool = True) -> dict:
    """
    Dense-only (retrieve_raw) vs hybrid (retrieve_raw_hybrid), same queries,
    same n_results — isolates whether BM25+RRF fusion actually changes
    outcomes, and specifically whether it helps or hurts.

    A pass-count delta alone can hide regressions (e.g. +3/-1 nets +2, but
    that -1 is a real query hybrid made worse), so flips are tracked and
    reported explicitly in both directions.
    """
    dense_passed = 0
    hybrid_passed = 0
    gained = []   # dense FAIL -> hybrid PASS
    regressed = []  # dense PASS -> hybrid FAIL
    results_log = []

    print("=" * 70)
    print("  MEDILEX ABLATION — Dense-only vs Hybrid (BM25 + RRF)")
    print(f"  {len(BENCHMARK)} queries | Top-6 chunks | identical n_results both paths")
    print("=" * 70)

    for test in BENCHMARK:
        dense_result = retriever.retrieve_raw(test["query"], n_results=6)
        hybrid_result = retriever.retrieve_raw_hybrid(test["query"], n_results=6)

        dense_laws = dense_result["laws_retrieved"]
        hybrid_laws = hybrid_result["laws_retrieved"]

        dense_hit = any(law in dense_laws for law in test["expected_laws"])
        hybrid_hit = any(law in hybrid_laws for law in test["expected_laws"])

        dense_passed += int(dense_hit)
        hybrid_passed += int(hybrid_hit)

        flip = ""
        if hybrid_hit and not dense_hit:
            gained.append(test)
            flip = "  ⬆ GAINED by hybrid"
        elif dense_hit and not hybrid_hit:
            regressed.append(test)
            flip = "  ⬇ REGRESSED by hybrid"

        results_log.append({
            "id": test["id"],
            "dense_hit": dense_hit,
            "hybrid_hit": hybrid_hit,
            "dense_laws": dense_laws,
            "hybrid_laws": hybrid_laws,
        })

        if verbose:
            d_mark = "✅" if dense_hit else "❌"
            h_mark = "✅" if hybrid_hit else "❌"
            print(f"\n[{test['id']}] dense={d_mark} hybrid={h_mark}{flip}")
            print(f"  Query   : {test['query'][:80]}")
            print(f"  Expected: {test['expected_laws']}")
            print(f"  Dense   : {dense_laws}")
            print(f"  Hybrid  : {hybrid_laws}")

    total = len(BENCHMARK)
    dense_pct = round(dense_passed / total * 100, 1)
    hybrid_pct = round(hybrid_passed / total * 100, 1)

    print("\n" + "=" * 70)
    print("  ABLATION SUMMARY")
    print("=" * 70)
    print(f"  Dense-only : {dense_passed}/{total} ({dense_pct}%)")
    print(f"  Hybrid     : {hybrid_passed}/{total} ({hybrid_pct}%)")
    print(f"  Net delta  : {hybrid_passed - dense_passed:+d} queries "
          f"({hybrid_pct - dense_pct:+.1f} pp)")

    if gained:
        print(f"\n  Gained by hybrid ({len(gained)}):")
        for t in gained:
            print(f"    [{t['id']}] {t['query'][:70]}")
    if regressed:
        print(f"\n  Regressed by hybrid ({len(regressed)}):")
        for t in regressed:
            print(f"    [{t['id']}] {t['query'][:70]}")
    if not gained and not regressed:
        print("\n  No flips — hybrid and dense agree on every query.")

    return {
        "dense_passed": dense_passed,
        "hybrid_passed": hybrid_passed,
        "total": total,
        "gained": [t["id"] for t in gained],
        "regressed": [t["id"] for t in regressed],
        "results": results_log,
    }


def dump_candidates(pool_size: int = 10) -> None:
    """
    Prints candidate chunks per BENCHMARK query (dense ∪ BM25 ∪ hybrid,
    deduplicated) so a human with legal-domain judgment can pick out which
    ones are actually correct and build scripts/ground_truth.json.

    Deliberately does not rank or pre-select "the answer" — that would just
    be this script grading its own homework. It surfaces candidates; a
    person decides relevance.
    """
    print("=" * 70)
    print("  CANDIDATE DUMP — for building scripts/ground_truth.json")
    print(f"  Up to {pool_size} candidates per query, pulled from dense ∪ BM25 ∪ hybrid")
    print("=" * 70)

    for test in BENCHMARK:
        candidates: dict[str, dict] = {}
        found_by: dict[str, set] = {}

        for label, result in (
            ("dense", retriever.retrieve_raw(test["query"], n_results=pool_size)),
            ("bm25", retriever.retrieve_raw_bm25(test["query"], n_results=pool_size)),
            ("hybrid", retriever.retrieve_raw_hybrid(test["query"], n_results=pool_size)),
        ):
            for c in result["chunks"]:
                cid = c["chunk_id"]
                candidates.setdefault(cid, c)
                found_by.setdefault(cid, set()).add(label)

        print(f"\n[{test['id']}] {test['query']}")
        print(f"  Expected laws (coarse, from BENCHMARK): {test['expected_laws']}")
        for cid, c in candidates.items():
            sources = ",".join(sorted(found_by[cid]))
            preview = c["text"][:90].replace("\n", " ")
            section = c.get("section") or "-"
            print(f"    {cid:16s} [{sources:16s}] {c['law']} {section:>5} | {preview}")

    print("\n" + "=" * 70)
    print("  Next: create scripts/ground_truth.json with the chunk_ids you")
    print("  confirm are actually relevant, keyed by query id, e.g.:")
    print('    {"BNS-01": ["POCSO_2", "BNS_14"], "BNS-02": ["BNS_7"]}')
    print("  Then run: python scripts/benchmark.py --eval")
    print("=" * 70)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float | None:
    """Fraction of the known-relevant chunks that appear in the top-k retrieved.
    Returns None (not 0.0) when there's no ground truth to compare against —
    the two mean different things and shouldn't be averaged together."""
    if not relevant_ids:
        return None
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / (rank of first relevant chunk). 0.0 if none of the relevant chunks appear."""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def run_evaluation(k: int = 5, pool: int = 10, verbose: bool = True) -> dict:
    """
    Recall@k and MRR across four retrieval configurations — Dense, BM25,
    Hybrid, Hybrid+Reranker — scored against scripts/ground_truth.json.

    This is the artifact that actually validates whether hybrid retrieval
    (step 3) and reranking (step 4) helped, as opposed to --ablation's
    law-name-level pass/fail, which saturates at 100% for both dense and
    hybrid on the current 25-query set and can't distinguish them further.

    All four methods are queried with the same pool depth (`pool`, default
    10) so Recall@k/MRR are computed over comparably deep rankings.
    """
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"No ground truth file at {GROUND_TRUTH_PATH}.")
        print("Run `python scripts/benchmark.py --dump-candidates` first, review")
        print("the output, and save confirmed-relevant chunk_ids there.")
        return {}

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth: dict[str, list[str]] = json.load(f)

    if not ground_truth:
        print(f"{GROUND_TRUTH_PATH} exists but is empty — nothing to evaluate.")
        return {}

    methods = {
        "Dense": lambda q: retriever.retrieve_raw(q, n_results=pool),
        "BM25": lambda q: retriever.retrieve_raw_bm25(q, n_results=pool),
        "Hybrid": lambda q: retriever.retrieve_raw_hybrid(
            q, n_results=pool, candidate_pool_size=pool * 2
        ),
        "Hybrid+Reranker": lambda q: retriever.retrieve_raw_hybrid_reranked(
            q, n_results=pool, rerank_pool_size=pool * 2, rerank=True
        ),
    }
    scores = {name: {"recall": [], "mrr": []} for name in methods}
    skipped = []

    print("=" * 70)
    print(f"  MEDILEX EVALUATION — Recall@{k} / MRR — Dense / BM25 / Hybrid / Hybrid+Reranker")
    print("=" * 70)

    for test in BENCHMARK:
        relevant = set(ground_truth.get(test["id"], []))
        if not relevant:
            skipped.append(test["id"])
            continue

        if verbose:
            print(f"\n[{test['id']}] {test['query'][:70]}")

        for name, fn in methods.items():
            result = fn(test["query"])
            retrieved_ids = [c["chunk_id"] for c in result["chunks"]]
            r = recall_at_k(retrieved_ids, relevant, k)
            rr = reciprocal_rank(retrieved_ids, relevant)
            scores[name]["recall"].append(r)
            scores[name]["mrr"].append(rr)
            if verbose:
                print(f"    {name:<16s} Recall@{k}={r:.2f}  RR={rr:.2f}")

    total_annotated = len(BENCHMARK) - len(skipped)
    print("\n" + "=" * 70)
    print(f"  ABLATION TABLE  ({total_annotated}/{len(BENCHMARK)} queries have ground truth)")
    print("=" * 70)
    header = f"  {'Method':<18}{'Recall@' + str(k):<12}{'MRR':<10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, s in scores.items():
        mean_recall = sum(s["recall"]) / len(s["recall"]) if s["recall"] else 0.0
        mean_mrr = sum(s["mrr"]) / len(s["mrr"]) if s["mrr"] else 0.0
        print(f"  {name:<18}{mean_recall:<12.3f}{mean_mrr:<10.3f}")

    if skipped:
        print(f"\n  Skipped (no ground truth yet): {skipped}")

    return {"scores": scores, "skipped": skipped, "total_annotated": total_annotated}


if __name__ == "__main__":
    verbose = "--quiet" not in sys.argv
    if "--dump-candidates" in sys.argv:
        dump_candidates()
    elif "--eval" in sys.argv:
        run_evaluation(verbose=verbose)
    elif "--ablation" in sys.argv:
        run_ablation(verbose=verbose)
    else:
        run_benchmark(verbose=verbose)
