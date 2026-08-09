import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.benchmark import BENCHMARK, retriever

def analyze_thresholds():
    print("Loading ground truth...")
    try:
        with open("scripts/ground_truth.json", "r") as f:
            ground_truth = json.load(f)
    except FileNotFoundError:
        print("Error: scripts/ground_truth.json not found.")
        return

    print("Warming up BM25 index...")
    retriever._ensure_bm25_index()

    scores = []
    
    for test in BENCHMARK:
        q_id = test["id"]
        if q_id not in ground_truth:
            continue
            
        # Run hybrid retrieval (Dense + BM25)
        res = retriever.retrieve_raw_hybrid(test["query"], n_results=6)
        
        if not res["chunks"]:
            continue
            
        top_chunk = res["chunks"][0]
        top_score = top_chunk.get("relevance_score", 0.0)
        chunk_id = top_chunk.get("chunk_id", "")
        
        # Verify if the highest-ranked chunk is actually correct
        is_hit = chunk_id in ground_truth[q_id]
        scores.append((top_score, is_hit, q_id))

    # Sort scores from lowest to highest to identify the natural drop-off point
    scores.sort(key=lambda x: x[0])
    
    print("\n=== RRF CONFIDENCE SCORE DISTRIBUTION ===")
    print(f"{'Score':<8} | {'Hit?':<6} | {'Query ID'}")
    print("-" * 35)
    for score, is_hit, qid in scores:
        hit_marker = "✅ YES" if is_hit else "❌ NO"
        print(f"{score:.4f}   | {hit_marker:<6} | {qid}")

if __name__ == "__main__":
    analyze_thresholds()