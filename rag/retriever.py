"""
MediLex India — ChromaDB retriever.

Retrieves relevant statute chunks from the local vector database
built by scripts/build_database.py.

Tech stack:
  ChromaDB     — embedded vector database.  Chosen because it runs
                 in-process (no separate server), persists to a local
                 SQLite file, and has a simple query API.  Good enough
                 for ~100–500 chunks; if the corpus grows to tens of
                 thousands, migrate to pgvector or Qdrant.

  SentenceTransformers (all-MiniLM-L6-v2) — lightweight 384-dim
                 encoder.  Chosen for fast inference on CPU (~80 MB
                 download, <100 ms per query) and decent semantic
                 quality on English legal text.  A domain-fine-tuned
                 model (e.g. legal-bert) could improve recall but
                 would increase size and latency.

Laws covered (post-2023 reform):
  BNS  (Bharatiya Nyaya Sanhita)     — replaces IPC
  BNSS (Bharatiya Nagarik Suraksha)  — replaces CrPC
  BSA  (Bharatiya Sakshya Adhiniyam) — replaces IEA
  POCSO 2012, JJ Act 2015 (amended 2021), MTP Act 1971 (amended 2021)
"""

import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

import config

# ── Case-type keyword expansion ───────────────────────────────────────────────
# Maps each case_type to a bag of keywords injected into the retrieval query.
# This compensates for short user queries (e.g. "pocso") by surfacing the
# specific statute sections that a dense-only retriever might miss.
#
# Maintenance note: when laws are amended, update these keyword strings
# *and* re-run scripts/build_database.py to re-chunk the new text.

CASE_TYPE_KEYWORDS: dict[str, str] = {
    "acid_burn": (
        "acid attack burns BNS 124 assault grievous hurt free treatment "
        "corrosive substance disfiguration"
    ),
    "sexual_assault": (
        "rape sexual abuse POCSO victim assault BNS 64 65 70 forensic examination "
        "BNSS 184 two-finger test banned DNA BSA 156"
    ),
    "pocso": (
        "child minor sexual abuse POCSO reporting mandatory JJ Act BNS 64 "
        "aggravated penetrative sexual assault special court"
    ),
    "pregnancy_termination": (
        "MTP abortion termination pregnancy weeks rape survivor "
        "medical termination registered medical practitioner"
    ),
    "road_accident": (
        "accident injury MLC medico legal case BNSS inquest BNS 106 "
        "rash negligent death Good Samaritan"
    ),
    "death": (
        "death unnatural BNS BNSS post mortem inquest autopsy "
        "BNSS 176 177 181 custodial death suspicious circumstances"
    ),
    "physical_assault": (
        "assault hurt grievous BNS 114 115 117 118 injury MLC "
        "simple hurt grievous hurt fracture weapon"
    ),
    "domestic_violence": (
        "domestic violence dowry assault protection women BNS 85 86 "
        "cruelty husband dowry death within seven years marriage"
    ),
    "poisoning": (
        "poisoning toxic substance MLC BNSS BNS 118 corrosive "
        "inhalation chemical evidence BSA expert"
    ),
    "dying_declaration": (
        "dying declaration BSA 23 fitness statement death imminent "
        "Magistrate oral written gesture doctor certify"
    ),
    "mlc_documentation": (
        "MLC medico legal case documentation MLC register injury certificate "
        "BNSS 184 185 BSA 37 expert witness doctor duty"
    ),
    "dna_evidence": (
        "DNA evidence BSA 156 forensic sample chain of custody swab "
        "biological evidence rape sexual assault presumption"
    ),
    "custodial_examination": (
        "arrested person examination BNSS 58 medical examination custody "
        "injury marks custodial torture female doctor"
    ),
    "minor_consent": (
        "minor consent treatment guardian JJ Act POCSO CWC Child Welfare "
        "Committee 24 hours production child in need of care and protection"
    ),
}


class MediLexRetriever:
    """
    Dense retriever backed by ChromaDB + MiniLM embeddings.

    Instantiate once (via get_retriever()) and reuse — the embedding
    model and ChromaDB client are heavyweight and should not be
    recreated per request.
    """

    def __init__(self) -> None:
        print("Initialising retriever (BNS/BNSS/BSA edition)...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        self.collection = self.client.get_collection(config.COLLECTION_NAME)
        print(f"Connected to ChromaDB — {self.collection.count()} chunks loaded.")
    # Inside class MediLexRetriever in rag/retriever.py:

    def warm_bm25_index(self) -> None:
        """Public startup method to pre-build the BM25 index on server boot."""
        self._ensure_bm25_index()
    def retrieve(
        self,
        case_type: str,
        symptoms: str,
        patient_age: int,
        n_results: int = config.TOP_K,
    ) -> dict:
        """
        Main retrieval method.  Called from main.py's /api/analyze.

        Returns
        -------
        dict with keys:
            query_used       : str   — the expanded query that was embedded
            is_minor          : bool  — True if patient_age < 18
            chunks            : list  — [{text, law, section, relevance_score}, ...]
            combined_context  : str   — formatted context string for the LLM prompt
            laws_retrieved    : list  — deduplicated law names found
        """
        is_minor = patient_age < 18
        query = self._build_query(case_type, symptoms, is_minor)
        safe_n = min(n_results, self.collection.count())

        results = self.collection.query(query_texts=[query], n_results=safe_n)

        chunks = []
        for i in range(len(results["documents"][0])):
            score = 1 - results["distances"][0][i]
            meta = results["metadatas"][0][i]
            chunks.append({
                "text": results["documents"][0][i],
                "law": meta["law"],
                "section": meta.get("section"),
                "relevance_score": round(score, 3),
                "chunk_id": results["ids"][0][i],
            })

        chunks.sort(key=lambda x: x["relevance_score"], reverse=True)

        return {
            "query_used": query,
            "is_minor": is_minor,
            "chunks": chunks,
            "combined_context": self._combine(chunks),
            "laws_retrieved": list({c["law"] for c in chunks}),
        }

    def retrieve_raw(self, query: str, n_results: int = config.TOP_K) -> dict:
        """
        Direct retrieval by raw query string — used by benchmark scripts.
        No case_type expansion; tests pure semantic retrieval quality.
        """
        safe_n = min(n_results, self.collection.count())
        results = self.collection.query(query_texts=[query], n_results=safe_n)

        chunks = []
        for i in range(len(results["documents"][0])):
            score = 1 - results["distances"][0][i]
            meta = results["metadatas"][0][i]
            chunks.append({
                "text": results["documents"][0][i],
                "law": meta["law"],
                "section": meta.get("section"),
                "relevance_score": round(score, 3),
                "chunk_id": results["ids"][0][i],
            })

        chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        return {
            "query": query,
            "chunks": chunks,
            "laws_retrieved": list({c["law"] for c in chunks}),
        }

    # ── Hybrid retrieval (BM25 + dense, fused via RRF) ──────────────────────────
    #
    # Additive, comparison-safe path.  retrieve()/retrieve_raw() above are left
    # completely untouched — including internally — so they remain a stable
    # dense-only baseline to benchmark this against.  See _hybrid_search() for
    # the fusion logic and _ensure_bm25_index() for why the sparse index is
    # built lazily instead of in __init__.

    def retrieve_hybrid(
        self,
        case_type: str,
        symptoms: str,
        patient_age: int,
        n_results: int = config.TOP_K,
        candidate_pool_size: int = 20,
        rrf_k: int = 60,
    ) -> dict:
        """
        Hybrid counterpart to retrieve().  Same signature and return shape,
        so main.py can swap to this later with a one-line change once it's
        been benchmarked.

        candidate_pool_size : how many candidates EACH ranker (dense, BM25)
            contributes before fusion.  Deliberately wider than n_results —
            a chunk ranked low by one method but high by the other should
            still get a chance to be fused in.
        rrf_k : RRF damping constant.  60 is the standard value from the
            original Cormack et al. (2009) paper; higher values flatten
            the influence of rank differences.

        Note: relevance_score here is an RRF score, not a 0–1 cosine
        similarity like retrieve() returns — the two are not directly
        comparable.  Relevant when confidence gating (roadmap step 7)
        is wired up.
        """
        is_minor = patient_age < 18
        query = self._build_query(case_type, symptoms, is_minor)
        chunks, laws_retrieved = self._hybrid_search(query, n_results, candidate_pool_size, rrf_k)

        return {
            "query_used": query,
            "is_minor": is_minor,
            "chunks": chunks,
            "combined_context": self._combine(chunks),
            "laws_retrieved": laws_retrieved,
        }

    def retrieve_raw_hybrid(
        self,
        query: str,
        n_results: int = config.TOP_K,
        candidate_pool_size: int = 20,
        rrf_k: int = 60,
    ) -> dict:
        """
        Hybrid counterpart to retrieve_raw() — direct query string, no
        case_type expansion.  Use this in benchmark.py to compare against
        retrieve_raw() on identical queries.
        """
        chunks, laws_retrieved = self._hybrid_search(query, n_results, candidate_pool_size, rrf_k)
        return {
            "query": query,
            "chunks": chunks,
            "laws_retrieved": laws_retrieved,
        }

    # ── Reranking (cross-encoder over the hybrid candidate set) ─────────────────
    #
    # Layered strictly on top of _hybrid_search(), not a replacement for it:
    # pull a wide candidate pool (default 20) via RRF fusion, then have a
    # cross-encoder re-score each (query, chunk_text) pair directly — a
    # cross-encoder sees the query and chunk together and can judge relevance
    # far more precisely than the bi-encoder cosine similarity or BM25 scores
    # that produced the candidate pool, at the cost of being too slow to run
    # over the full corpus (hence: rerank a shortlist, don't replace retrieval).
    #
    # rerank=False on both methods below returns the plain hybrid top-N
    # unchanged — this is the toggle the ablation study needs to isolate
    # the reranker's effect from the fusion's effect.

    def retrieve_hybrid_reranked(
        self,
        case_type: str,
        symptoms: str,
        patient_age: int,
        n_results: int = config.TOP_K,
        rerank_pool_size: int = 20,
        rrf_k: int = 60,
        rerank: bool = True,
    ) -> dict:
        """Hybrid retrieval + cross-encoder rerank. Same return shape as
        retrieve_hybrid(). Set rerank=False to get the pre-rerank hybrid
        top-N instead — used by the ablation study to isolate this stage."""
        is_minor = patient_age < 18
        query = self._build_query(case_type, symptoms, is_minor)
        chunks, laws_retrieved = self._hybrid_then_rerank(
            query, n_results, rerank_pool_size, rrf_k, rerank
        )

        return {
            "query_used": query,
            "is_minor": is_minor,
            "chunks": chunks,
            "combined_context": self._combine(chunks),
            "laws_retrieved": laws_retrieved,
        }

    def retrieve_raw_hybrid_reranked(
        self,
        query: str,
        n_results: int = config.TOP_K,
        rerank_pool_size: int = 20,
        rrf_k: int = 60,
        rerank: bool = True,
    ) -> dict:
        """Raw-query counterpart to retrieve_hybrid_reranked() — used by the
        evaluation harness to compare against retrieve_raw()/retrieve_raw_hybrid()
        on identical queries."""
        chunks, laws_retrieved = self._hybrid_then_rerank(
            query, n_results, rerank_pool_size, rrf_k, rerank
        )
        return {
            "query": query,
            "chunks": chunks,
            "laws_retrieved": laws_retrieved,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_query(self, case_type: str, symptoms: str, is_minor: bool) -> str:
        """
        Expands a short case_type + symptoms into a richer query string
        by appending domain-specific keywords.  This is a retrieval
        heuristic, not a replacement for hybrid search (planned).
        """
        extra = CASE_TYPE_KEYWORDS.get(case_type, case_type)
        age_tag = "minor child" if is_minor else "adult"
        return f"{case_type} {symptoms} {age_tag} {extra} India legal medical obligation"

    def _combine(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a single context string for the LLM."""
        parts = []
        for c in chunks:
            citation = c["law"]
            if c.get("section"):
                citation += f" Section {c['section']}"
            parts.append(f"[{citation}]\n{c['text']}")
        return "\n\n---\n\n".join(parts)

    def _ensure_bm25_index(self) -> None:
        """
        Build the BM25 sparse index once, on first hybrid call, and cache it
        on the instance.  Deliberately NOT built in __init__: the dense path
        (retrieve/retrieve_raw) doesn't need it, so callers who never touch
        hybrid retrieval pay zero extra startup cost.  Corpus is pulled via
        collection.get() — the same 124 chunks already living in ChromaDB,
        so there is no second source of truth to keep in sync.
        """
        if getattr(self, "_bm25", None) is not None:
            return

        print("Building BM25 sparse index (first hybrid query)...")
        corpus = self.collection.get(include=["documents", "metadatas"])
        self._bm25_corpus_ids = corpus["ids"]
        self._bm25_corpus_texts = corpus["documents"]
        self._bm25_corpus_meta = corpus["metadatas"]

        tokenized_corpus = [self._tokenize(t) for t in self._bm25_corpus_texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index built — {len(self._bm25_corpus_ids)} chunks indexed.")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase alphanumeric word split. Simple on purpose — BM25 over
        a 124-chunk legal corpus doesn't need stemming/stopword removal to
        be useful, and a simpler tokenizer is easier to reason about."""
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _hybrid_search(
        self,
        query: str,
        n_results: int,
        candidate_pool_size: int,
        rrf_k: int,
    ) -> tuple[list[dict], list[str]]:
        """
        Runs dense + BM25 candidate retrieval independently, fuses via
        Reciprocal Rank Fusion, and returns the top n_results chunks.

        Kept fully separate from retrieve()/retrieve_raw()'s internals
        (no shared helper beyond the pure, side-effect-free _build_query
        and _combine) so the dense-only baseline used for comparison can't
        be affected by changes here.
        """
        self._ensure_bm25_index()
        pool = min(candidate_pool_size, self.collection.count())

        # ── Dense candidates ─────────────────────────────────────────────
        dense_results = self.collection.query(query_texts=[query], n_results=pool)
        dense_ids = dense_results["ids"][0]

        chunk_info: dict[str, dict] = {}
        for i, cid in enumerate(dense_ids):
            meta = dense_results["metadatas"][0][i]
            chunk_info[cid] = {
                "text": dense_results["documents"][0][i],
                "law": meta["law"],
                "section": meta.get("section"),
            }

        # ── Sparse (BM25) candidates ─────────────────────────────────────
        bm25_scores = self._bm25.get_scores(self._tokenize(query))
        ranked_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:pool]
        bm25_ids = [self._bm25_corpus_ids[i] for i in ranked_indices]

        for i in ranked_indices:
            cid = self._bm25_corpus_ids[i]
            if cid not in chunk_info:  # don't overwrite dense metadata if already present
                chunk_info[cid] = {
                    "text": self._bm25_corpus_texts[i],
                    "law": self._bm25_corpus_meta[i]["law"],
                    "section": self._bm25_corpus_meta[i].get("section"),
                }

        # ── Reciprocal Rank Fusion ───────────────────────────────────────
        fused_scores: dict[str, float] = {}
        for rank, cid in enumerate(dense_ids):
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, cid in enumerate(bm25_ids):
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

        top_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:n_results]

        chunks = [
            {
                "text": chunk_info[cid]["text"],
                "law": chunk_info[cid]["law"],
                "section": chunk_info[cid]["section"],
                "relevance_score": round(fused_scores[cid], 4),
                "chunk_id": cid,
            }
            for cid in top_ids
        ]
        laws_retrieved = list({c["law"] for c in chunks})
        return chunks, laws_retrieved

    def _ensure_cross_encoder(self) -> None:
        """
        Lazy-load the cross-encoder, same pattern as _ensure_bm25_index():
        callers who never touch reranking (including retrieve_hybrid()) pay
        zero extra startup/memory cost.
        """
        if getattr(self, "_cross_encoder", None) is not None:
            return
        print(f"Loading cross-encoder reranker ({config.RERANKER_MODEL})...")
        self._cross_encoder = CrossEncoder(config.RERANKER_MODEL)
        print("Cross-encoder loaded.")

    def _rerank(self, query: str, candidates: list[dict], n_results: int) -> list[dict]:
        """
        Re-score each candidate against the query with the cross-encoder and
        keep the top n_results. Candidates already carry chunk_id/text/law/
        section from _hybrid_search(); this only re-sorts and truncates.
        """
        if not candidates:
            return candidates

        self._ensure_cross_encoder()
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._cross_encoder.predict(pairs)

        reranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)

        out = []
        for chunk, score in reranked[:n_results]:
            chunk = dict(chunk)  # copy — don't mutate the candidate the caller may reuse
            chunk["rerank_score"] = round(float(score), 4)
            out.append(chunk)
        return out

    def _hybrid_then_rerank(
        self,
        query: str,
        n_results: int,
        rerank_pool_size: int,
        rrf_k: int,
        rerank: bool,
    ) -> tuple[list[dict], list[str]]:
        """
        Pulls a wide RRF-fused candidate pool, then optionally reranks it
        down to n_results.  rerank=False returns the top n_results of the
        pool unranked by the cross-encoder — i.e. identical to what
        retrieve_hybrid() would have returned.
        """
        candidates, _ = self._hybrid_search(query, rerank_pool_size, rerank_pool_size, rrf_k)

        if rerank:
            chunks = self._rerank(query, candidates, n_results)
        else:
            chunks = candidates[:n_results]

        laws_retrieved = list({c["law"] for c in chunks})
        return chunks, laws_retrieved

    def retrieve_raw_bm25(self, query: str, n_results: int = config.TOP_K) -> dict:
        """
        Pure sparse (BM25-only) retrieval — no dense, no fusion.

        Not used by any live endpoint.  Exists purely as a diagnostic arm
        for the evaluation harness: without it, an improvement from
        retrieve_raw_hybrid() is ambiguous — it could be BM25's signal
        doing the work, or the RRF fusion itself.  Isolating BM25 alone
        answers that question.
        """
        self._ensure_bm25_index()
        scores = self._bm25.get_scores(self._tokenize(query))
        safe_n = min(n_results, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:safe_n]

        chunks = [
            {
                "text": self._bm25_corpus_texts[i],
                "law": self._bm25_corpus_meta[i]["law"],
                "section": self._bm25_corpus_meta[i].get("section"),
                "relevance_score": round(float(scores[i]), 4),
                "chunk_id": self._bm25_corpus_ids[i],
            }
            for i in top_indices
        ]
        return {
            "query": query,
            "chunks": chunks,
            "laws_retrieved": list({c["law"] for c in chunks}),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
# Why a module-level singleton?  The SentenceTransformer model load takes
# ~2–5 seconds.  We want to pay that cost once at startup (via get_retriever()
# in main.py's startup hook), not on every request.

_instance: MediLexRetriever | None = None


def get_retriever() -> MediLexRetriever:
    global _instance
    if _instance is None:
        _instance = MediLexRetriever()
    return _instance
