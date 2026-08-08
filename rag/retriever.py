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

import chromadb
from sentence_transformers import SentenceTransformer

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
            })

        chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        return {
            "query": query,
            "chunks": chunks,
            "laws_retrieved": list({c["law"] for c in chunks}),
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
