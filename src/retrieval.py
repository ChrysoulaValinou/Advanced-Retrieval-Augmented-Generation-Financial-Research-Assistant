"""
src/retrieval.py
================
Task 2 – Metadata Filtering in Retrieval

Implements three retrieval strategies, all supporting optional
metadata pre-filtering via ChromaDB's `where` clause:

  2.1  Dense retrieval    – vector similarity search (OpenAI embeddings + ChromaDB)
  2.2  Sparse retrieval   – BM25 keyword search (rank-bm25)
  2.3  Hybrid retrieval   – Reciprocal Rank Fusion (RRF) of dense + sparse

RRF formula (k=60):
    RRF_score(d) = Σ  1 / (k + rank(d, r))
                  r ∈ {dense_list, sparse_list}

Public API
----------
    retriever = Retriever(vectorstore, documents)

    # Dense only
    results = retriever.dense(query, k=5)

    # Sparse (BM25) only
    results = retriever.sparse(query, k=5)

    # Hybrid RRF
    results = retriever.hybrid(query, k=5)

    # Any of the above with category pre-filter
    results = retriever.dense(query, k=5, filter_category="analyst_report")

All methods return a list of RetrievedDoc(content, metadata, score).
"""

from __future__ import annotations

import logging
import math
import re
import string
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import time

# ── LangChain / ChromaDB (install: pip install langchain langchain-openai chromadb rank-bm25) ──
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# BM25  (install: pip install rank-bm25)
from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)

# ── RRF constant ─────────────────────────────────────────────────────────────────
RRF_K = 60


# ── Result container ──────────────────────────────────────────────────────────────

@dataclass
class RetrievedDoc:
    """
    A single retrieved chunk with its content, metadata, and retrieval score.

    Attributes
    ----------
    content  : the chunk text
    metadata : dict with document_category, source, author, etc.
    score    : relevance score (cosine similarity, BM25, or RRF — higher = better)
    """
    content:  str
    metadata: dict = field(default_factory=dict)
    score:    float = 0.0

    def __repr__(self) -> str:
        src   = self.metadata.get("source", "?")
        cat   = self.metadata.get("document_category", "?")
        chunk = self.metadata.get("chunk_index", "?")
        return (f"RetrievedDoc(score={self.score:.4f}, source={src!r}, "
                f"category={cat!r}, chunk={chunk}, "
                f"preview={self.content[:80]!r})")


# ── Tokeniser shared by BM25 and query preprocessing ─────────────────────────────

def _tokenise(text: str) -> list[str]:
    """
    Lowercase, remove punctuation, split on whitespace.
    Simple but consistent — the same function is used at index time
    and at query time so the vocabulary always matches.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [tok for tok in text.split() if tok]


# ── Main Retriever class ──────────────────────────────────────────────────────────

class Retriever:
    """
    Unified retriever supporting dense, sparse, and hybrid strategies.

    Parameters
    ----------
    vectorstore : Chroma
        The persisted ChromaDB vector store built by ingestion.py.
    documents   : list[Document]
        The same LangChain Document objects that were ingested.
        Required to build the in-memory BM25 index.
    """

    def __init__(self, vectorstore: Chroma, documents: list[Document]) -> None:
        self._vs   = vectorstore
        self._docs = documents

        # Build BM25 index over all documents (in-memory, fast)
        self._build_bm25_index(documents)
        log.info("Retriever ready — %d documents in BM25 index", len(documents))

    # ── BM25 index construction ───────────────────────────────────────────────────

    def _build_bm25_index(self, documents: list[Document]) -> None:
        """
        Tokenise every document and build a BM25Okapi index.
        We also store (content, metadata) per document for fast retrieval.
        """
        self._corpus_tokens: list[list[str]] = []
        self._corpus_docs:   list[tuple[str, dict]] = []

        for doc in documents:
            tokens = _tokenise(doc.page_content)
            self._corpus_tokens.append(tokens)
            self._corpus_docs.append((doc.page_content, doc.metadata))

        self._bm25 = BM25Okapi(self._corpus_tokens)

    # ── 2.1  Dense retrieval ──────────────────────────────────────────────────────

    def dense(
        self,
        query:           str,
        k:               int            = 5,
        filter_category: Optional[str]  = None,
    ) -> list[RetrievedDoc]:
        """
        Vector similarity search using Google Gemini embeddings + ChromaDB.

        Parameters
        ----------
        query           : natural language question
        k               : number of results to return
        filter_category : if provided, restrict search to chunks whose
                          document_category == filter_category
                          (uses ChromaDB `where` clause — exact match)

        Returns
        -------
        List of RetrievedDoc sorted by cosine similarity (descending).
        """
        # Build the ChromaDB `where` filter dict
        where_filter: Optional[dict] = None
        if filter_category:
            where_filter = {"document_category": {"$eq": filter_category}}
            log.info("Dense retrieval: filter_category=%r", filter_category)

        # similarity_search_with_relevance_scores returns (Document, score) tuples
        # where score ∈ [0, 1] (1 = most similar)
        raw_results = self._vs.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=where_filter,
        )

        results = [
            RetrievedDoc(
                content=doc.page_content,
                metadata=doc.metadata,
                score=float(score),
            )
            for doc, score in raw_results
        ]

        log.info("Dense   → %d results for %r (filter=%r)", len(results), query[:60], filter_category)
        time.sleep(2)
        return results

    # ── 2.2  Sparse retrieval (BM25) ──────────────────────────────────────────────

    def sparse(
        self,
        query:           str,
        k:               int           = 5,
        filter_category: Optional[str] = None,
    ) -> list[RetrievedDoc]:
        """
        BM25 keyword retrieval over the full corpus.

        BM25 (Best Match 25) is a probabilistic ranking function that scores
        documents based on term frequency (TF) and inverse document frequency
        (IDF), with length normalisation.  It excels at exact keyword matches
        that vector search sometimes misses (e.g. ticker symbols, specific
        numbers, proper nouns).

        Parameters
        ----------
        query           : natural language question
        k               : number of results to return
        filter_category : if provided, only documents whose metadata
                          document_category == filter_category are eligible.
                          Filtering is done *after* BM25 scoring on the
                          full corpus (pre-filtering BM25 would require
                          rebuilding the index — see note below).

        Note
        ----
        BM25 is an in-memory index.  Category filtering is applied as a
        post-filter on BM25 scores, then the top-k are returned.
        The BM25 scores themselves are computed on the full corpus so IDF
        values remain accurate (not biased by category size).
        """
        query_tokens = _tokenise(query)
        scores       = self._bm25.get_scores(query_tokens)  # np.ndarray, len = corpus size

        # Pair each score with its document index, then sort descending
        scored_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results: list[RetrievedDoc] = []
        for idx in scored_indices:
            content, metadata = self._corpus_docs[idx]

            # Apply category post-filter
            if filter_category and metadata.get("document_category") != filter_category:
                continue

            results.append(RetrievedDoc(
                content=content,
                metadata=metadata,
                score=float(scores[idx]),
            ))

            if len(results) == k:
                break

        log.info("Sparse  → %d results for %r (filter=%r)", len(results), query[:60], filter_category)
        return results

    # ── 2.3  Hybrid retrieval with RRF ────────────────────────────────────────────

    def hybrid(
        self,
        query:           str,
        k:               int           = 5,
        filter_category: Optional[str] = None,
        dense_k:         int           = 20,
        sparse_k:        int           = 20,
        rrf_k:           int           = RRF_K,
    ) -> list[RetrievedDoc]:
        """
        Hybrid retrieval combining dense + sparse via Reciprocal Rank Fusion.

        Algorithm
        ---------
        1. Run dense retrieval  → ranked list R_dense  (top dense_k)
        2. Run sparse retrieval → ranked list R_sparse (top sparse_k)
        3. For every unique document d that appears in either list:

               RRF_score(d) = Σ        1
                              r        ─────────────
                                       k + rank(d, r)

           where rank(d, r) is 1-based position in list r.
           If d is absent from a list, it contributes 0 for that list.

        4. Re-rank all documents by RRF score (descending).
        5. Return the top-k.

        Why RRF?
        --------
        - Dense search finds semantically similar passages even with
          different wording.
        - Sparse search finds exact keyword/number matches.
        - RRF fuses the two lists without needing to normalise their
          scores (which live on incompatible scales: cosine vs. BM25).

        Parameters
        ----------
        query           : natural language question
        k               : final number of results to return
        filter_category : passed through to both dense and sparse retrievers
        dense_k         : how many candidates to fetch from dense (default 20)
        sparse_k        : how many candidates to fetch from sparse (default 20)
        rrf_k           : RRF smoothing constant (default 60 per the assignment)
        """
        # Step 1 & 2: get candidate lists (use larger pools for better fusion)
        dense_results  = self.dense( query, k=dense_k,  filter_category=filter_category)
        sparse_results = self.sparse(query, k=sparse_k, filter_category=filter_category)

        # Step 3: compute RRF scores
        # Use (source, chunk_index) as a unique document key
        rrf_scores: dict[tuple, float]        = defaultdict(float)
        doc_store:  dict[tuple, RetrievedDoc] = {}

        def _doc_key(doc: RetrievedDoc) -> tuple:
            m = doc.metadata
            return (m.get("source", ""), m.get("chunk_index", -1))

        for rank_0based, doc in enumerate(dense_results):
            key = _doc_key(doc)
            rrf_scores[key] += 1.0 / (rrf_k + rank_0based + 1)   # rank is 1-based
            doc_store[key]   = doc

        for rank_0based, doc in enumerate(sparse_results):
            key = _doc_key(doc)
            rrf_scores[key] += 1.0 / (rrf_k + rank_0based + 1)
            if key not in doc_store:
                doc_store[key] = doc

        # Step 4: sort by combined RRF score
        ranked_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

        # Step 5: build final result list
        results: list[RetrievedDoc] = []
        for key in ranked_keys[:k]:
            doc          = doc_store[key]
            final_doc    = RetrievedDoc(
                content=doc.content,
                metadata=doc.metadata,
                score=rrf_scores[key],   # RRF score replaces original score
            )
            results.append(final_doc)

        log.info("Hybrid  → %d results for %r (filter=%r)", len(results), query[:60], filter_category)
        return results

    # ── Convenience: format results for LLM prompt ────────────────────────────────

    def format_context(self, results: list[RetrievedDoc]) -> str:
        """
        Format a list of RetrievedDoc objects into a context string
        suitable for injection into an LLM prompt.

        Each chunk is labelled with its source and category so the LLM
        can attribute its answers correctly.
        """
        if not results:
            return "No relevant context found."

        parts: list[str] = []
        for i, doc in enumerate(results, start=1):
            src  = doc.metadata.get("source", "unknown")
            cat  = doc.metadata.get("document_category", "unknown")
            auth = doc.metadata.get("author", "unknown")
            date = doc.metadata.get("creation_date", "unknown")
            header = (f"[Context {i} | source: {src} | "
                      f"category: {cat} | author: {auth} | date: {date}]")
            parts.append(f"{header}\n{doc.content}")

        return "\n\n---\n\n".join(parts)


# ── Standalone test (run: python -m src.retrieval) ───────────────────────────────

if __name__ == "__main__":
    """
    Quick smoke-test that exercises all three retrieval modes.
    Requires a built ChromaDB index (run ingestion.py first).
    """
    import os
    from pathlib import Path
    from src.ingestion import build_vectorstore, build_documents, DATA_DIR, CHROMA_DIR

    logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")

    print("Loading vector store …")
    vs   = build_vectorstore()
    docs = build_documents(DATA_DIR)
    retriever = Retriever(vs, docs)

    TEST_QUERIES = [
        ("What is TechVision's total revenue for FY2024?",           None),
        ("What is Germany's GDP growth rate?",                        "macroeconomic_data"),
        ("What is the price target for TVC?",                        "analyst_report"),
        ("What are the quarterly revenues of the AI platform?",      "stock_and_financial_data"),
        ("Who is the CEO of TechVision and what did he say?",        "financial_news"),
    ]

    for query, category in TEST_QUERIES:
        print(f"\n{'='*70}")
        print(f"QUERY     : {query}")
        print(f"FILTER    : {category or 'none'}")

        print("\n── Dense ──")
        for r in retriever.dense(query, k=3, filter_category=category):
            print(f"  {r}")

        print("\n── Sparse (BM25) ──")
        for r in retriever.sparse(query, k=3, filter_category=category):
            print(f"  {r}")

        print("\n── Hybrid (RRF) ──")
        for r in retriever.hybrid(query, k=3, filter_category=category):
            print(f"  {r}")
