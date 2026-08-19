"""Lightweight lexical (BM25) retrieval over evidence chunks.

No large ML dependencies. Chunks are scored with a small in-memory BM25
implementation over their tokenized text.
"""

import math
import re
import statistics


_TOKEN_RE = re.compile(r"[a-z0-9_]+")

# Minimal stopwords so common English words don't create false matches.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "of", "to", "in", "on", "is",
    "are", "be", "was", "were", "with", "as", "at", "by", "that", "this",
    "what", "how", "does", "should", "why", "from", "it", "its",
}


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric/underscore tokens, dropping stopwords."""
    return [
        t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS
    ]


def _df_docs(documents: list[list[str]]) -> dict[str, int]:
    """Document frequency for each term across the corpus."""
    df: dict[str, int] = {}
    for doc in documents:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    return df


def _bm25_scores(query_terms, documents, avgdl, n_docs, k1=1.5, b=0.75):
    df = _df_docs(documents)
    idf = {}
    for term in df:
        idf[term] = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
    scores = []
    for doc in documents:
        dl = len(doc)
        tf = {}
        for term in doc:
            tf[term] = tf.get(term, 0) + 1
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            f = tf[term]
            denom = f + k1 * (1 - b + b * dl / avgdl) if avgdl else f + k1
            score += idf.get(term, 0) * (f * (k1 + 1)) / denom
        scores.append(score)
    return scores


def rank_evidence(chunks: list[dict], query: str, top_k: int = 5) -> list[dict]:
    """Rank evidence chunks against a query and return the top_k.

    Args:
        chunks: list of dicts with keys evidence_id, file_path, line_start,
            line_end, text.
        query: the investigation query.
        top_k: number of results to return.

    Returns:
        Ranked list of dicts with evidence_id, file_path, line_start, line_end,
        text and score (score 0.0 means no match).
    """
    docs = [tokenize(c["text"]) for c in chunks]
    n_docs = len(docs)
    if n_docs == 0:
        return []
    avgdl = statistics.mean(len(d) for d in docs) if docs else 0.0
    q_terms = tokenize(query)
    scores = _bm25_scores(q_terms, docs, avgdl, n_docs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    results = []
    for chunk, score in ranked[:top_k]:
        if score <= 0:
            break
        results.append(
            {
                "evidence_id": chunk["evidence_id"],
                "file_path": chunk["file_path"],
                "line_start": chunk["line_start"],
                "line_end": chunk["line_end"],
                "text": chunk["text"],
                "score": round(score, 6),
            }
        )
    return results


def lexical_search(
    corpus: list[dict], query: str, top_k: int = 5
) -> list[dict]:
    """Public API: rank evidence chunks by lexical relevance."""
    return rank_evidence(corpus, query, top_k)
