import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    N = len(docs)
    if N == 0:
        return np.array([], dtype=float)

    unique_query_terms = list(dict.fromkeys(query_tokens))

    doc_lens = np.array([len(doc) for doc in docs], dtype=float)
    avgdl = doc_lens.mean() if N > 0 else 0.0

    doc_term_counts = [Counter(doc) for doc in docs]

    df = {}

    for term in unique_query_terms:
        df[term] = sum(1 for tc in doc_term_counts if term in tc)

    idf = {}

    for term in unique_query_terms:
        d = df[term]
        if d == 0:
            idf[term] = 0.0
        else:
            idf[term] = math.log((N - d + 0.5) / (d + 0.5) + 1)

    scores = np.zeros(N, dtype=float)

    # (1 - b + b * |D| / avgd1)
    if avgdl > 0:
        len_norm = (1 - b + b * (doc_lens / avgdl))
    else:
        len_norm = np.ones(N, dtype=float)

    for i, tc in enumerate(doc_term_counts):
        doc_score = 0.0
        denom_base = k1 * len_norm[i]

        for term in unique_query_terms:
            term_idf = idf[term]
            if term_idf == 0.0:
                continue
            tf = tc.get(term, 0)
            if tf == 0:
                continue
            number = term_idf * tf * (k1 + 1)
            denom = tf + denom_base
            doc_score += number / denom
        scores[i] = doc_score

    return scores
