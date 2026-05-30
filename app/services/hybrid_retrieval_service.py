import re

from rank_bm25 import BM25Okapi

from app.services.vector_db_service import (
    get_all_chunks,
    search_embeddings
)


def _tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25.
    """

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


def hybrid_search(
    query: str,
    top_k: int = 10
) -> list[dict]:
    """
    Hybrid search:
    - Vector search from ChromaDB
    - BM25 keyword search from stored chunks
    """

    vector_results = search_embeddings(
        query=query,
        top_k=top_k
    )

    corpus = get_all_chunks()

    if not corpus:
        return vector_results

    tokenized_corpus = [
        _tokenize(chunk)
        for chunk in corpus
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = _tokenize(query)

    bm25_scores = bm25.get_scores(query_tokens)

    combined_results = {}

    for item in vector_results:

        chunk = item["chunk"]

        distance = item.get("distance")

        vector_score = (
            1 / (1 + distance)
            if distance is not None
            else 0.0
        )

        combined_results[chunk] = {
            "chunk": chunk,
            "source": item.get("source", "unknown"),
            "distance": distance,
            "vector_score": round(float(vector_score), 4),
            "bm25_score": 0.0,
            "combined_score": round(float(vector_score), 4)
        }

    ranked_bm25 = sorted(
        enumerate(bm25_scores),
        key=lambda item: item[1],
        reverse=True
    )[:top_k]

    for index, score in ranked_bm25:

        chunk = corpus[index]

        if chunk in combined_results:

            combined_results[chunk]["bm25_score"] = round(float(score), 4)

            combined_results[chunk]["combined_score"] = max(
                combined_results[chunk]["combined_score"],
                float(score)
            )

        else:

            combined_results[chunk] = {
                "chunk": chunk,
                "source": "bm25",
                "distance": None,
                "vector_score": 0.0,
                "bm25_score": round(float(score), 4),
                "combined_score": round(float(score), 4)
            }

    final_results = sorted(
        combined_results.values(),
        key=lambda item: item["combined_score"],
        reverse=True
    )

    return final_results[:top_k]