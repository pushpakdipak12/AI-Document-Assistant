from sentence_transformers import CrossEncoder


reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_chunks(
    query: str,
    retrieved_chunks: list[dict],
    top_k: int = 3
) -> list[dict]:
    """
    Rerank chunks using a cross-encoder.
    """
    if not retrieved_chunks:
        return []

    pairs = [(query, item["chunk"]) for item in retrieved_chunks]
    scores = reranker_model.predict(pairs)

    for item, score in zip(retrieved_chunks, scores):
        item["rerank_score"] = float(score)

    reranked_results = sorted(
        retrieved_chunks,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked_results[:top_k]