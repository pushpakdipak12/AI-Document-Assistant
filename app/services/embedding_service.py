from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks: list[str]) -> list[list[float]]:
    """
    Generate embeddings for document chunks.
    """
    if not chunks:
        return []

    return model.encode(chunks).tolist()


def generate_query_embedding(query: str) -> list[float]:
    """
    Generate embedding for a user query.
    """
    return model.encode([query])[0].tolist()