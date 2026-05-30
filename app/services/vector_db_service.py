from pathlib import Path
import uuid

import chromadb

from app.services.embedding_service import generate_query_embedding


CHROMA_PATH = Path(__file__).resolve().parents[2] / "data" / "chroma"

CHROMA_PATH.mkdir(
    parents=True,
    exist_ok=True
)

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_or_create_collection(
    name="rag_docs"
)


def store_embeddings(
    chunks: list[str],
    embeddings: list[list[float]],
    filename: str
):
    """
    Store chunks and embeddings in ChromaDB.
    """

    if not chunks or not embeddings:
        print("No chunks or embeddings to store.")
        return

    ids = [
        f"{filename}_{uuid.uuid4().hex}_{index}"
        for index in range(len(chunks))
    ]

    metadatas = [
        {
            "source": filename,
            "chunk_index": index
        }
        for index in range(len(chunks))
    ]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB at {CHROMA_PATH}")


def search_embeddings(
    query: str,
    top_k: int = 5
) -> list[dict]:
    """
    Search ChromaDB using vector similarity.
    """

    if collection.count() == 0:
        return []

    query_embedding = generate_query_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []

    for index, document in enumerate(documents):

        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        retrieved.append(
            {
                "chunk": document,
                "source": metadata.get("source", "unknown"),
                "distance": float(distance) if distance is not None else None
            }
        )

    return retrieved


def get_all_chunks() -> list[str]:
    """
    Get all chunks from ChromaDB for BM25.
    """

    if collection.count() == 0:
        return []

    results = collection.get(
        include=["documents"]
    )

    documents = results.get("documents", [])

    return [
        document
        for document in documents
        if document
    ]