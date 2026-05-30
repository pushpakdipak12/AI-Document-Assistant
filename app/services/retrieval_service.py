from app.services.vector_db_service import collection


def retrieve_relevant_chunks(
    query: str,
    n_results: int = 5
) -> list[dict]:

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    formatted_results = []

    for i, doc in enumerate(documents):

        metadata = (
            metadatas[i]
            if i < len(metadatas)
            else {}
        )

        distance = (
            distances[i]
            if i < len(distances)
            else None
        )

        formatted_results.append(
            {
                "chunk": doc,
                "source": metadata.get("source"),
                "distance": round(distance, 4)
                if distance is not None
                else None
            }
        )

    return formatted_results