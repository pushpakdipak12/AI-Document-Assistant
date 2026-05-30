import chromadb


client = chromadb.PersistentClient(
    path="data/vector_db"
)

collection = client.get_collection(
    name="documents"
)

results = collection.query(
    query_texts=[
        "What is BERT model?"
    ],
    n_results=3
)

print(results)