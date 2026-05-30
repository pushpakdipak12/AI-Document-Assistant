# ❗ We still keep memory for fallback/debugging
all_document_chunks = []


def add_documents(chunks: list):
    global all_document_chunks

    if not chunks:
        print("⚠️ No chunks received to store")
        return

    all_document_chunks.extend(chunks)

    print(f"✅ Added {len(chunks)} chunks to memory")


def get_documents():
    return all_document_chunks


def clear_documents():
    global all_document_chunks
    all_document_chunks = []