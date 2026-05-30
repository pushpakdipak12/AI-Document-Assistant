from fastapi import UploadFile

from app.utils.file_handler import save_uploaded_file
from app.utils.pdf_processor import extract_text_from_pdf
from app.utils.text_cleaner import clean_text

from app.services.chunking_service import chunk_text
from app.services.embedding_service import generate_embeddings
from app.services.vector_db_service import store_embeddings


async def upload_document(file: UploadFile) -> dict:
    """
    Upload pipeline:
    save PDF -> extract text -> clean text -> chunk -> embed -> store in ChromaDB.
    """

    saved_path = await save_uploaded_file(file)

    extracted_text = extract_text_from_pdf(saved_path)

    if not extracted_text or not extracted_text.strip():
        raise ValueError(
            "No readable text found in this PDF. Try a text-based PDF, not a scanned image PDF."
        )

    cleaned_text = clean_text(extracted_text)

    chunks = chunk_text(cleaned_text)

    if not chunks:
        raise ValueError("No valid chunks were created from the PDF.")

    print(f"Chunks created: {len(chunks)}")

    embeddings = generate_embeddings(chunks)

    if not embeddings:
        raise ValueError("Embeddings could not be generated.")

    print(f"Embeddings created: {len(embeddings)}")

    store_embeddings(
        chunks=chunks,
        embeddings=embeddings,
        filename=file.filename
    )

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "total_chunks": len(chunks),
        "message": "File uploaded and processed successfully"
    }