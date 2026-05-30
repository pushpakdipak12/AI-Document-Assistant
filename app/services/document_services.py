from fastapi import UploadFile

from app.utils.file_handler import save_uploaded_file
from app.utils.pdf_processor import extract_text_from_pdf
from app.utils.text_cleaned import clean_text

from app.services.chunking_services import chunk_text
from app.services.embedding_service import generate_embeddings
from app.services.vector_db_service import store_embeddings


async def upload_document(
    file: UploadFile
) -> dict:
    """
    Handle document upload and processing.
    """

    saved_path = await save_uploaded_file(file)

    extracted_text = extract_text_from_pdf(saved_path)

    cleaned_text = clean_text(extracted_text)

    chunks = chunk_text(cleaned_text)

    embeddings = generate_embeddings(chunks)
    

    print(f"Generated embeddings: {len(embeddings)}")

    
    store_embeddings(
        chunks=chunks,
        embeddings=embeddings,
        filename=file.filename
    )
    
    print(f"Total chunks created: {len(chunks)}")

    print("Embeddings stored in ChromaDB successfully")

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "saved_path": saved_path,
        "total_chunks": len(chunks)
    }