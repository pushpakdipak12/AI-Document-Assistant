from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException
)

from app.models.response_models import UploadResponse
from app.services.upload_service import upload_document

router = APIRouter(
    prefix="/upload",
    tags=["Upload"]
)

ALLOWED_FILE_TYPES = ["application/pdf"]


@router.post("/", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF document.
    """
    if file.content_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    try:
        uploaded_file = await upload_document(file)

        return UploadResponse(
            filename=uploaded_file["filename"],
            content_type=uploaded_file["content_type"],
            total_chunks=uploaded_file["total_chunks"],
            message=uploaded_file["message"]
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"upload_failed: {e}"
        )