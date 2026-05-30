from pydantic import BaseModel


class UploadResponse(BaseModel):
    filename: str
    content_type: str
    total_chunks: int
    message: str