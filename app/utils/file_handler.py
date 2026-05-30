from pathlib import Path

from fastapi import UploadFile


UPLOAD_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "raw"


async def save_uploaded_file(file: UploadFile) -> str:
    """
    Save uploaded PDF to backend/data/raw.
    """

    UPLOAD_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    file_location = UPLOAD_DIRECTORY / file.filename

    content = await file.read()

    with open(file_location, "wb") as buffer:
        buffer.write(content)

    print(f"Saved file to: {file_location}")

    return str(file_location)