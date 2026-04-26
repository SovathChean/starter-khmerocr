import io

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


async def load_image_from_upload(file: UploadFile) -> Image.Image:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="File is not a valid image") from exc
