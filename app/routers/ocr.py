import time

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.dependencies import (
    get_easyocr_service,
    get_kiri_service,
    get_paddleocr_service,
    get_tesseract_service,
)
from app.images import load_image_from_upload
from app.schemas import OCRResponse
from app.services.base import OCRService

router = APIRouter(prefix="/ocr", tags=["ocr"])


async def _run(service: OCRService, file: UploadFile, **kwargs) -> OCRResponse:
    image = await load_image_from_upload(file)
    start = time.perf_counter()
    try:
        result = service.recognize(image, **kwargs)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{service.name} failed: {exc}") from exc
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return OCRResponse(
        engine=service.name,
        languages=result.languages,
        text=result.text,
        elapsed_ms=elapsed_ms,
        note=result.note,
    )


@router.post("/tesseract", response_model=OCRResponse)
async def ocr_tesseract(
    file: UploadFile = File(...),
    lang: str | None = Query(None, description="Tesseract lang code, e.g. 'eng', 'khm', 'eng+khm'"),
    service: OCRService = Depends(get_tesseract_service),
) -> OCRResponse:
    return await _run(service, file, lang=lang)


@router.post("/easyocr", response_model=OCRResponse)
async def ocr_easyocr(
    file: UploadFile = File(...),
    lang: str | None = Query(None, description="Comma-separated EasyOCR lang codes, e.g. 'en' or 'en,fr'"),
    service: OCRService = Depends(get_easyocr_service),
) -> OCRResponse:
    return await _run(service, file, lang=lang)


@router.post("/paddleocr", response_model=OCRResponse)
async def ocr_paddleocr(
    file: UploadFile = File(...),
    lang: str | None = Query(None, description="PaddleOCR lang code, e.g. 'ch', 'en', 'korean'"),
    service: OCRService = Depends(get_paddleocr_service),
) -> OCRResponse:
    return await _run(service, file, lang=lang)


@router.post("/kiri", response_model=OCRResponse)
async def ocr_kiri(
    file: UploadFile = File(...),
    decode_method: str = Query(
        "accurate",
        description="Kiri decode method: 'fast' (CTC), 'accurate' (default), or 'beam' (best quality)",
    ),
    service: OCRService = Depends(get_kiri_service),
) -> OCRResponse:
    return await _run(service, file, decode_method=decode_method)
