import time

import pytesseract
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from PIL import Image

from app.dependencies import (
    get_easyocr_service,
    get_kiri_service,
    get_paddleocr_service,
    get_tesseract_service,
)
from app.images import load_image_from_upload
from app.khmer_id import extract_after_colon, locate_name_line, pad_bbox
from app.schemas import KhmerIDNameResponse, OCRResponse
from app.services.base import OCRService
from app.services.kiri import KiriOCRService

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


@router.post(
    "/khmer-id-name",
    response_model=KhmerIDNameResponse,
    summary="Extract the Khmer name from a Cambodian National ID card",
    description=(
        "Auto-locates the line containing 'នាម' via Tesseract image_to_data, splits it at the colon, "
        "then runs Tesseract on the full line and Kiri's single-line recognizer on the name-only crop. "
        "The two engines have complementary failure modes — Tesseract preserves structure, Kiri preserves "
        "the surname; the caller can compare or ensemble the two outputs."
    ),
)
async def ocr_khmer_id_name(
    file: UploadFile = File(...),
    kiri: KiriOCRService = Depends(get_kiri_service),
) -> KhmerIDNameResponse:
    image = await load_image_from_upload(file)

    location = locate_name_line(image)
    if location is None:
        raise HTTPException(status_code=404, detail="Could not locate the Khmer name line on this image")

    line_crop = image.crop(pad_bbox(location.line_bbox, image.size, pad=20))
    name_crop = image.crop(pad_bbox(location.name_bbox, image.size, pad=15))

    # Tesseract LSTM and Kiri both struggle when the line height is below ~80 px.
    # When the source ID image is small (e.g. a phone screenshot), upscale so the
    # recogniser sees ~150 px tall lines.
    def _upscale_if_small(crop: Image.Image, target_height: int = 150) -> Image.Image:
        if crop.height >= target_height:
            return crop
        factor = max(2, target_height // crop.height + 1)
        return crop.resize((crop.width * factor, crop.height * factor), Image.LANCZOS)

    line_for_ocr = _upscale_if_small(line_crop)
    name_for_ocr = _upscale_if_small(name_crop)

    # psm 7 (single text line) is best for long lines, but returns empty on short
    # 1-2 word crops. Fall back to psm 8 (single word) for the name crop.
    def _tess(image: Image.Image, psm_chain: tuple[int, ...]) -> str:
        for psm in psm_chain:
            text = pytesseract.image_to_string(
                image, lang="khm", config=f"--oem 1 --psm {psm}"
            ).strip()
            if text:
                return text
        return ""

    tess_full = _tess(line_for_ocr, (7, 6))
    tess_name = _tess(name_for_ocr, (7, 8))
    kiri_result = kiri.recognize(name_for_ocr, mode="line")

    return KhmerIDNameResponse(
        line_bbox=list(location.line_bbox),
        name_bbox=list(location.name_bbox),
        full_line_tesseract=tess_full,
        name_after_colon=extract_after_colon(tess_full),
        name_tesseract=tess_name,
        name_kiri_single_line=kiri_result.text,
        kiri_note=kiri_result.note,
    )


@router.post("/kiri", response_model=OCRResponse)
async def ocr_kiri(
    file: UploadFile = File(...),
    decode_method: str = Query(
        "accurate",
        description="Kiri decode method: 'fast' (CTC), 'accurate' (default), or 'beam' (best quality)",
    ),
    mode: str = Query(
        "full",
        description="'full' uses Kiri's text detector + recognizer (default). 'line' skips detection — use when the input is already a tight single-line crop. The 'line' mode also returns the model's confidence in the response note.",
    ),
    service: OCRService = Depends(get_kiri_service),
) -> OCRResponse:
    return await _run(service, file, decode_method=decode_method, mode=mode)
