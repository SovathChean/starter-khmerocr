import tempfile
from functools import lru_cache
from pathlib import Path

from PIL import Image

from app.schemas import OCRResult
from app.services.base import OCRService

VALID_DECODE_METHODS = {"fast", "accurate", "beam"}


@lru_cache(maxsize=4)
def _kiri_engine(decode_method: str):
    from kiri_ocr import OCR

    return OCR(decode_method=decode_method)


class KiriOCRService(OCRService):
    name = "kiri"
    default_decode_method = "accurate"

    def recognize(self, image: Image.Image, decode_method: str | None = None) -> OCRResult:
        method = decode_method if decode_method in VALID_DECODE_METHODS else self.default_decode_method
        engine = _kiri_engine(method)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            path = tmp.name
        try:
            image.save(path, format="PNG")
            text, _ = engine.extract_text(path)
        finally:
            Path(path).unlink(missing_ok=True)

        return OCRResult(
            text=(text or "").strip(),
            languages=["khm", "eng"],
            note=f"decode_method={method}",
        )
