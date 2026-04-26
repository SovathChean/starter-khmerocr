import tempfile
from functools import lru_cache
from pathlib import Path

from PIL import Image

from app.schemas import OCRResult
from app.services.base import OCRService

VALID_DECODE_METHODS = {"fast", "accurate", "beam"}
VALID_MODES = {"full", "line"}


@lru_cache(maxsize=4)
def _kiri_engine(decode_method: str):
    from kiri_ocr import OCR

    return OCR(decode_method=decode_method)


def _save_temp_png(image: Image.Image) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    image.save(path, format="PNG")
    return path


class KiriOCRService(OCRService):
    name = "kiri"
    default_decode_method = "accurate"
    default_mode = "full"

    def recognize(
        self,
        image: Image.Image,
        decode_method: str | None = None,
        mode: str | None = None,
    ) -> OCRResult:
        method = decode_method if decode_method in VALID_DECODE_METHODS else self.default_decode_method
        chosen_mode = mode if mode in VALID_MODES else self.default_mode
        engine = _kiri_engine(method)

        path = _save_temp_png(image)
        try:
            if chosen_mode == "line":
                # Single-line recognizer — skips Kiri's text detector entirely.
                # Use this when the input is already a tightly-cropped single text line.
                text, conf = engine.recognize_single_line_image(path)
                note = f"decode_method={method}; mode=line; conf={conf:.2f}"
            else:
                # Full pipeline: Kiri detects regions, then recognizes each one.
                text, _ = engine.extract_text(path)
                note = f"decode_method={method}; mode=full"
        finally:
            Path(path).unlink(missing_ok=True)

        return OCRResult(
            text=(text or "").strip(),
            languages=["khm", "eng"],
            note=note,
        )
