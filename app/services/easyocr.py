from functools import lru_cache

import numpy as np
from PIL import Image

from app.schemas import OCRResult
from app.services.base import OCRService


@lru_cache(maxsize=4)
def _easyocr_reader(langs_key: str):
    import easyocr

    return easyocr.Reader(langs_key.split(","), gpu=False)


class EasyOCRService(OCRService):
    name = "easyocr"
    default_lang = "en"

    def recognize(self, image: Image.Image, lang: str | None = None) -> OCRResult:
        requested = lang or self.default_lang
        note: str | None = None
        codes = requested.split(",")
        if "khm" in codes or "khmer" in codes:
            note = "EasyOCR has no Khmer model; falling back to English only."
            requested = "en"
        reader = _easyocr_reader(requested)
        lines = reader.readtext(np.array(image), detail=0)
        return OCRResult(
            text="\n".join(lines),
            languages=requested.split(","),
            note=note,
        )
