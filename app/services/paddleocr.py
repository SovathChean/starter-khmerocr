from functools import lru_cache

import numpy as np
from PIL import Image

from app.schemas import OCRResult
from app.services.base import OCRService


@lru_cache(maxsize=4)
def _paddle_engine(lang: str):
    from paddleocr import PaddleOCR

    return PaddleOCR(lang=lang)


class PaddleOCRService(OCRService):
    name = "paddleocr"
    default_lang = "en"

    def recognize(self, image: Image.Image, lang: str | None = None) -> OCRResult:
        used_lang = lang or self.default_lang
        engine = _paddle_engine(used_lang)
        results = engine.predict(np.array(image))

        lines: list[str] = []
        for r in results or []:
            texts = r.get("rec_texts") if hasattr(r, "get") else None
            if texts:
                lines.extend(t for t in texts if t)

        return OCRResult(text="\n".join(lines), languages=[used_lang])
