from functools import lru_cache

import numpy as np
from PIL import Image

from app.schemas import OCRResult
from app.services.base import OCRService


@lru_cache(maxsize=4)
def _paddle_engine(lang: str):
    from paddleocr import PaddleOCR

    return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)


class PaddleOCRService(OCRService):
    name = "paddleocr"
    default_lang = "ch"

    def recognize(self, image: Image.Image, lang: str | None) -> OCRResult:
        used_lang = lang or self.default_lang
        engine = _paddle_engine(used_lang)
        result = engine.ocr(np.array(image), cls=True)

        lines: list[str] = []
        for page in result or []:
            if not page:
                continue
            for entry in page:
                if entry and len(entry) >= 2 and entry[1]:
                    lines.append(entry[1][0])

        return OCRResult(text="\n".join(lines), languages=[used_lang])
