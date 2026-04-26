import pytesseract
from PIL import Image

from app.schemas import OCRResult
from app.services.base import OCRService


class TesseractOCRService(OCRService):
    name = "pytesseract"
    default_lang = "eng+khm"

    def recognize(self, image: Image.Image, lang: str | None) -> OCRResult:
        used_lang = lang or self.default_lang
        text = pytesseract.image_to_string(image, lang=used_lang)
        return OCRResult(text=text.strip(), languages=used_lang.split("+"))
