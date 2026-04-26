from abc import ABC, abstractmethod

from PIL import Image

from app.schemas import OCRResult


class OCRService(ABC):
    name: str = ""
    default_lang: str = ""

    @abstractmethod
    def recognize(self, image: Image.Image, lang: str | None) -> OCRResult:
        ...
