from abc import ABC, abstractmethod

from PIL import Image

from app.schemas import OCRResult


class OCRService(ABC):
    name: str = ""

    @abstractmethod
    def recognize(self, image: Image.Image, **kwargs) -> OCRResult:
        ...
