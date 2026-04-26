from app.services.base import OCRService
from app.services.easyocr import EasyOCRService
from app.services.paddleocr import PaddleOCRService
from app.services.tesseract import TesseractOCRService

__all__ = [
    "OCRService",
    "TesseractOCRService",
    "EasyOCRService",
    "PaddleOCRService",
]
