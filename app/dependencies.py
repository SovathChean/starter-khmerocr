from app.services import (
    EasyOCRService,
    KiriOCRService,
    PaddleOCRService,
    TesseractOCRService,
)

_tesseract = TesseractOCRService()
_easyocr = EasyOCRService()
_paddleocr = PaddleOCRService()
_kiri = KiriOCRService()


def get_tesseract_service() -> TesseractOCRService:
    return _tesseract


def get_easyocr_service() -> EasyOCRService:
    return _easyocr


def get_paddleocr_service() -> PaddleOCRService:
    return _paddleocr


def get_kiri_service() -> KiriOCRService:
    return _kiri
