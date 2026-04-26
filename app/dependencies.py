from app.services import EasyOCRService, PaddleOCRService, TesseractOCRService

_tesseract = TesseractOCRService()
_easyocr = EasyOCRService()
_paddleocr = PaddleOCRService()


def get_tesseract_service() -> TesseractOCRService:
    return _tesseract


def get_easyocr_service() -> EasyOCRService:
    return _easyocr


def get_paddleocr_service() -> PaddleOCRService:
    return _paddleocr
