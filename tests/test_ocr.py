import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.dependencies import (
    get_easyocr_service,
    get_paddleocr_service,
    get_tesseract_service,
)
from app.main import app
from app.schemas import OCRResult

client = TestClient(app)


def _png_bytes(size: tuple[int, int] = (4, 4), color: tuple[int, int, int] = (255, 255, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class StubService:
    def __init__(self, name: str, result: OCRResult):
        self.name = name
        self._result = result

    def recognize(self, image, lang):
        return self._result


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def _override(dependency, name: str, result: OCRResult):
    app.dependency_overrides[dependency] = lambda: StubService(name, result)


def test_tesseract_endpoint_returns_text():
    _override(
        get_tesseract_service,
        "pytesseract",
        OCRResult(text="STUB TEXT", languages=["eng", "khm"]),
    )
    files = {"file": ("t.png", _png_bytes(), "image/png")}
    response = client.post("/ocr/tesseract", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "pytesseract"
    assert body["text"] == "STUB TEXT"
    assert body["languages"] == ["eng", "khm"]
    assert isinstance(body["elapsed_ms"], int)


def test_easyocr_endpoint_returns_text():
    _override(
        get_easyocr_service,
        "easyocr",
        OCRResult(text="hello\nworld", languages=["en"]),
    )
    files = {"file": ("t.png", _png_bytes(), "image/png")}
    response = client.post("/ocr/easyocr", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "easyocr"
    assert body["text"] == "hello\nworld"
    assert body["languages"] == ["en"]
    assert body["note"] is None


def test_easyocr_falls_back_when_khmer_requested():
    _override(
        get_easyocr_service,
        "easyocr",
        OCRResult(
            text="fallback",
            languages=["en"],
            note="EasyOCR has no Khmer model; falling back to English only.",
        ),
    )
    files = {"file": ("t.png", _png_bytes(), "image/png")}
    response = client.post("/ocr/easyocr?lang=khm", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["languages"] == ["en"]
    assert "Khmer" in (body["note"] or "")


def test_paddleocr_endpoint_returns_text():
    _override(
        get_paddleocr_service,
        "paddleocr",
        OCRResult(text="from paddle", languages=["ch"]),
    )
    files = {"file": ("t.png", _png_bytes(), "image/png")}
    response = client.post("/ocr/paddleocr", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "paddleocr"
    assert body["text"] == "from paddle"
    assert body["languages"] == ["ch"]


def test_rejects_non_image():
    _override(get_tesseract_service, "pytesseract", OCRResult(text="", languages=["eng"]))
    files = {"file": ("note.txt", b"hello world", "text/plain")}
    response = client.post("/ocr/tesseract", files=files)
    assert response.status_code == 400


def test_missing_file():
    response = client.post("/ocr/tesseract")
    assert response.status_code == 422
