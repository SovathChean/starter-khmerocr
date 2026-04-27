import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.dependencies import (
    get_easyocr_service,
    get_kiri_service,
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

    def recognize(self, image, **kwargs):
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


def test_kiri_endpoint_returns_text():
    _override(
        get_kiri_service,
        "kiri",
        OCRResult(
            text="សួស្តី Hello",
            languages=["khm", "eng"],
            note="decode_method=accurate",
        ),
    )
    files = {"file": ("t.png", _png_bytes(), "image/png")}
    response = client.post("/ocr/kiri", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "kiri"
    assert body["languages"] == ["khm", "eng"]
    assert body["text"] == "សួស្តី Hello"
    assert body["note"] == "decode_method=accurate"


def test_kiri_endpoint_forwards_decode_method_and_mode():
    captured: dict = {}

    class CapturingService:
        name = "kiri"

        def recognize(self, image, **kwargs):
            captured.update(kwargs)
            return OCRResult(
                text="ok",
                languages=["khm", "eng"],
                note=f"decode_method={kwargs.get('decode_method')}; mode={kwargs.get('mode')}",
            )

    app.dependency_overrides[get_kiri_service] = lambda: CapturingService()
    files = {"file": ("t.png", _png_bytes(), "image/png")}
    response = client.post("/ocr/kiri?decode_method=beam&mode=line", files=files)
    assert response.status_code == 200
    assert captured == {"decode_method": "beam", "mode": "line"}
    assert response.json()["note"] == "decode_method=beam; mode=line"


def test_locate_name_line_falls_back_when_keyword_misread(monkeypatch):
    """When Tesseract misreads 'នាម' (e.g. as 'នាទ'), the keyword search fails
    but the structural fallback should still find the topmost line with a colon
    in the upper half of the image."""
    from PIL import Image as _Image
    import pytesseract as _pyt
    from app import khmer_id

    fake_data = {
        # Note: the first word is a misread of "គោត្តនាមនិងនាម:ជាន" — no "នាម"
        # substring survives, but the colon does. The fallback should still pick
        # this line because it's the topmost in the upper half with a colon.
        "text":      ["ទូនាទនិងខាទ:បាន", "ស៊ីវិន",
                      "ថ្ងៃខែឆ្នាំកំណើត:", "០១.០៩.១៩៩៦"],
        "left":      [200, 700, 200, 600],
        "top":       [50, 50, 200, 200],
        "width":     [400, 150, 350, 200],
        "height":    [80, 80, 80, 80],
        "conf":      [40, 75, 80, 70],
        "block_num": [1, 1, 1, 1],
        "par_num":   [1, 1, 1, 1],
        "line_num":  [1, 1, 2, 2],   # two lines: name (no នាម due to misread) + DOB
    }
    monkeypatch.setattr(_pyt, "image_to_data", lambda *a, **kw: fake_data)

    img = _Image.new("RGB", (1800, 1000), "white")  # tall enough that y=50 is in upper half
    loc = khmer_id.locate_name_line(img)
    assert loc is not None, "fallback should have found the colon-bearing top line"
    # The fallback line top is 50, name region starts after the colon proportionally
    assert loc.line_bbox[1] == 50
    # The label text is whatever Tesseract returned for the topmost line
    assert "បាន" in loc.label_text  # the misread first name


def test_khmer_id_name_no_line_found(monkeypatch):
    """When Tesseract finds no line containing 'នាម' on the image, return 404."""
    import app.routers.ocr as ocr_mod
    monkeypatch.setattr(ocr_mod, "locate_name_line", lambda image: None)
    files = {"file": ("t.png", _png_bytes(), "image/png")}
    response = client.post("/ocr/khmer-id-name", files=files)
    assert response.status_code == 404
    assert "name line" in response.json()["detail"].lower()


def test_khmer_id_name_returns_combined_payload(monkeypatch):
    """Locate is stubbed; both Tesseract calls and Kiri are stubbed.
    Verifies the response shape and that all three readings flow through."""
    import app.routers.ocr as ocr_mod
    from app.khmer_id import KhmerIDNameLocation

    fake_loc = KhmerIDNameLocation(
        line_bbox=(100, 50, 800, 200),
        name_bbox=(400, 50, 800, 200),
        label_text="គោត្តនាមនិងនាម: ...",
    )
    monkeypatch.setattr(ocr_mod, "locate_name_line", lambda image: fake_loc)

    tess_calls = []

    def fake_image_to_string(image, lang=None, config=None):
        tess_calls.append((image.size, lang, config))
        # First call is the wide line ("LABEL:NAME"); second is the name-only crop
        return "LABEL៖TESS_FULL_LINE" if len(tess_calls) == 1 else "TESS_NAME_ONLY"

    monkeypatch.setattr(ocr_mod.pytesseract, "image_to_string", fake_image_to_string)

    class FakeKiri:
        name = "kiri"
        def recognize(self, image, **kwargs):
            return OCRResult(
                text="KIRI_NAME",
                languages=["khm", "eng"],
                note=f"mode={kwargs.get('mode')}",
            )

    app.dependency_overrides[get_kiri_service] = lambda: FakeKiri()

    # Use an image already at/above UPSCALE_TARGET_WIDTH so the router doesn't
    # upscale (which would scale bboxes back to original coords for the response).
    files = {"file": ("t.png", _png_bytes(size=(1800, 500)), "image/png")}
    response = client.post("/ocr/khmer-id-name", files=files)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["line_bbox"] == [100, 50, 800, 200]
    assert body["name_bbox"] == [400, 50, 800, 200]
    assert body["full_line_tesseract"] == "LABEL៖TESS_FULL_LINE"
    assert body["name_after_colon"] == "TESS_FULL_LINE"
    assert body["name_tesseract"] == "TESS_NAME_ONLY"
    assert body["name_kiri_single_line"] == "KIRI_NAME"
    assert body["kiri_note"] == "mode=line"
    assert len(tess_calls) == 2


def test_rejects_non_image():
    _override(get_tesseract_service, "pytesseract", OCRResult(text="", languages=["eng"]))
    files = {"file": ("note.txt", b"hello world", "text/plain")}
    response = client.post("/ocr/tesseract", files=files)
    assert response.status_code == 400


def test_missing_file():
    response = client.post("/ocr/tesseract")
    assert response.status_code == 422
