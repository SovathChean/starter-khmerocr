# khmerid-ocr

FastAPI starter for the Khmer ID OCR service.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

System dependency for the Tesseract engine (macOS):

```bash
brew install tesseract tesseract-lang   # tesseract-lang provides khm.traineddata
tesseract --list-langs                  # confirm "eng" and "khm" appear
```

Kiri OCR install (needs `--no-deps` on macOS — its `pyproject.toml` requires `onnxruntime-gpu`, which is Linux-CUDA only; we already pulled the CPU `onnxruntime` via `requirements.txt`):

```bash
pip install --no-deps git+https://github.com/mrrtmob/kiri-ocr.git
```

Kiri auto-downloads its weights from Hugging Face on first call (cached under `~/.cache/huggingface/`). The PyPI release `kiri-ocr 0.2.2` is **not** compatible with the current HF model — install from GitHub `main` as shown above.

## Run

```bash
uvicorn app.main:app --reload
```

The API will be available at http://127.0.0.1:8000. Interactive docs at http://127.0.0.1:8000/docs.

## Test

```bash
pytest
```

## OCR engines

Four POST endpoints, each accepting `multipart/form-data` with a single field `file`:

| Endpoint | Default | Override |
| --- | --- | --- |
| `POST /ocr/tesseract` | `lang=eng+khm` | `?lang=eng` or `?lang=khm` |
| `POST /ocr/easyocr`   | `lang=en` (no Khmer model — falls back with a `note`) | `?lang=en,fr` |
| `POST /ocr/paddleocr` | `lang=en` | `?lang=ch`, `?lang=korean`, etc. (no Khmer model) |
| `POST /ocr/kiri`      | `decode_method=accurate` (Khmer+English native) | `?decode_method=fast` or `?decode_method=beam` |

Try it:

```bash
curl -F "file=@sample.png" http://localhost:8000/ocr/tesseract
curl -F "file=@sample.png" http://localhost:8000/ocr/easyocr
curl -F "file=@sample.png" http://localhost:8000/ocr/paddleocr
curl -F "file=@sample.png" http://localhost:8000/ocr/kiri
```

The first call to EasyOCR, PaddleOCR, and Kiri downloads model weights and is slow; subsequent calls reuse the cached engine. Kiri caches its model under `~/.cache/huggingface/`.

## Layout

```
app/
  main.py             FastAPI app factory
  schemas.py          OCRResponse (API) + OCRResult (internal)
  images.py           shared upload-to-PIL.Image helper
  dependencies.py     FastAPI providers for each OCR service
  routers/
    health.py         / and /health endpoints
    ocr.py            thin handlers for /ocr/* — delegate to services
  services/
    base.py           OCRService ABC (recognize -> OCRResult)
    tesseract.py      TesseractOCRService
    easyocr.py        EasyOCRService (with lru_cache lazy reader)
    paddleocr.py      PaddleOCRService (with lru_cache lazy engine)
    kiri.py           KiriOCRService — Khmer+English native (mrrtmob/kiri-ocr)
tests/
  test_health.py      smoke test with TestClient
  test_ocr.py         OCR endpoint tests via app.dependency_overrides
```
