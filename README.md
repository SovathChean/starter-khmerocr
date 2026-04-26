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
| `POST /ocr/kiri`      | `decode_method=accurate` `mode=full` | `?decode_method=fast\|beam`, `?mode=line` (single-line recognizer that skips Kiri's text detector — use on tightly-cropped lines) |
| `POST /ocr/khmer-id-name` | n/a | Auto-locates the Khmer name line on a Cambodian ID, returns Tesseract + Kiri readings of the cropped name (see below) |

Try it:

```bash
curl -F "file=@sample.png" http://localhost:8000/ocr/tesseract
curl -F "file=@sample.png" http://localhost:8000/ocr/easyocr
curl -F "file=@sample.png" http://localhost:8000/ocr/paddleocr
curl -F "file=@sample.png" http://localhost:8000/ocr/kiri
```

The first call to EasyOCR, PaddleOCR, and Kiri downloads model weights and is slow; subsequent calls reuse the cached engine. Kiri caches its model under `~/.cache/huggingface/`.

### Extracting the Khmer name from an ID card

Tesseract on the full ID image bleeds Latin glyphs from the English transliteration onto the Khmer name (`HO 9ាជ្រ...`) and Kiri's text detector sometimes locks onto the digit row instead of the name. `POST /ocr/khmer-id-name` works around both:

1. Run Tesseract `image_to_data` on the whole image to find the topmost line containing `នាម`.
2. Locate the colon inside that line — split into a label region and a name region.
3. Clamp the line bottom at ~1.5× the label height (otherwise long subscripts like `្រ` make Tesseract's bbox extend down into the English transliteration line below).
4. Run **Tesseract `psm=7 lang=khm`** on the full line *and* on the name-only crop.
5. Run **Kiri's `recognize_single_line_image`** (skips Kiri's broken detector) on the name-only crop.

The response returns **four** independent readings of the name:

| Field | Source | Best for |
|---|---|---|
| `name_after_colon` | Tesseract on the full line, then `split(":")` | **Often the most accurate** — the full-line OCR has more context and the post-split removes the label noise |
| `name_tesseract` | Tesseract directly on the name-only crop | Falls back to `psm=8` (single-word) for short crops |
| `name_kiri_single_line` | Kiri's `recognize_single_line_image` on the name-only crop | Often gets surname clusters Tesseract misses |
| `full_line_tesseract` | Raw Tesseract output of the whole label+name line | Useful for debugging / comparing |

Plus the detected `line_bbox` and `name_bbox` so you can verify the crop visually. The four readings have complementary failure modes — compare them to ensemble or pick the right one per ID.

The endpoint also adapts to small images (e.g. phone screenshots): if the detected line is shorter than ~150 px tall, it upscales the crop with Lanczos before passing to OCR.

```bash
curl -F "file=@my-id.jpg" http://localhost:8000/ocr/khmer-id-name
```

## Layout

```
app/
  main.py             FastAPI app factory
  schemas.py          OCRResponse + OCRResult + KhmerIDNameResponse
  images.py           shared upload-to-PIL.Image helper
  khmer_id.py         Cambodian ID layout helpers (locate_name_line)
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
