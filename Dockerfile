FROM python:3.12-slim

# System deps:
#   tesseract-ocr + khm + eng  -> Tesseract engine + Khmer/English language data
#   libgl1, libglib2.0-0       -> OpenCV runtime (transitive via easyocr / paddleocr)
#   libgomp1                   -> OpenMP runtime needed by paddlepaddle (CPU)
#   git                        -> required to pip-install kiri-ocr from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-khm \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- CPU-only Python deps ---

# 1. Pre-install CPU-only torch + torchvision from PyTorch's CPU index.
#    Without this, easyocr would pull the CUDA-bundled torch wheel (multi-GB).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

# 2. Project requirements (paddlepaddle + onnxruntime here are CPU variants;
#    easyocr reuses the CPU torch installed above).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 3. Kiri OCR from GitHub with --no-deps (avoids onnxruntime-gpu pin).
RUN pip install --no-cache-dir --no-deps git+https://github.com/mrrtmob/kiri-ocr.git

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
