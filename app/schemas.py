from dataclasses import dataclass

from pydantic import BaseModel


class OCRResponse(BaseModel):
    engine: str
    languages: list[str]
    text: str
    elapsed_ms: int
    note: str | None = None


@dataclass
class OCRResult:
    text: str
    languages: list[str]
    note: str | None = None


class KhmerIDNameResponse(BaseModel):
    line_bbox: list[int]
    name_bbox: list[int]
    full_line_tesseract: str
    name_after_colon: str
    name_tesseract: str
    name_kiri_single_line: str
    kiri_note: str | None = None
