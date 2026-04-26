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
