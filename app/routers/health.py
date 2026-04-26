from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def root() -> dict[str, str]:
    return {"message": "khmerid-ocr API"}


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
