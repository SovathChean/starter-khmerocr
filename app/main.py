from fastapi import FastAPI

from app.routers import health, ocr


def create_app() -> FastAPI:
    app = FastAPI(title="khmerid-ocr")
    app.include_router(health.router)
    app.include_router(ocr.router)
    return app


app = create_app()
