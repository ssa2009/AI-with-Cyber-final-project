from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.pii_detector import PIIDetector


detector: PIIDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector

    try:
        detector = PIIDetector()
    except FileNotFoundError:
        detector = None

    yield


app = FastAPI(
    title="Cyber for AI PII Detector",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)

templates = Jinja2Templates(directory="templates")


class TextRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=10_000,
        description="Text that will be checked for PII",
    )


class DetectionResponse(BaseModel):
    contains_pii: bool
    confidence: float
    pattern_matches: list[dict[str, Any]]
    reason: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
    }


@app.post(
    "/api/check-pii",
    response_model=DetectionResponse,
)
async def check_pii(request: TextRequest):
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The model is not available. "
                "Run python train_model.py and restart the server."
            ),
        )

    return detector.predict(request.text)