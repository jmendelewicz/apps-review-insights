import asyncio

from dotenv import load_dotenv

from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import List

from fastapi import FastAPI, Depends, HTTPException, Query, Request
from pydantic import BaseModel

load_dotenv()  # Load environment variables from .env file before container import

from src.API.container import Container


# ---------------------------------------------------------------------------
# Lifespan: boot the DI container once, tear it down on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    container = Container()

    # Warm up the model without blocking the event loop.
    # Use get_running_loop (get_event_loop is deprecated in Python 3.10+).
    loop = asyncio.get_running_loop()
    print("Starting model load...")
    await loop.run_in_executor(None, container.sentiment_model)
    print("Model loaded.")
    print("Starting Google Play service initialization...")
    container.google_play_service()  # Pre-initialize service (optional)
    print("Google Play service initialized.")
    app.state.container = container
    yield


app = FastAPI(
    title="Google Play Review Analyser",
    description="Scrape, clean, analyse sentiment and summarise Play Store reviews.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Dependency: resolve GooglePlayService from the container for each request
# ---------------------------------------------------------------------------

def get_service(request: Request):
    return request.app.state.container.google_play_service()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok"}


# -- Full pipeline -----------------------------------------------------------

@app.post(
    "/pipeline/{app_id}",
    summary="Run the full pipeline (scrape → clean → sentiment → summary)",
    tags=["Pipeline"],
)
def run_pipeline(
    app_id: str,
    limit: int = Query(default=1000, ge=1, le=5000, description="Max reviews to scrape"),
    service=Depends(get_service),
):
    """
    Run the full pipeline for a given app_id and review limit.
    Returns a summary of top positive and negative features.
    """
    try:
        result = service.run_pipeline(app_id, limit=limit)  
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "result": asdict(result)
    }

@app.get("/ready")
def ready(request: Request):
    container: Container = request.app.state.container
    if not container.is_loaded("sentiment_model"):
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ready"}


# -- Sentiment classification (used by the Streamlit frontend via HFSentimentBackend) --

class ClassifyRequest(BaseModel):
    texts: List[str]


@app.post(
    "/classify",
    summary="Classify sentiment of a list of texts",
    tags=["Sentiment"],
)
def classify(payload: ClassifyRequest, request: Request):
    """
    Accepts a list of texts and returns their sentiment labels.
    Response schema: {"sentiments": ["positive", "negative", "neutral", ...]}
    Consumed by HFSentimentBackend in Models/sentiment_client.py.
    """
    if not payload.texts:
        return {"sentiments": []}
    model = request.app.state.container.sentiment_model()
    sentiments = model.classify_sentiment(payload.texts)
    return {"sentiments": sentiments}