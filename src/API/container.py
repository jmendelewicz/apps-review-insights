"""
container.py — Production-grade Dependency Injection container

Designed for:
- FastAPI (Depends)
- Hugging Face deployment
- Remote LLMs (Gemini Flash)
- Thread-safe singleton lifecycle

Key properties:
- Container-scoped singletons (NOT global function closures)
- Explicit construction (no __new__ hacks)
- Lazy + thread-safe initialization
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Any

from src.DataAccess.Ingestion.google_play_wrapper import GooglePlayScraper
from src.DataAccess.Refinary.google_review_cleaner import GoogleReviewCleaner
from src.Orchestration.google_play_service import GooglePlayService

from src.Models.nlptown_bert_base_model import BertBaseSentimentModel
from src.Models.gemini_flash_model import GeminiFlashModel


class Container:
    """
    Central dependency registry with container-scoped lifecycle control.
    """

    def __init__(self, lang: str = "es", country: str = "ar"):
        self._lang = lang
        self._country = country

        self._instances: Dict[str, Any] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Core singleton resolver
    # ------------------------------------------------------------------

    def _get(self, key: str, factory):
        """
        Thread-safe singleton resolver.
        """
        if key not in self._instances:
            with self._lock:
                if key not in self._instances:
                    self._instances[key] = factory()
        return self._instances[key]

    # ------------------------------------------------------------------
    # Data Access Layer
    # ------------------------------------------------------------------

    def scraper(self) -> GooglePlayScraper:
        print("Resolving GooglePlayScraper with lang =", self._lang, "and country =", self._country)
        return self._get(
            "scraper",
            lambda: GooglePlayScraper(
                lang=self._lang,
                country=self._country
            )
        )

    def cleaner(self) -> GoogleReviewCleaner:
        print("Resolving GoogleReviewCleaner...")
        return self._get(
            "cleaner",
            lambda: GoogleReviewCleaner()
        )


    # ------------------------------------------------------------------
    # Model Layer
    # ------------------------------------------------------------------

    def sentiment_model(self) -> BertBaseSentimentModel:
        print("Resolving BertBaseSentimentModel...")
        return self._get(
            "sentiment_model",
            lambda: BertBaseSentimentModel()
        )

    def summarization_model(self) -> GeminiFlashModel:
        """
        Gemini = remote inference → lightweight initialization
        """
        print("Resolving GeminiFlashModel...")
        print("GOOGLE_API_KEY =", os.getenv("GOOGLE_API_KEY"))
        return self._get(
            "summarization_model",
            lambda: GeminiFlashModel(
                api_key=os.getenv("GOOGLE_API_KEY")
                )
        )

    # ------------------------------------------------------------------
    # Service Layer
    # ------------------------------------------------------------------

    def google_play_service(self) -> GooglePlayService:
        print("Resolving GooglePlayService dependencies...")
        return self._get(
            "google_play_service",
            lambda: GooglePlayService(
                scraper=self.scraper(),
                cleaner=self.cleaner(),
                sentiment_model=self.sentiment_model(),
                summarization_model=self.summarization_model(),
            )
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def wire(self) -> None:
        """
        Optional eager initialization.

        For Hugging Face:
        - Keep sentiment model warm
        - Gemini does NOT need preloading
        """
        self.sentiment_model()
        self.google_play_service()


# ---------------------------------------------------------------------------
# Global container (process-scoped)
# ---------------------------------------------------------------------------

_container: Container | None = None
_container_lock = threading.Lock()

def get_container() -> Container:
    """
    FastAPI-compatible provider.
    """
    global _container
    if _container is None:
        with _container_lock:
            if _container is None:
                c = Container()
                c.wire()           # wire before assigning
                _container = c     # assign only after fully ready
    return _container