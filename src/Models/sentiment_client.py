from __future__ import annotations

import os
from typing import Iterable, Protocol

import requests


_LABELS = {"positive", "negative", "neutral"}
_STAR_MAP = {
    "1 star": "negative", "2 stars": "negative", "3 stars": "neutral",
    "4 stars": "positive", "5 stars": "positive",
}


class SentimentBackend(Protocol):
    def classify(self, texts: list[str]) -> list[str]: ...


class HFSentimentBackend:

    def __init__(self, endpoint: str, token: str | None = None, timeout: int = 60):
        self.endpoint = endpoint.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self.timeout = timeout

    def classify(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        resp = requests.post(
            f"{self.endpoint}/classify",
            json={"texts": texts},
            headers=self.headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        labels = data.get("sentiments") or data.get("labels") or []
        return [l if l in _LABELS else "neutral" for l in labels]


class LocalSentimentBackend:

    MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"

    def __init__(self, batch_size: int = 32):
        self._pipe = None
        self.batch_size = batch_size

    def _ensure_pipe(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline(
                "text-classification",
                model=self.MODEL,
                truncation=True,
                max_length=512,
            )
        return self._pipe

    def classify(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        pipe = self._ensure_pipe()
        out: list[str] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            for r in pipe(chunk):
                out.append(_STAR_MAP.get(r["label"].lower(), "neutral"))
        return out


class SentimentClient:

    def __init__(self, primary: SentimentBackend, fallback: SentimentBackend | None = None):
        self.primary = primary
        self.fallback = fallback

    def classify(self, texts: Iterable[str]) -> list[str]:
        items = [str(t) for t in texts]
        try:
            return self.primary.classify(items)
        except Exception:
            if self.fallback is None:
                raise
            return self.fallback.classify(items)


def build_client() -> SentimentClient:
    endpoint = os.environ.get("HF_SENTIMENT_URL", "").strip()
    token = os.environ.get("HF_SENTIMENT_TOKEN", "").strip() or None
    local = LocalSentimentBackend()
    if endpoint:
        return SentimentClient(primary=HFSentimentBackend(endpoint, token), fallback=local)
    return SentimentClient(primary=local)
