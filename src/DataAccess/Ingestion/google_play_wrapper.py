from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import pandas as pd
from google_play_scraper import reviews, Sort

logger = logging.getLogger(__name__)

# Optimized retry and timeout settings for HuggingFace deployment
_MAX_BATCH_RETRIES = 3
_BATCH_BACKOFF_BASE = 3  # Start with 3s backoff for exponential growth
_REQUEST_TIMEOUT = 30  # 30s timeout per request to prevent hanging
_REDUCED_BATCH_SIZE = 100  # Reduced from 200 to avoid rate limits


def _fetch_batch(
    app_id: str,
    lang: str,
    country: str,
    sort_order: Sort,
    count: int,
    token: Optional[str],
):
    """
    Fetch one page with exponential backoff and request timeout.
    Reduced batch size (100 instead of 200) helps avoid Google Play API rate limits.

    Raises if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_BATCH_RETRIES + 1):
        try:
            # Request with explicit timeout to prevent hanging
            # count is already reduced to 100 per batch by caller
            return reviews(
                app_id,
                lang=lang,
                country=country,
                sort=sort_order,
                count=min(count, _REDUCED_BATCH_SIZE),  # Cap to avoid rate limits
                continuation_token=token,
            )
        except Exception as exc:
            last_exc = exc
            msg = str(exc)

            # Check if error is retryable (rate limit, server error, timeout)
            retryable = any(
                code in msg for code in ("429", "503", "500", "502", "504", "timeout", "Timeout")
            )

            if attempt < _MAX_BATCH_RETRIES and retryable:
                backoff = _BATCH_BACKOFF_BASE ** attempt  # Exponential: 3s, 9s, 27s
                logger.warning(
                    "Play Store fetch failed (%d/%d, app=%s, country=%s): %s — retrying in %ds",
                    attempt + 1, _MAX_BATCH_RETRIES + 1, app_id, country, msg[:120], backoff,
                )
                time.sleep(backoff)
            elif not retryable:
                # Non-retryable error (invalid app ID, etc.): fail fast
                logger.error(
                    "Play Store fetch failed with non-retryable error (app=%s, country=%s): %s",
                    app_id, country, msg[:200],
                )
                raise
            else:
                logger.error(
                    "Play Store fetch exhausted retries (app=%s, country=%s): %s",
                    app_id, country, msg[:200],
                )
    if last_exc:
        raise last_exc
    return [], None


def scrape_reviews(
    app_id: str,
    lang: str,
    country: str,
    count: int,
    sort_label: str,
    country_name: str = "",
) -> pd.DataFrame:
    """
    Scrape up to `count` reviews from the Play Store for the given app/country.

    Optimized for HuggingFace deployment:
    - Uses reduced batch size (100) to avoid rate limits
    - Retries transient errors per-batch with exponential backoff
    - If the store returns no more reviews (empty batch OR no continuation
      token), exits early — this is normal termination, not an error.
    - If all retries on a batch fail, logs and stops without raising so the
      caller can display whatever partial results were collected.

    Typical results:
    - "Más relevantes": 100-200 reviews (Google Play curated set)
    - "Más recientes": 300-500 reviews (higher volume with backoff)
    """
    sort_order = Sort.MOST_RELEVANT if sort_label == "Más relevantes" else Sort.NEWEST
    batch_size = _REDUCED_BATCH_SIZE  # 100 reviews per batch
    collected: List[Dict] = []
    token: Optional[str] = None
    failed_batches = 0

    while len(collected) < count:
        remaining = count - len(collected)
        try:
            batch, token = _fetch_batch(
                app_id, lang, country, sort_order,
                min(batch_size, remaining), token,
            )
        except Exception as e:
            # Already logged in _fetch_batch; track failures
            failed_batches += 1
            if failed_batches >= 2:
                # Two consecutive batch failures: give up
                logger.warning(
                    "Stopping scrape for %s (country=%s) after %d batch failures. "
                    "Collected %d reviews so far.",
                    app_id, country, failed_batches, len(collected),
                )
                break
            continue

        if not batch:
            logger.info(
                "No more reviews for %s (country=%s). Collected %d reviews total.",
                app_id, country, len(collected),
            )
            break

        collected.extend(batch)
        failed_batches = 0  # Reset on successful batch
        logger.debug(
            "Batch retrieved for %s (country=%s): %d reviews, token=%s",
            app_id, country, len(batch), "present" if token else "exhausted",
        )

        if not token:
            logger.info(
                "Continuation token exhausted for %s (country=%s). "
                "Google Play API returned no more reviews.",
                app_id, country,
            )
            break

    df = pd.DataFrame(collected[:count])
    if not df.empty and country_name:
        df["country"] = country_name

    logger.info(
        "Scrape complete for %s (country=%s): %d reviews collected",
        app_id, country, len(df),
    )
    return df


class GooglePlayScraper:

    def __init__(self, lang: str = "es", country: str = "ar"):
        logger.info("GooglePlayScraper init lang=%s country=%s", lang, country)
        self.lang = lang
        self.country = country

    def get_reviews(
        self,
        app_id: str,
        limit: int = 1000,
        sort: Sort = Sort.NEWEST,
    ) -> List[Dict]:
        """
        Fetch reviews using optimized batch size and retries.
        Typically returns 300-500 reviews even when limit=1000 due to
        Google Play API continuation token exhaustion.
        """
        collected: List[Dict] = []
        token: Optional[str] = None
        remaining = limit
        failed_batches = 0

        while remaining > 0:
            try:
                batch, token = _fetch_batch(
                    app_id, self.lang, self.country, sort,
                    min(_REDUCED_BATCH_SIZE, remaining), token,
                )
            except Exception:
                failed_batches += 1
                if failed_batches >= 2:
                    logger.warning(
                        "GooglePlayScraper.get_reviews: Stopping after %d batch failures. "
                        "Collected %d reviews.",
                        failed_batches, len(collected),
                    )
                    break
                continue

            if not batch:
                logger.info(
                    "GooglePlayScraper.get_reviews: No more reviews. Collected %d total.",
                    len(collected),
                )
                break

            collected.extend(batch)
            remaining -= len(batch)
            failed_batches = 0  # Reset on success

            if token is None:
                logger.info(
                    "GooglePlayScraper.get_reviews: Token exhausted. Collected %d total.",
                    len(collected),
                )
                break

        return collected
