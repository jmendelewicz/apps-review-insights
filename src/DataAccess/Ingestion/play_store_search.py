from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Optional

from google_play_scraper import app as gplay_app, search as gplay_search

logger = logging.getLogger(__name__)

_PACKAGE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+){1,}$", re.IGNORECASE)


def extract_app_id(text: str) -> Optional[str]:
    if not text:
        return None
    s = text.strip()
    if "play.google.com" in s:
        try:
            params = urllib.parse.parse_qs(urllib.parse.urlparse(s).query)
            candidate = params.get("id", [""])[0]
            if candidate and _PACKAGE_ID_RE.match(candidate):
                return candidate
        except Exception:
            logger.debug("parse_qs failed for %r", s, exc_info=True)
        m = re.search(r"[?&]id=([A-Za-z0-9._]+)", s)
        if m:
            return m.group(1)
    if _PACKAGE_ID_RE.match(s):
        return s
    return None


def _format_app_data(data: dict) -> dict:
    """Normalize a google-play-scraper hit or app-info dict into our shape."""
    return {
        "appId": data.get("appId", ""),
        "title": data.get("title", ""),
        "icon": data.get("icon", ""),
        "developer": data.get("developer", ""),
        "score": float(data.get("score") or 0),
        "installs": data.get("installs", ""),
        "genre": data.get("genre", ""),
    }


def resolve_query(
    query: str,
    lang: str = "es",
    countries: Optional[list[str]] = None,
    n_hits: int = 10,
) -> list[dict]:
    countries = countries or ["us", "ar"]
    app_id = extract_app_id(query)
    if app_id:
        for country in countries:
            try:
                info = gplay_app(app_id, lang=lang, country=country)
                return [_format_app_data(info)]
            except Exception:
                logger.debug("gplay_app failed for %s in %s", app_id, country, exc_info=True)
                continue
        # Unknown app id — return a placeholder so the caller can still proceed.
        return [{"appId": app_id, "title": app_id, "icon": "", "developer": "",
                 "score": 0.0, "installs": "", "genre": ""}]

    seen: set[str] = set()
    results: list[dict] = []
    for country in countries:
        try:
            hits = gplay_search(query, lang=lang, country=country, n_hits=n_hits)
        except Exception:
            logger.debug("gplay_search failed in %s", country, exc_info=True)
            continue
        for h in hits:
            aid = h.get("appId", "")
            if aid and aid not in seen:
                seen.add(aid)
                results.append(_format_app_data(h))
        if len(results) >= n_hits:
            break
    return results[:n_hits]


def get_app_info(app_id: str, lang: str = "es", countries: Optional[list[str]] = None) -> Optional[dict]:
    countries = countries or ["us", "ar"]
    for country in countries:
        try:
            return _format_app_data(gplay_app(app_id, lang=lang, country=country))
        except Exception:
            logger.debug("get_app_info failed for %s in %s", app_id, country, exc_info=True)
            continue
    return None
