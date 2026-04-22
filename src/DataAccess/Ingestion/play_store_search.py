from __future__ import annotations

import re
import urllib.parse
from typing import Optional

from google_play_scraper import app as gplay_app, search as gplay_search


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
            pass
        m = re.search(r"[?&]id=([A-Za-z0-9._]+)", s)
        if m:
            return m.group(1)
    if _PACKAGE_ID_RE.match(s):
        return s
    return None


def _format_hit(hit: dict) -> dict:
    return {
        "appId": hit.get("appId", ""),
        "title": hit.get("title", ""),
        "icon": hit.get("icon", ""),
        "developer": hit.get("developer", ""),
        "score": float(hit.get("score") or 0),
        "installs": hit.get("installs", ""),
        "genre": hit.get("genre", ""),
    }


def _format_info(info: dict) -> dict:
    return {
        "appId": info.get("appId", ""),
        "title": info.get("title", ""),
        "icon": info.get("icon", ""),
        "developer": info.get("developer", ""),
        "score": float(info.get("score") or 0),
        "installs": info.get("installs", ""),
        "genre": info.get("genre", ""),
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
                return [_format_info(info)]
            except Exception:
                continue
        return [{"appId": app_id, "title": app_id, "icon": "", "developer": "",
                 "score": 0.0, "installs": "", "genre": ""}]

    seen: set[str] = set()
    results: list[dict] = []
    for country in countries:
        try:
            hits = gplay_search(query, lang=lang, country=country, n_hits=n_hits)
        except Exception:
            continue
        for h in hits:
            aid = h.get("appId", "")
            if aid and aid not in seen:
                seen.add(aid)
                results.append(_format_hit(h))
        if len(results) >= n_hits:
            break
    return results[:n_hits]


def get_app_info(app_id: str, lang: str = "es", countries: Optional[list[str]] = None) -> Optional[dict]:
    countries = countries or ["us", "ar"]
    for country in countries:
        try:
            return _format_info(gplay_app(app_id, lang=lang, country=country))
        except Exception:
            continue
    return None
