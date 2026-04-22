from typing import Dict, List, Optional

import pandas as pd
from google_play_scraper import reviews, Sort


def scrape_reviews(
    app_id: str,
    lang: str,
    country: str,
    count: int,
    sort_label: str,
    country_name: str = "",
) -> pd.DataFrame:
    """
    Standalone function for the Streamlit frontend.
    Scrapes up to `count` reviews in a single call, using pagination tokens.
    """
    sort_order = Sort.MOST_RELEVANT if sort_label == "Más relevantes" else Sort.NEWEST
    batch_size = 200
    collected: List[Dict] = []
    token: Optional[str] = None
    while len(collected) < count:
        remaining = count - len(collected)
        try:
            batch, token = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=sort_order,
                count=min(batch_size, remaining),
                continuation_token=token,
            )
        except Exception:
            break
        if not batch:
            break
        collected.extend(batch)
        if not token:
            break
    df = pd.DataFrame(collected[:count])
    if not df.empty and country_name:
        df["country"] = country_name
    return df


class GooglePlayScraper:

    def __init__(self, lang: str = "es", country: str = "ar"):
        print("Initializing GooglePlayScraper with lang =", lang, "and country =", country)
        self.lang = lang
        self.country = country
        print(f"GooglePlayScraper initialized with lang='{self.lang}' and country='{self.country}'")

    def get_reviews(
        self,
        app_id: str,
        limit: int = 1000,
        sort: Sort = Sort.NEWEST
    ) -> List[Dict]:

        collected: List[Dict] = []
        token: Optional[str] = None
        remaining = limit

        while remaining > 0:

            batch, token = reviews(
                app_id,
                lang=self.lang,
                country=self.country,
                sort=sort,
                count=min(200, remaining),
                continuation_token=token
            )

            # Early exit if API returns nothing
            if not batch:
                break

            collected.extend(batch)
            remaining -= len(batch)

            # Early exit if no continuation token
            if token is None:
                break

        return collected