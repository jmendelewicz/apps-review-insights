from __future__ import annotations

# Google Play Scraper column names — used by the Streamlit frontend.
# The API layer uses the equivalent review_columns StrEnum defined in
# DataAccess/Refinary/google_review_cleaner.py; both map to the same strings.
CONTENT_COL = "content"
SCORE_COL = "score"
DATE_COL = "at"
LIKES_COL = "thumbsUpCount"
VERSION_COL = "reviewCreatedVersion"
