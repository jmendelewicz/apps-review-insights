from dataclasses import dataclass
from src.DataAccess.Refinary.google_review_cleaner import review_columns
import pandas as pd

@dataclass
class ReviewStats:
    total: int
    pct_pos: float
    pct_neg: float
    pct_neu: float
    count_pos: int
    count_neg: int
    count_neu: int
    avg_score: float | None
    avg_likes: float | None

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "ReviewStats":
        def to_int(x):
            return int(x) if x is not None else 0
    
        def to_float(x):
            return float(x) if x is not None else 0.0
        
        total = int(len(df))
        counts = df["sentiment"].value_counts()

        count_pos = to_int(counts.get("positive", 0))
        count_neg = to_int(counts.get("negative", 0))
        count_neu = to_int(counts.get("neutral", 0))

        pct_pos = to_float(count_pos / total * 100) if total > 0 else 0.0
        pct_neg = to_float(count_neg / total * 100) if total > 0 else 0.0
        pct_neu = to_float(count_neu / total * 100) if total > 0 else 0.0

        avg_score = (
            to_float(df[review_columns.SCORE_COL].mean())
            if review_columns.SCORE_COL in df.columns and total > 0
            else None
        )

        avg_likes = (
            to_float(df[review_columns.LIKES_COL].mean())
            if review_columns.LIKES_COL in df.columns and total > 0
            else None
        )

        return cls(
            total=total,
            pct_pos=pct_pos,
            pct_neg=pct_neg,
            pct_neu=pct_neu,
            count_pos=count_pos,
            count_neg=count_neg,
            count_neu=count_neu,
            avg_score=avg_score,
            avg_likes=avg_likes,
        )
    