from enum import StrEnum

import pandas as pd
import re

# Columnas esperadas del CSV (google-play-scraper)
class review_columns(StrEnum):
    CONTENT_COL = "content"
    SCORE_COL   = "score"
    LIKES_COL    = "thumbsUpCount"
    DATE_COL     = "at"
    VERSION_COL  = "reviewCreatedVersion"
    SENTIMENT_COL = "sentiment"  # columna adicional para análisis de sentimiento

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Module-level function used by the Streamlit frontend.
    Filters short/empty reviews and coerces column types without stripping
    display characters (preserves emojis and punctuation for UI rendering).
    """
    df = df.dropna(subset=[review_columns.CONTENT_COL])
    df[review_columns.CONTENT_COL] = df[review_columns.CONTENT_COL].astype(str).str.strip()
    df = df[df[review_columns.CONTENT_COL].str.len() > 5]
    if review_columns.LIKES_COL in df.columns:
        df[review_columns.LIKES_COL] = (
            pd.to_numeric(df[review_columns.LIKES_COL], errors="coerce").fillna(0).astype(int)
        )
    if review_columns.SCORE_COL in df.columns:
        df[review_columns.SCORE_COL] = (
            pd.to_numeric(df[review_columns.SCORE_COL], errors="coerce").fillna(3).astype(int)
        )
    if review_columns.DATE_COL in df.columns:
        df[review_columns.DATE_COL] = pd.to_datetime(df[review_columns.DATE_COL], errors="coerce")
    return df.reset_index(drop=True)


class GoogleReviewCleaner:
    """Clase para limpiar y validar reseñas de Google Play."""

    @staticmethod
    def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia el DataFrame de reseñas de Google Play.
        - Elimina filas sin texto o con texto muy corto.
        - Elimina caracteres no deseados (ej. emojis, saltos de línea).
        - Normaliza espacios.
        """
        # Eliminar filas sin texto o con texto muy corto
        reviews = reviews.dropna(subset=[review_columns.CONTENT_COL])
        reviews[review_columns.CONTENT_COL] = reviews[review_columns.CONTENT_COL].astype(str).str.strip()
        reviews = reviews[reviews[review_columns.CONTENT_COL].str.len() > 5]

        # Limpiar texto: eliminar emojis, saltos de línea, etc.
        def clean_text(text):
            text = re.sub(r'\s+', ' ', text)  # Normalizar espacios
            text = re.sub(r'[^\w\s.,!?]', '', text)  # Eliminar caracteres no deseados
            return text.strip()

        reviews[review_columns.CONTENT_COL] = reviews[review_columns.CONTENT_COL].apply(clean_text)
        
        return reviews

    @staticmethod
    def validate_schema(reviews: pd.DataFrame) -> bool:
        """Valida que el DataFrame tenga las columnas necesarias."""
        error_messages = []

        if review_columns.CONTENT_COL not in reviews.columns:
            error_messages.append(f"Error: No se encontró la columna '{review_columns.CONTENT_COL}' en el DataFrame.")
        if review_columns.SCORE_COL not in reviews.columns:
            error_messages.append(f"Error: No se encontró la columna '{review_columns.SCORE_COL}' en el DataFrame.")
        if review_columns.LIKES_COL not in reviews.columns:
            error_messages.append(f"Error: No se encontró la columna '{review_columns.LIKES_COL}' en el DataFrame.")

        for msg in error_messages:
            print(msg)  # TODO reemplazar por logger

        return len(error_messages) == 0

    @staticmethod
    def coerce_types(reviews: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica coerciones de tipo:
        - likes → int (default 0)
        - score → int (default 3)
        - date → datetime
        """
        if review_columns.LIKES_COL in reviews.columns:
            reviews[review_columns.LIKES_COL] = (
                pd.to_numeric(reviews[review_columns.LIKES_COL], errors="coerce")
                .fillna(0)
                .astype(int)
            )

        if review_columns.SCORE_COL in reviews.columns:
            reviews[review_columns.SCORE_COL] = (
                pd.to_numeric(reviews[review_columns.SCORE_COL], errors="coerce")
                .fillna(3)
                .astype(int)
            )

        if review_columns.DATE_COL in reviews.columns:
            reviews[review_columns.DATE_COL] = pd.to_datetime(
                reviews[review_columns.DATE_COL], errors="coerce"
            )

        return reviews

    @staticmethod
    def prepare_version_data(df: pd.DataFrame) -> pd.DataFrame:
        """Filtra filas válidas para análisis por versión."""
        if review_columns.VERSION_COL not in df.columns:
            return pd.DataFrame()

        df_v = df.dropna(subset=[review_columns.VERSION_COL])
        df_v = df_v[df_v[review_columns.VERSION_COL].astype(str).str.strip() != ""]

        return df_v
    
    
    @staticmethod
    def group_version(df_v: pd.DataFrame) -> pd.DataFrame:
        """Calcula métricas agregadas por versión."""
        if df_v.empty:
            return pd.DataFrame()

        stats = (
            df_v.groupby(review_columns.VERSION_COL)
            .agg(
                total=("sentiment", "size"),
                positivas=("sentiment", lambda x: (x == "positive").sum()),
                negativas=("sentiment", lambda x: (x == "negative").sum()),
                neutrales=("sentiment", lambda x: (x == "neutral").sum()),
                score_promedio=(
                    (review_columns.SCORE_COL, "mean")
                    if review_columns.SCORE_COL in df_v.columns
                    else ("sentiment", "size")  # fallback dummy
                ),
            )
            .sort_values("total", ascending=False)
        )

        return stats


    @staticmethod
    def format_version_report(stats: pd.DataFrame, top_n: int = 10) -> str:
        """Convierte métricas en tabla formateada."""
        if stats.empty:
            return "(No hay versiones con suficientes reseñas para analizar)"

        lines = []
        lines.append(f"  {'Versión':<16} {'Total':>6} {'👍 Pos':>7} {'👎 Neg':>7} {'😐 Neu':>7} {'Score ⭐':>8}")
        lines.append("  " + "─" * 55)

        for version, row in stats.head(top_n).iterrows():
            pct_neg = row["negativas"] / row["total"] * 100
            flag = " ⚠️" if pct_neg > 40 else ""

            score_str = (
                f"{row['score_promedio']:.1f}"
                if "score_promedio" in row
                else "N/A"
            )

            lines.append(
                f"  {str(version):<16} {int(row['total']):>6} "
                f"{int(row['positivas']):>7} {int(row['negativas']):>7} "
                f"{int(row['neutrales']):>7} {score_str:>8}{flag}"
            )

        return "\n".join(lines)
    
    @staticmethod
    def filter_versions(stats: pd.DataFrame, min_reviews: int = 5) -> pd.DataFrame:
        """Aplica reglas de negocio (mínimo de reseñas)."""
        if stats.empty:
            return stats

        return stats[stats["total"] >= min_reviews]