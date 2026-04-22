# import datetime
# from typing import List

from dataclasses import dataclass
import pandas as pd

# from src.Orchestration.path_helper import PathHelper, PipelineStep
from src.DataAccess.Ingestion.google_play_wrapper import GooglePlayScraper
from src.DataAccess.Refinary.google_review_cleaner import GoogleReviewCleaner, review_columns
# from src.DataAccess.Storage.csv_review_repository import CSVReviewRepository
from src.Models.gemini_flash_model import GeminiFlashModel, Sentiments
from src.Models.nlptown_bert_base_model import BertBaseSentimentModel
from src.Orchestration.review_stats import ReviewStats


# Cuántas reseñas mandarle al modelo de resumen (no saturar?)
MAX_REVIEWS_FOR_SUMMARY = 80

@dataclass
class FeatureSummary:
    positive: list[str]
    negative: list[str]

@dataclass 
class FodaAnalysis:
    fortalezas: list[str]
    oportunidades: list[str]
    debilidades: list[str]
    amenazas: list[str]
    resumen_ejecutivo: str

@dataclass
class PipelineResult:
    app_id: str
    limit: int
    summary: FeatureSummary
    foda: FodaAnalysis
    stats: ReviewStats


class GooglePlayService:

    def __init__(self, 
                 scraper: GooglePlayScraper,
                 cleaner: GoogleReviewCleaner,
                 sentiment_model: BertBaseSentimentModel,
                 summarization_model: GeminiFlashModel):
        self.scraper = scraper
        self.cleaner = cleaner
        self.sentiment_model = sentiment_model
        self.summarization_model = summarization_model
    

    def run_pipeline(self, app_id: str, limit=1000) -> PipelineResult:

        reviews_df = pd.DataFrame(self.scraper.get_reviews(app_id, limit))
        reviews_df = self.cleaner.clean_reviews(reviews_df)

        sentiments = self.sentiment_model.classify_sentiment(reviews_df[review_columns.CONTENT_COL].tolist())
        reviews_df[review_columns.SENTIMENT_COL] = sentiments

        stats = ReviewStats.from_dataframe(reviews_df)

        #────────────────── SUMMARY ──────────────────
        
        summary = self.generate_summarey(reviews_df)
        
        #────────────────── FODA ──────────────────

        if any("Gemini error" in feature for feature in summary.positive) or any("Gemini error" in feature for feature in summary.negative):
            foda_analysis = FodaAnalysis(
                fortalezas = ["No se pudieron analizar las fortalezas debido a un error en el modelo de resumen."],
                debilidades = ["No se pudieron analizar las debilidades debido a un error en el modelo de resumen."],
                oportunidades = ["No se pudieron analizar las oportunidades debido a un error en el modelo de resumen."],
                amenazas = ["No se pudieron analizar las amenazas debido a un error en el modelo de resumen."],
                resumen_ejecutivo = "No se pudo generar el resumen ejecutivo debido a un error en el modelo de resumen.")
        else:
            foda_analysis = self.generate_foda(reviews_df, 
                                               summary.positive, 
                                               summary.negative, 
                                               stats)
        #────────────────── RESULT ──────────────────

        result = self.format_pipeline_result(
            app_id=app_id,  
            limit=limit,
            summary=summary,
            foda=foda_analysis,
            stats=stats)

        return result

    def generate_summarey(self, reviews_df: pd.DataFrame) -> FeatureSummary:
        weighted_positive_reviews = self.prepare_weighted_reviews(reviews_df, 
                                                                  Sentiments.POSITIVE.value)
        positive_features = self.summarization_model.extract_top_features(weighted_positive_reviews,
                                                                          sentiment = Sentiments.POSITIVE.value,
                                                                          top_n=5)
        
        weighted_negative_reviews = self.prepare_weighted_reviews(reviews_df,
                                                                  Sentiments.NEGATIVE.value)
        negative_features = self.summarization_model.extract_top_features(weighted_negative_reviews,
                                                                          sentiment = Sentiments.NEGATIVE.value,
                                                                          top_n=5)
        
        return FeatureSummary(positive=positive_features, negative=negative_features)

    def generate_foda(self, 
                      reviews_df: pd.DataFrame, 
                      positive_features: list, 
                      negative_features: list, 
                      stats: ReviewStats) -> FodaAnalysis:

        neutral_features = reviews_df[reviews_df[review_columns.SENTIMENT_COL] == "neutral"][review_columns.CONTENT_COL].tolist()

        version_analysis = self.analyze_by_version(reviews_df)

        foda_result = self.summarization_model.generate_foda(
            praised = positive_features,
            neutral = neutral_features,
            criticized = negative_features,
            version_analysis = version_analysis,
            stats = stats)

        return FodaAnalysis(fortalezas = foda_result.get("fortalezas", []),
                            debilidades = foda_result.get("debilidades", []),
                            oportunidades = foda_result.get("oportunidades", []),
                            amenazas = foda_result.get("amenazas", []),
                            resumen_ejecutivo = foda_result.get("resumen_ejecutivo", ""))


    def prepare_weighted_reviews(self,
                                 df: pd.DataFrame,
                                 sentiment: str) -> str:

        subset = df[df[review_columns.SENTIMENT_COL] == sentiment]

        if subset.empty:
            return ""

        if review_columns.LIKES_COL in subset.columns:
            subset = subset.sort_values(review_columns.LIKES_COL, ascending=False)

        subset = subset.head(MAX_REVIEWS_FOR_SUMMARY) # limitamos para no saturar el modelo

        lines = []
        for idx, (_, row) in enumerate(subset.iterrows()):
            text = str(row[review_columns.CONTENT_COL])[:500]
            likes = int(row.get(review_columns.LIKES_COL, 0))
            version = row.get(review_columns.VERSION_COL, "?")

            if likes > 0:
                lines.append(f"{idx+1}. [👍{likes} likes | v{version}] {text}")
            else:
                lines.append(f"{idx+1}. [v{version}] {text}")

        return "\n".join(lines)

    
    def analyze_by_version(self, df: pd.DataFrame) -> str:
        """Orquesta el análisis por versión."""
        df_v = self.cleaner.prepare_version_data(df)

        if df_v.empty:
            return "(No hay datos de versión)"

        stats = self.cleaner.group_version(df_v)
        stats = self.cleaner.filter_versions(stats)

        return self.cleaner.format_version_report(stats)
    

    def format_pipeline_result(self, app_id: str, limit: int, summary: FeatureSummary, foda: FodaAnalysis, stats: ReviewStats) -> PipelineResult:

        if any("Gemini error" in feature for feature in summary.positive):
            summary.positive = ["No se pudieron extraer aspectos positivos debido a un error del modelo."]

        if any("Gemini error" in feature for feature in summary.negative):
            summary.negative = ["No se pudieron extraer aspectos negativos debido a un error del modelo."]

        return PipelineResult(
            app_id=app_id,
            limit=limit,
            summary=summary,
            foda=foda,
            stats=stats
        )