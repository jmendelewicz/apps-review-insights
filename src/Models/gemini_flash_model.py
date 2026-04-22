from enum import StrEnum
import json
import re
from typing import Any, Any, List

from google import genai

from src.Orchestration.review_stats import ReviewStats

GEMINI_MODEL = "gemini-2.5-flash"

class Sentiments(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class GeminiFlashModel:
    """
    Drop-in replacement for FlanT5LargeModel using Gemini.

    Contract:
        extract_top_features(reviews, sentiment, top_n) -> list[str]

    Responsibilities:
        - Prompt construction
        - API call
        - Output parsing

    Non-responsibilities:
        - Sampling
        - Sentiment filtering
        - DataFrame handling
    """

    MODEL_NAME = "gemini-2.5-flash"

    MAX_CHARS_PER_REVIEW = 512

    def __init__(self, api_key: str):
        print("Initializing GeminiFlashModel...")
        print("Starting Gemini client initialization...")
        self.client = genai.Client(api_key=api_key)

    def extract_top_features(
        self,
        reviews: List[str],
        sentiment: str,
        top_n: int
    ) -> List[str]:

        if not reviews:
            return [f"(No reviews found for sentiment '{sentiment}')"]

        prompt = self._build_summary_prompt(reviews, sentiment, top_n)

        try:
            response = self.client.models.generate_content(
                model=self.MODEL_NAME,
                contents=prompt,
            )

            raw = response.text.strip()

            return self._summary_parse_output(raw, top_n)

        except Exception as e:
            return [f"(Gemini error: {str(e)})"]

    # ─────────────────────────────────────────────
    # Prompts
    # ─────────────────────────────────────────────

    def _build_summary_prompt(
        self,
        reviews: List[str],
        sentiment: str,
        top_n: int
    ) -> str:

        sentiment_label = {
            Sentiments.POSITIVE: "POSITIVAS (elogiosas)",
            Sentiments.NEGATIVE: "NEGATIVAS (críticas)",
        }.get(sentiment, sentiment)

        # Keep truncation responsibility local (same as FLAN parity)
        joined = "\n".join(
            f"- {r[:self.MAX_CHARS_PER_REVIEW].replace('\n', ' ')}"
            for r in reviews
        )

        count = len(reviews)

        return f"""
            Sos un analista de producto experto.

            Analizá las siguientes {count} reseñas {sentiment_label} de una aplicación móvil (escritas en español).
            Las reseñas están ordenadas por cantidad de likes (las más votadas primero — esas representan opiniones más compartidas). 
            Cada reseña indica sus likes y la versión de la app.

            Tu tarea:
            1. Identificá los {top_n} aspectos o características MÁS MENCIONADOS.
            2. Agrupá menciones similares bajo un mismo aspecto.
            3. Priorizá aspectos que aparecen en reseñas con más likes.

            Respondé ÚNICAMENTE con un JSON válido:

            {{
              "aspectos": [
                {{"aspecto": "nombre del aspecto", "menciones_aprox": 15, "ejemplo_reseña": "texto breve de ejemplo"}},
                ...
              ]
            }}

            REGLAS:
            - Todo en ESPAÑOL
            - Aspectos concisos y específicos
            - Ordená de más mencionado a menos mencionado
            - No incluyas texto fuera del JSON

            RESEÑAS:
            {joined}"""
    
    

    # ─────────────────────────────────────────────
    # Parsing
    # ─────────────────────────────────────────────

    def _summary_parse_output(self, raw: str, top_n: int) -> List[str]:

        cleaned = self._summary_strip_code_fences(raw)

        try:
            data = json.loads(cleaned)
            aspectos = data.get("aspectos", [])

            if isinstance(aspectos, list) and aspectos:
                return [
                    self._summary_format_aspect(a)
                    for a in aspectos[:top_n]
                ]

        except json.JSONDecodeError:
            pass

        return self._fallback_parse(raw, top_n)

    def _summary_strip_code_fences(self, raw_text: str) -> str:
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)
        return raw_text.strip()

    
    def _summary_format_aspect(self, a: Any) -> str:
        if isinstance(a, dict):
            nombre = a.get("aspecto", "?")
            menciones = a.get("menciones_aprox", "?")
            ejemplo = a.get("ejemplo_reseña", "")
    
            line = f"{nombre} (~{menciones} menciones)"
    
            if ejemplo:
                ejemplo = ejemplo.strip()
                line += (
                    f' → "{ejemplo[:80]}..."'
                    if len(ejemplo) > 80
                    else f' → "{ejemplo}"'
                )
    
            return line
    
        return str(a).strip()
    
    
    def _summary_fallback_parse(self, raw: str, top_n: int) -> List[str]:

        lines = raw.splitlines()
        
        features = [
            re.sub(r"^[\d\.\)\-\•\–\s]+", "", l).strip()
            for l in lines
            if len(l.strip()) > 3
        ]

        return features[:top_n] if features else [raw]
    

    # ─────────────────────────────────────────────
    # FODA
    # ─────────────────────────────────────────────


    def _build_foda_prompt(self,
        stats: ReviewStats,
        praised: list[str],
        criticized: list[str],
        version_analysis: str,
        neutral_sample: str,
    ) -> str:
        return f"""Sos un consultor estratégico de producto digital. Con base en el análisis de {stats.total} reseñas de una aplicación móvil, generá un análisis FODA (SWOT) completo.
    
    DATOS DE ENTRADA:
    
    Distribución de sentimiento:
    - Positivas: {stats.pct_pos:.1f}% ({stats.count_pos} reseñas)
    - Negativas: {stats.pct_neg:.1f}% ({stats.count_neg} reseñas)
    - Neutrales: {stats.pct_neu:.1f}% ({stats.count_neu} reseñas)
    - Score promedio: {stats.avg_score:.2f}/5 ⭐
    - Likes promedio por reseña: {stats.avg_likes:.1f}
    
    🟢 Aspectos más elogiados:
    {chr(10).join(f'  {i+1}. {a}' for i, a in enumerate(praised))}
    
    🔴 Aspectos más criticados:
    {chr(10).join(f'  {i+1}. {a}' for i, a in enumerate(criticized))}
    
    📱 Análisis por versión de la app:
    {version_analysis}
    
    😐 Muestra de reseñas neutrales (posibles sugerencias):
    {neutral_sample}
    
    INSTRUCCIONES:
    Generá un JSON con la siguiente estructura exacta:
    
    {{
      "fortalezas": [
        {{"punto": "descripción concisa", "evidencia": "dato o reseña que lo respalda"}},
        ...
      ],
      "oportunidades": [
        {{"punto": "descripción concisa", "evidencia": "dato o reseña que lo respalda"}},
        ...
      ],
      "debilidades": [
        {{"punto": "descripción concisa", "evidencia": "dato o reseña que lo respalda"}},
        ...
      ],
      "amenazas": [
        {{"punto": "descripción concisa", "evidencia": "dato o reseña que lo respalda"}},
        ...
      ],
      "resumen_ejecutivo": "Párrafo breve (3-4 oraciones) con la conclusión general del análisis."
    }}
    
    REGLAS:
    - Entre 3 y 5 puntos por categoría.
    - Todo en ESPAÑOL.
    - Las Oportunidades deben derivarse de las sugerencias y necesidades no cubiertas de los usuarios.
    - Las Amenazas deben inferirse de tendencias negativas (ej: si una actualización reciente genera muchas quejas, eso es una amenaza de churn).
    - Incluí evidencia concreta (citas de reseñas, porcentajes, datos de versiones).
    - No incluyas texto fuera del JSON.
    """
    
    
    # ── Llamado a la API y formateo de respuesta ───────────────────────────────────
    
    def call_foda_api(self, prompt: str) -> dict:
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
    
            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)
    
            return json.loads(raw_text)
    
        except json.JSONDecodeError:
            print("  ⚠ Gemini no devolvió JSON válido para el FODA.")
            return {"error": response.text[:500]}
    
        except Exception as e:
            return {"error": str(e)}
    
    
    # ── Orquestador (mantiene la firma pública original) ───────────────────────────
    
    def generate_foda(
        self,
        praised: list[str],
        neutral: list[str],
        criticized: list[str],
        version_analysis: str,
        stats: ReviewStats,
    ) -> dict:
        """
        Genera un análisis FODA completo usando Gemini, alimentado por
        los aspectos ya extraídos, la data de versiones, y las reseñas.
        """
        neutral_sample = "\n".join(f"- {r[:200]}" for r in neutral[:30])
    
        prompt = self._build_foda_prompt(stats, praised, criticized, version_analysis, neutral_sample)
    
        foda_call = self.call_foda_api(prompt)

        return foda_call