from __future__ import annotations

import json
import re
import time
from typing import Optional

import pandas as pd

from constants import CONTENT_COL, SCORE_COL

GEMINI_MODEL = "gemini-2.5-flash"


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text).strip("`").strip()
    return text


def extract_topics_gemini(
    df: pd.DataFrame,
    client,
    app_name: str,
    n: int = 8,
) -> list[dict]:
    sample = df[df[CONTENT_COL].notna()].head(150)
    reviews_text = "\n---\n".join(
        f"Rating: {r.get(SCORE_COL, 3)}/5 | {str(r.get(CONTENT_COL, ''))[:150]}"
        for _, r in sample.iterrows()
    )
    prompt = (
        f'Analiza las siguientes reseñas de la app "{app_name}" y extrae los {n} temas '
        f"principales que mencionan los usuarios.\n\n"
        f"Para cada tema proporciona:\n"
        f'- "topic": nombre del tema (máximo 3 palabras, en español)\n'
        f'- "count": número estimado de reseñas que lo mencionan\n'
        f'- "sentiment": sentimiento predominante ("positive", "negative" o "neutral")\n\n'
        f"Responde ÚNICAMENTE con un array JSON válido. Sin markdown, sin texto extra.\n"
        f'Ejemplo: [{{"topic": "Diseño de interfaz", "count": 45, "sentiment": "positive"}}]\n\n'
        f"Reseñas:\n{reviews_text}"
    )
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        topics = json.loads(_strip_json_fences(resp.text))
        return topics[:10] if isinstance(topics, list) else []
    except Exception:
        return []


def generate_deep_analysis(
    df: pd.DataFrame,
    client,
    app_name: str,
    topics: list[dict],
) -> str:
    counts = df["sentiment"].value_counts()
    total = len(df)
    n_pos = counts.get("positive", 0)
    n_neg = counts.get("negative", 0)
    n_neu = counts.get("neutral", 0)
    avg = df[SCORE_COL].mean() if SCORE_COL in df.columns else 0
    topics_text = ", ".join(t.get("topic", "") for t in topics[:6])

    pos_sample = df[df["sentiment"] == "positive"].head(15)
    neg_sample = df[df["sentiment"] == "negative"].head(15)
    pos_text = "\n".join(
        f"- [{r.get(SCORE_COL, 0)}★] {str(r.get(CONTENT_COL, ''))[:200]}"
        for _, r in pos_sample.iterrows()
    )
    neg_text = "\n".join(
        f"- [{r.get(SCORE_COL, 0)}★] {str(r.get(CONTENT_COL, ''))[:200]}"
        for _, r in neg_sample.iterrows()
    )

    prompt = (
        f'Realiza un análisis profundo de las reseñas de la app "{app_name}" en Google Play Store.\n\n'
        f"## Datos clave:\n"
        f"- Reseñas analizadas: {total}\n"
        f"- Rating promedio: {avg:.1f}/5\n"
        f"- Sentimiento: {n_pos / total * 100:.0f}% positivo, "
        f"{n_neg / total * 100:.0f}% negativo, {n_neu / total * 100:.0f}% neutro\n"
        f"- Temas principales: {topics_text}\n\n"
        f"## Muestra de reseñas POSITIVAS:\n{pos_text}\n\n"
        f"## Muestra de reseñas NEGATIVAS:\n{neg_text}\n\n"
        f"Proporciona un análisis estructurado con estas secciones usando formato markdown:\n\n"
        f"## Resumen Ejecutivo\n(2-3 párrafos con el estado general)\n\n"
        f"## Fortalezas Principales\n(lista de puntos)\n\n"
        f"## Problemas Críticos\n(lista con impacto)\n\n"
        f"## Perfil del Usuario\n(qué tipo de usuarios)\n\n"
        f"## Recomendaciones de Mejora\n(acciones concretas priorizadas)\n\n"
        f"Responde en español. Sé específico y accionable."
    )

    for attempt in range(3):
        try:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return resp.text
        except Exception as e:
            if "503" in str(e) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            return f"Error al generar análisis: {str(e)[:200]}"
    return ""
