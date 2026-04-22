# Discovery: Modelos para Opinion Mining / Sentiment Modeling

## Métricas de éxito

1. Clasificar correctamente un porcentaje representativo de reseñas en polaridades (positiva, neutra, negativa).
2. Producir un resumen condensado y jerarquizado de las características más mencionadas.

## Fuente de datos

Se utiliza la biblioteca [`google-play-scraper`](https://pypi.org/project/google-play-scraper/) para extraer reseñas. El output es un CSV con los siguientes campos relevantes:

| Campo | Descripción |
|---|---|
| `content` | Texto de la reseña |
| `score` | Puntaje del usuario (1–5 estrellas) |
| `thumbsUpCount` | Cantidad de likes de la reseña |
| `reviewCreatedVersion` | Versión de la app al momento de la reseña |
| `at` | Fecha de la reseña |

Para este sprint, los campos utilizados son `content` (texto) y `score` (referencia de validación).

## Modelos seleccionados

### 1. `cardiffnlp/twitter-roberta-base-sentiment-latest` — Clasificación de sentimiento

**¿Qué es?**
RoBERTa-base fine-tuneado sobre ~124 millones de tweets para clasificación de sentimiento en tres clases: `positive`, `neutral`, `negative`.

**¿Por qué este modelo?**
- Gratuito y disponible en Hugging Face sin restricciones.
- Entrenado sobre lenguaje informal y coloquial, que es el mismo registro de las reseñas de app stores.
- Liviano: corre en CPU estándar con tiempos razonables.
- No requiere fine-tuning adicional para una primera prueba.

**Limitación conocida:**
No extrae aspectos específicos (qué parte de la app es buena o mala), solo la polaridad global de cada reseña. La extracción de aspectos se delega al modelo de resumen.

### 2. `google/flan-t5-large` — Extracción de características y resumen

**¿Qué es?**
Flan-T5 es la versión instruction-tuned de T5 (Text-to-Text Transfer Transformer) de Google. La variante `large` tiene 780M de parámetros y fue entrenada con cientos de tareas en lenguaje natural, lo que le permite seguir instrucciones complejas sin fine-tuning adicional.

**¿Por qué este modelo?**
- Gratuito y open source (licencia Apache 2.0).
- Puede recibir un prompt con múltiples reseñas y devolver una lista estructurada de aspectos frecuentes.
- La variante `large` ofrece razonamiento significativamente mejor que `base` para extracción de información, manteniendo un peso manejable (~3GB).
- No requiere GPU: puede correr en CPU, aunque una GPU modesta (T4 en Colab, por ejemplo) acelera notablemente la inferencia.
- Compatible con el deploy en Hugging Face Spaces.

**¿Alternativas más potentes a probar?**

- GPT-4 / Claude API (costo por token, no gratuito)
- `flan-t5-xl` / `flan-t5-xxl` (requieren GPU de alta gama, inviable en etapa de prueba)
- ABSA fine-tuneado (PyABSA/DeBERTa) (requiere dataset etiquetado propio; overkill para Sprint 1)
- LLaMA / Mistral locales (peso >7GB, requieren infraestructura que aún no tenemos)

## Arquitectura del script (Sprint 1)

```
CSV (google-play-scraper)
        │
        ▼
  Limpieza y carga (pandas)
        │
        ▼
  Clasificación de sentimiento
  (cardiffnlp/twitter-roberta)
        │
        ├──── reseñas positivas ──▶  flan-t5-large ──▶ Top N características elogiadas
        │
        └──── reseñas negativas ──▶  flan-t5-large ──▶ Top N características criticadas
        │
        ▼
  Output en consola + CSV enriquecido con columna `sentiment`
```

---

## Cómo usar el script

### Instalación de dependencias

```bash
pip install transformers torch pandas google-play-scraper
```

### Ejecución

```bash
# Análisis básico (top 3 por defecto)
python analyze_reviews.py --csv reviews.csv

# Personalizar cantidad de features
python analyze_reviews.py --csv reviews.csv --top 5
```

### Output esperado

```
── Distribución de sentimiento ──────────────────
  positive    3241 reseñas  (64.8%)
  negative    1102 reseñas  (22.0%)
  neutral      659 reseñas  (13.2%)

Top 3 características MÁS ELOGIADAS:
  1. Smooth gameplay and controls
  2. Regular content updates
  3. Good graphics quality

Top 3 características MÁS CRITICADAS:
  1. Excessive ads interrupting gameplay
  2. Pay-to-win mechanics
  3. Frequent crashes on older devices
```

---

## Próximos pasos

- **Fine-tuning con datos propios:** una vez que tengamos un dataset etiquetado de reseñas de videojuegos, se puede hacer fine-tuning de flan-t5 o migrar a un modelo ABSA dedicado para extracción de aspectos más precisa.
- **Factor de peso por reiterancia:** incorporar `thumbsUpCount` para ponderar las reseñas más votadas.

---

## Referencias

- [google-play-scraper PyPI](https://pypi.org/project/google-play-scraper/)
- [cardiffnlp/twitter-roberta-base-sentiment-latest – Hugging Face](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [google/flan-t5-large – Hugging Face](https://huggingface.co/google/flan-t5-large)
- [Flan: Finetuned Language Models are Zero-Shot Learners (Google, 2022)](https://arxiv.org/abs/2109.01652)