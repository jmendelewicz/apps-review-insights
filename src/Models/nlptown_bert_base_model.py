from transformers import pipeline

SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
STAR_TO_SENTIMENT = {
    "1 star":  "negative",
    "2 stars": "negative",
    "3 stars": "neutral",
    "4 stars": "positive",
    "5 stars": "positive",
}

class BertBaseSentimentModel:

    def __init__(self):
        print(f"Cargando modelo de sentimiento: {SENTIMENT_MODEL}")
        self.pipeline = pipeline(
            "text-classification",
            model=SENTIMENT_MODEL,
            truncation=True,
            max_length=512,
        )

    def classify_sentiment(self, content: list[str]) -> list[str]:
        """Clasifica sentimiento con nlptown (multilingüe)."""
        labels: list[str] = []
        batch_size = 32
        total = len(content)

        for i in range(0, total, batch_size):
            batch = content[i : i + batch_size]
            results = self.pipeline(batch)
            for r in results:
                star_label = r["label"].lower()
                sentiment = STAR_TO_SENTIMENT.get(star_label, "neutral")
                labels.append(sentiment)

        return labels