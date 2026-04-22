from __future__ import annotations

import collections
import re

import pandas as pd

from constants import CONTENT_COL

STOPWORDS: set[str] = {
    "de", "la", "el", "en", "y", "a", "que", "es", "un", "una", "los", "las", "por",
    "con", "para", "del", "al", "se", "no", "lo", "le", "su", "me", "más", "pero",
    "muy", "ya", "si", "mi", "como", "sin", "todo", "esta", "este", "eso", "esto",
    "hay", "les", "nos", "te", "tan", "he", "ha", "ser", "son", "era", "tiene",
    "bien", "hace", "solo", "cada", "donde", "cuando", "entre", "después",
    "the", "is", "it", "to", "and", "of", "in", "for", "on", "my", "i", "this",
    "that", "you", "not", "are", "was", "but", "have", "has", "app", "its", "can",
    "so", "from", "with", "all", "an", "be", "been",
}


def get_word_freq(df: pd.DataFrame, n: int = 60) -> list[tuple[str, int]]:
    text = " ".join(df[CONTENT_COL].dropna().astype(str).str.lower())
    words = [
        w for w in re.findall(r"[a-záéíóúñü]+", text)
        if len(w) > 3 and w not in STOPWORDS
    ]
    return collections.Counter(words).most_common(n)


def build_wordcloud_html(word_freq: list[tuple[str, int]]) -> str:
    if not word_freq:
        return ""
    max_c = word_freq[0][1]
    min_c = word_freq[-1][1] if len(word_freq) > 1 else max_c
    spans: list[str] = []
    for word, count in word_freq:
        ratio = (count - min_c) / (max_c - min_c) if max_c != min_c else 0.5
        size = round(12 + ratio * 30)
        opacity = round(0.45 + ratio * 0.55, 2)
        weight = "700" if ratio > 0.7 else "600" if ratio > 0.4 else "500"
        spans.append(
            f'<span style="font-size:{size}px;opacity:{opacity};font-weight:{weight}" '
            f'title="{word}: {count}">{word}</span>'
        )
    return f'<div class="html-wordcloud">{"".join(spans)}</div>'
