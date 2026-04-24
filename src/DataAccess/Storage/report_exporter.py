from __future__ import annotations

import re
import unicodedata

from fpdf import FPDF


def safe_filename(name: str) -> str:
    """Slugify a name while preserving readability.

    NFKD-normalize accents to their ASCII equivalents (á → a) instead of
    collapsing them into underscores, then replace anything that's not
    alphanumeric/dash/underscore with underscores. This keeps filenames
    readable for apps like "Mercadolibre México".
    """
    if not name:
        return "app"
    normalized = unicodedata.normalize("NFKD", str(name))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_only).strip("_")
    return slug or "app"


def _to_latin1(text: str) -> str:
    """Best-effort encoding for FPDF core fonts (Helvetica → Latin-1 only).

    Drops characters outside Latin-1 (emojis, CJK, etc.) but *preserves*
    accented letters (á, é, ñ, ü) which are in Latin-1.
    """
    return text.encode("latin-1", errors="replace").decode("latin-1").replace("?", " ")


_BULLET_RE = re.compile(r"^\s*[-*•·]\s+")
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+")


def build_pdf_bytes(app_name: str, ai_analysis: str) -> bytes:
    """Render a Markdown-ish analysis into a structured PDF.

    Handles:
    - H1 (`# `) and H2 (`## `, `### `) as bold section headers.
    - Bullet lines (`- `, `* `, `• `) and numbered lists as indented items.
    - Bold (`**...**`) and inline code (`` `...` ``) stripped to plain text.
    - Blank lines as paragraph breaks.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Helvetica", style="B", size=18)
    pdf.multi_cell(0, 9, _to_latin1(f"Análisis — {app_name}"))
    pdf.ln(3)
    pdf.set_draw_color(124, 58, 237)
    pdf.set_line_width(0.4)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(4)

    for raw_line in (ai_analysis or "").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            pdf.ln(3)
            continue

        # Strip inline markdown markers we don't render (keep text).
        stripped = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        stripped = re.sub(r"`([^`]+)`", r"\1", stripped)

        if stripped.startswith("# "):
            pdf.ln(2)
            pdf.set_font("Helvetica", style="B", size=15)
            pdf.set_text_color(91, 33, 182)
            pdf.multi_cell(0, 8, _to_latin1(stripped[2:].strip()))
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)
        elif stripped.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Helvetica", style="B", size=13)
            pdf.set_text_color(91, 33, 182)
            pdf.multi_cell(0, 7, _to_latin1(stripped[3:].strip()))
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)
        elif stripped.startswith("### "):
            pdf.set_font("Helvetica", style="B", size=11)
            pdf.multi_cell(0, 6, _to_latin1(stripped[4:].strip()))
        elif _BULLET_RE.match(stripped) or _NUMBERED_RE.match(stripped):
            body = _BULLET_RE.sub("", stripped, count=1)
            body = _NUMBERED_RE.sub("", body, count=1)
            pdf.set_font("Helvetica", size=10)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(0, 5, _to_latin1(f"• {body}"))
        else:
            pdf.set_font("Helvetica", size=10)
            pdf.multi_cell(0, 5, _to_latin1(stripped))

    return bytes(pdf.output())
