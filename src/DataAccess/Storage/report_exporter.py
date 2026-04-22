from __future__ import annotations

import re

from fpdf import FPDF


def safe_filename(name: str) -> str:
    return re.sub(r"\W+", "_", name).strip("_") or "app"


def build_pdf_bytes(app_name: str, ai_analysis: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=16)
    pdf.cell(0, 10, f"Analisis - {app_name}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    for line in re.sub(r"[#*`]", "", ai_analysis).split("\n"):
        if line.strip():
            pdf.multi_cell(
                0, 5,
                line.strip().encode("latin-1", errors="replace").decode("latin-1"),
            )
            pdf.ln(1)
    return bytes(pdf.output())
