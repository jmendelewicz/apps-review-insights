from __future__ import annotations

import base64
import html
import os
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from google import genai

sys.path.insert(0, os.path.dirname(__file__))

from constants import CONTENT_COL, DATE_COL, LIKES_COL, SCORE_COL, VERSION_COL
from DataAccess.Ingestion.google_play_wrapper import scrape_reviews
from DataAccess.Ingestion.play_store_search import extract_app_id, get_app_info, resolve_query
from DataAccess.Refinary.google_review_cleaner import clean_dataframe
from DataAccess.Storage.report_exporter import build_pdf_bytes, safe_filename
from Models.gemini_analyzer import extract_topics_gemini, generate_deep_analysis
from Models.sentiment_client import build_client as build_sentiment_client
from Models.text_analysis import build_wordcloud_html, get_word_freq

# ---------- Constants ----------

COLORS = {"positive": "#22c55e", "neutral": "#f59e0b", "negative": "#ef4444"}
VIOLET = "#7c3aed"
VIOLET_DARK = "#5b21b6"


def esc(value) -> str:
    """Escape user/external strings before interpolating into HTML markdown."""
    return html.escape(str(value), quote=True) if value is not None else ""


@st.cache_data(show_spinner=False)
def _logo_data_url() -> str:
    """Load the brand logo from docs/ and return it as a base64 data URL.

    Kept as a data URL (not an st.image) so the navbar can lay out the logo
    and the "Play Insights" wordmark on a single row with consistent sizing.
    Returns "" if no logo file is found — caller falls back to a styled chip.
    """
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "docs" / "Logo 2.png",
        root / "docs" / "Logo2.png",
        root / "docs" / "Logo 2.jpeg",
        root / "docs" / "Logo.jpeg",
        root / "docs" / "Logo.png",
        root / "docs" / "logo.png",
        root / "docs" / "image.png",
    ]
    for p in candidates:
        if p.exists():
            mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
            try:
                data = base64.b64encode(p.read_bytes()).decode("ascii")
                return f"data:{mime};base64,{data}"
            except Exception:
                continue
    return ""


def _unique_countries(*groups) -> tuple[str, ...]:
    merged: list[str] = []
    for g in groups:
        merged.extend(g)
    return tuple(dict.fromkeys(merged))

COUNTRIES = {
    "Argentina": "ar", "México": "mx", "España": "es", "Colombia": "co",
    "Chile": "cl", "Perú": "pe", "Uruguay": "uy", "Paraguay": "py",
    "Estados Unidos": "us", "Brasil": "br",
}

# Regions group countries for a simpler UX. Each region maps to the display
# names used as keys in COUNTRIES. Brasil queda en "Global" (no en Sudamérica)
# porque el idioma por defecto es español; si se agrega PT en el futuro,
# puede moverse. Al analizar, la cantidad total se reparte entre los países
# de la(s) región(es) seleccionada(s).
REGIONS: dict[str, list[str]] = {
    "Sudamérica": ["Argentina", "Chile", "Colombia", "Perú", "Uruguay", "Paraguay"],
    "Norteamérica": ["México", "Estados Unidos"],
    "Europa": ["España"],
    "Global": ["Argentina", "México", "España", "Estados Unidos", "Brasil"],
}

# Idioma: por ahora solo español. Dejamos el mapa por si se agregan otros.
LANGUAGES = {"Español": "es"}

# Cuando el usuario pide "Más relevantes", Google Play devuelve un set curado
# de pocas reseñas (típicamente <200). Capear evita prometer volumen que la
# API no entrega.
MAX_COUNT_MOST_RELEVANT = 200


def regions_to_countries(sel_regions: list[str]) -> list[str]:
    """Flatten selected regions into a unique list of country display names."""
    seen: list[str] = []
    for r in sel_regions:
        for c in REGIONS.get(r, []):
            if c not in seen and c in COUNTRIES:
                seen.append(c)
    return seen

BADGE_MAP = {"positive": "badge-positive", "negative": "badge-negative", "neutral": "badge-neutral"}
LABEL_MAP = {"positive": "Positivo", "negative": "Negativo", "neutral": "Neutro"}
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#475569", family="Inter"), margin=dict(t=10, b=20, l=40, r=20),
)


# ---------- Page config ----------

st.set_page_config(
    page_title="PlayInsights", page_icon="📊", layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------- Session state ----------

def _init_state():
    defaults = {"page": "home", "history": [], "search_results": None, "rev_page": 1}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()


# ---------- Styling ----------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Sora:wght@500;600;700;800&display=swap');
    :root {
        --violet: #7c3aed;
        --violet-dark: #5b21b6;
        --violet-light: #ede9fe;
        --violet-soft: #f5f3ff;
    }
    html, body, .stApp, .block-container { font-family: 'Inter', sans-serif; }
    h1,h2,h3,h4 { font-family: 'Sora', 'Inter', sans-serif !important; font-weight: 700 !important; color: var(--violet-dark) !important; }

    /* Preserve Material Symbols ligatures used by Streamlit ≥1.40 for expander
       caret, button chevrons, etc. Without this, the global Inter override
       causes the literal ligature text (e.g. "arrow_drop_down") to render on
       top of user content in Streamlit Cloud. */
    [data-testid="stIconMaterial"],
    span[class*="icon"].material-symbols-outlined,
    span.material-symbols-outlined,
    span.material-symbols-rounded,
    span.material-icons,
    span.material-icons-outlined,
    .stApp [class*="MaterialIcon"] {
        font-family: 'Material Symbols Outlined', 'Material Symbols Rounded',
                     'Material Icons', 'Material Icons Outlined' !important;
        font-feature-settings: 'liga' !important;
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24 !important;
        letter-spacing: normal !important;
    }
    .block-container { padding-top: 3rem; max-width: 1180px; margin: 0 auto; }
    header[data-testid="stHeader"] { background: white; border-bottom: 1px solid #e2e8f0; }
    div[data-testid="stSidebar"] { display: none; }

    button[kind="primary"], .stButton button[kind="primary"] {
        background: var(--violet) !important; border-color: var(--violet) !important;
    }
    button[kind="primary"]:hover { background: var(--violet-dark) !important; }

    .stat-card {
        background: linear-gradient(135deg, var(--violet) 0%, var(--violet-dark) 100%);
        border: none; border-radius: 16px; padding: 1.2rem 1.3rem 1.3rem;
        box-shadow: 0 4px 14px rgba(124,58,237,0.18);
        height: 138px; display: flex; flex-direction: column; justify-content: flex-start;
        color: white;
    }
    .stat-card .label {
        font-size: 0.68rem; color: rgba(255,255,255,0.85); text-transform: uppercase;
        letter-spacing: 1.5px; font-weight: 700; margin-bottom: 0.55rem;
        min-height: 0.9rem;
    }
    .stat-card .value {
        font-size: 1.85rem; font-weight: 800; color: white;
        font-family: 'Sora', 'Inter', sans-serif; line-height: 1.1;
    }
    .stat-card .sub {
        font-size: 0.78rem; margin-top: auto; font-weight: 700;
        letter-spacing: 0.2px;
    }
    .stat-card .sub-placeholder { margin-top: auto; min-height: 0.9rem; }

    .chart-card { background: white; border: 1px solid #e2e8f0; border-radius: 14px; padding: 1.2rem 1.3rem 1.4rem; margin-bottom: 0.8rem; box-shadow: 0 1px 2px rgba(15,23,42,0.03); }

    /* Chart titles: matched to the loading page section-title style —
       gradient text, centered, with a small accent bar underneath. */
    .chart-card-title {
        font-family: 'Sora', 'Inter', sans-serif;
        font-size: 1.2rem; font-weight: 800;
        background: linear-gradient(135deg, var(--violet-dark) 0%, var(--violet) 60%, #a855f7 100%);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center; margin: 0.3rem 0 0;
        letter-spacing: -0.01em; line-height: 1.2;
    }
    .chart-card-title::after {
        content: ""; display: block; width: 38px; height: 3px;
        margin: 0.55rem auto 0;
        background: linear-gradient(90deg, var(--violet), #a855f7);
        border-radius: 2px;
    }
    .chart-card-kicker {
        display: block; text-align: center;
        font-size: 0.66rem; color: #94a3b8; text-transform: uppercase;
        letter-spacing: 1.8px; font-weight: 600;
        margin: 0.55rem 0 1rem;
    }

    .app-card { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 1rem; transition: all 0.15s; }
    .app-card:hover { border-color: var(--violet); box-shadow: 0 2px 8px rgba(124,58,237,0.1); }

    .review-card { background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.4rem; }

    .badge-positive { background: #dcfce7; color: #166534; padding: 1px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 500; border: 1px solid #bbf7d0; }
    .badge-negative { background: #fee2e2; color: #991b1b; padding: 1px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 500; border: 1px solid #fecaca; }
    .badge-neutral  { background: #fef3c7; color: #92400e; padding: 1px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 500; border: 1px solid #fde68a; }

    .html-wordcloud { display: flex; flex-wrap: wrap; gap: 6px 12px; justify-content: center; align-items: center; min-height: 180px; padding: 1rem; }
    .html-wordcloud span { color: var(--violet-dark); line-height: 1.2; cursor: default; transition: opacity 0.15s; }
    .html-wordcloud span:hover { opacity: 1 !important; }

    .hero-label {
        font-size: 0.72rem; color: var(--violet); text-transform: uppercase;
        letter-spacing: 3px; font-weight: 700; text-align: center;
    }
    .hero-title {
        font-family: 'Sora', 'Inter', sans-serif;
        font-size: 3rem; font-weight: 800;
        background: linear-gradient(135deg, var(--violet-dark) 0%, var(--violet) 60%, #a855f7 100%);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.05; letter-spacing: -0.02em; margin: 0.4rem 0 0.6rem;
        text-align: center;
    }
    .hero-sub { font-size: 0.95rem; color: #64748b; text-align: center; max-width: 620px; margin: 0 auto; }

    .section-title {
        font-family: 'Sora', 'Inter', sans-serif;
        font-size: 1.55rem; font-weight: 800;
        background: linear-gradient(135deg, var(--violet-dark) 0%, var(--violet) 100%);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center; margin: 1.4rem 0 0.3rem; letter-spacing: -0.01em;
    }
    .section-title::after {
        content: ""; display: block; width: 42px; height: 3px; margin: 0.6rem auto 0;
        background: linear-gradient(90deg, var(--violet), #a855f7);
        border-radius: 2px;
    }
    .section-sub { text-align: center; color: #64748b; font-size: 0.88rem; margin: 0.6rem 0 1.3rem; }

    .ai-hero {
        background: linear-gradient(135deg, var(--violet) 0%, var(--violet-dark) 100%);
        color: white; border-radius: 16px; padding: 1.5rem 1.8rem;
        box-shadow: 0 6px 20px rgba(124,58,237,0.25); margin-bottom: 1.2rem;
    }
    .ai-hero-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 2px; font-weight: 600; opacity: 0.85; }
    .ai-hero-title { font-family: 'Sora', 'Inter', sans-serif; font-size: 1.6rem; font-weight: 700; margin-top: 0.3rem; }

    /* Expanders — card-like with violet accent */
    div[data-testid="stExpander"] {
        background: white; border: 1px solid var(--violet-light);
        border-left: 4px solid var(--violet); border-radius: 12px !important;
        margin-bottom: 0.6rem; box-shadow: 0 1px 3px rgba(15,23,42,0.04);
        transition: box-shadow 0.2s, transform 0.15s; overflow: hidden;
    }
    div[data-testid="stExpander"]:hover {
        box-shadow: 0 4px 14px rgba(124,58,237,0.15); transform: translateY(-1px);
    }
    div[data-testid="stExpander"] summary {
        font-weight: 700 !important; color: var(--violet-dark) !important;
        font-size: 1rem !important; padding: 0.85rem 1.1rem !important;
    }
    div[data-testid="stExpander"] summary p {
        font-family: 'Sora', 'Inter', sans-serif !important;
        font-weight: 700 !important; color: var(--violet-dark) !important;
    }

    /* Pill-style tabs: selected one paints violet. */
    div[data-baseweb="tab-list"] {
        gap: 0.5rem !important;
        border-bottom: none !important;
        background: var(--violet-soft);
        padding: 0.35rem !important;
        border-radius: 12px !important;
        width: fit-content;
    }
    button[data-baseweb="tab"] {
        font-family: 'Sora', 'Inter', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.1rem !important;
        background: transparent !important;
        color: #64748b !important;
        transition: all 0.18s ease;
        border: 1px solid transparent !important;
    }
    button[data-baseweb="tab"]:hover { color: var(--violet-dark) !important; background: rgba(124,58,237,0.05) !important; }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: var(--violet) !important;
        color: white !important;
        border-color: var(--violet) !important;
        box-shadow: 0 2px 8px rgba(124,58,237,0.28);
    }
    /* Hide the default underline highlight — pill replaces it. */
    div[data-baseweb="tab-highlight"] { display: none !important; }
    div[data-baseweb="tab-border"] { display: none !important; }

    /* AI Key Insight callout */
    .ai-key-insight {
        background: linear-gradient(135deg, #faf5ff 0%, #f5f3ff 100%);
        border: 1px solid var(--violet-light);
        border-left: 4px solid var(--violet);
        border-radius: 12px; padding: 1rem 1.2rem;
        margin: 0.6rem 0 1rem;
    }
    .ai-key-insight .kicker {
        font-size: 0.68rem; color: var(--violet); text-transform: uppercase;
        letter-spacing: 2px; font-weight: 700; margin-bottom: 0.35rem;
    }
    .ai-key-insight .body {
        font-family: 'Sora', 'Inter', sans-serif;
        color: var(--violet-dark); font-size: 1rem; font-weight: 600;
        line-height: 1.45;
    }

    /* AI top-level title (when the LLM adds its own report title) */
    .ai-report-title {
        font-family: 'Sora', 'Inter', sans-serif;
        font-size: 1.25rem; font-weight: 800; color: var(--violet-dark);
        margin: 0.4rem 0 0.2rem; letter-spacing: -0.01em;
    }

    /* Evidence quotes inside AI sections */
    .ai-evidence {
        margin-top: 0.9rem; padding-top: 0.7rem;
        border-top: 1px dashed var(--violet-light);
    }
    .ai-evidence-label {
        font-size: 0.68rem; color: var(--violet); text-transform: uppercase;
        letter-spacing: 1.5px; font-weight: 700; margin-bottom: 0.4rem;
    }
    .ai-quote {
        background: #faf5ff; border-left: 3px solid var(--violet);
        padding: 0.55rem 0.85rem; margin-bottom: 0.4rem; border-radius: 6px;
        font-size: 0.83rem; color: #475569; line-height: 1.5;
    }
    .ai-quote-meta {
        display: block; font-size: 0.7rem; color: #94a3b8; margin-top: 0.25rem;
    }

    /* History: small app icon sits next to the native Streamlit expander.
       Padding matches the expander summary's vertical padding so the icon
       appears centered on the header row even when the expander is closed. */
    .history-icon-wrap {
        padding: 0.55rem 0 0;
        display: flex; justify-content: center;
    }
    .history-icon {
        width: 36px; height: 36px; border-radius: 8px;
        object-fit: cover; display: block;
        border: 1px solid var(--violet-light);
        box-shadow: 0 1px 3px rgba(15,23,42,0.06);
    }
    .history-icon-placeholder {
        width: 36px; height: 36px; background: var(--violet-light);
        border-radius: 8px; display: flex; align-items: center; justify-content: center;
        font-size: 1.15rem;
    }
    .history-header-main { padding: 0.2rem 0; }
    .history-header-title {
        font-family: 'Sora', 'Inter', sans-serif;
        font-weight: 700; color: var(--violet-dark); font-size: 1.02rem;
        line-height: 1.2;
    }
    .history-header-sub {
        color: #64748b; font-size: 0.78rem; margin-top: 0.2rem;
    }
    .history-wrap {
        background: white; border: 1px solid var(--violet-light);
        border-left: 4px solid var(--violet); border-radius: 12px;
        padding: 0.6rem 0.9rem; margin-bottom: 0.6rem;
        box-shadow: 0 1px 3px rgba(15,23,42,0.04);
    }
</style>
""", unsafe_allow_html=True)


# ---------- Secrets / clients ----------

def _secret(key: str, default: str = "") -> str:
    try:
        val = st.secrets["general"][key]
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key, default)


def get_api_key() -> str:
    # Accept either name — some deployments still use GOOGLE_API_KEY.
    return _secret("GEMINI_API_KEY") or _secret("GOOGLE_API_KEY")


@st.cache_resource(show_spinner=False)
def get_sentiment_client():
    hf_url = _secret("HF_SENTIMENT_URL")
    hf_token = _secret("HF_SENTIMENT_TOKEN")
    if hf_url:
        os.environ["HF_SENTIMENT_URL"] = hf_url
    if hf_token:
        os.environ["HF_SENTIMENT_TOKEN"] = hf_token
    return build_sentiment_client()


@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)


# ---------- Cached data fetchers ----------

@st.cache_data(show_spinner=False, ttl=300)
def cached_resolve_query(query: str, lang: str, countries: tuple[str, ...]) -> list[dict]:
    return resolve_query(query, lang=lang, countries=list(countries))


@st.cache_data(show_spinner=False, ttl=600)
def cached_get_app_info(app_id: str, lang: str, countries: tuple[str, ...]) -> Optional[dict]:
    return get_app_info(app_id, lang=lang, countries=list(countries))


@st.cache_data(show_spinner=False, ttl=600)
def cached_scrape_reviews(
    app_id: str, lang: str, country: str, country_name: str,
    count: int, sort_label: str,
) -> pd.DataFrame:
    return scrape_reviews(app_id, lang, country, count, sort_label, country_name)


# ---------- UI helpers ----------

_STAT_TONES = {
    # Bright, high-contrast colors over the violet gradient background.
    "positive": "#bbf7d0",   # mint green — legible sobre violeta
    "negative": "#fecaca",   # rojo suave — legible sobre violeta
    "neutral":  "#fde68a",
    "default":  "rgba(255,255,255,0.92)",
}


def render_stat(label: str, value: str, sub: str = "", tone: str = "default"):
    color = _STAT_TONES.get(tone, _STAT_TONES["default"])
    if sub:
        sub_html = f'<div class="sub" style="color:{color}">{sub}</div>'
    else:
        # Keep the value block at the same vertical position across all cards.
        sub_html = '<div class="sub-placeholder">&nbsp;</div>'
    st.markdown(
        f'<div class="stat-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def render_html_wordcloud(word_freq: list[tuple[str, int]]):
    html = build_wordcloud_html(word_freq)
    if html:
        st.markdown(html, unsafe_allow_html=True)


# ---------- Navigation ----------

def go_to(page: str, **extra):
    st.session_state["page"] = page
    for k, v in extra.items():
        st.session_state[k] = v
    st.rerun()


def load_history_entry(idx: int):
    entry = st.session_state["history"][idx]
    st.session_state.update({
        "sel_app_id": entry["app_id"],
        "sel_app_name": entry["name"],
        "sel_app_icon": entry.get("icon", ""),
        "sel_app_dev": entry.get("dev", ""),
        "sel_app_genre": entry.get("genre", ""),
        "sel_app_score": entry.get("score", 0),
        "sel_app_installs": entry.get("installs", ""),
        "sel_lang": entry.get("lang", "es"),
        "sel_countries": entry.get("countries", ["Argentina"]),
        "sel_regions": entry.get("regions", ["Sudamérica"]),
        "sel_sort": entry.get("sort", "Más recientes"),
        "sel_count": entry.get("count", 100),
        "df_analyzed": entry["df"],
        "topics": entry.get("topics", []),
        "ai_analysis": entry.get("ai_analysis", ""),
        "rev_page": 1,
    })
    go_to("dashboard")


def push_history(app_name: str, app_id: str, df: pd.DataFrame, topics: list, ai_analysis: str):
    hist = st.session_state["history"]
    hist[:] = [h for h in hist if h["app_id"] != app_id]
    hist.append({
        "app_id": app_id,
        "name": app_name,
        "icon": st.session_state.get("sel_app_icon", ""),
        "dev": st.session_state.get("sel_app_dev", ""),
        "genre": st.session_state.get("sel_app_genre", ""),
        "score": st.session_state.get("sel_app_score", 0),
        "installs": st.session_state.get("sel_app_installs", ""),
        "count": len(df),
        "df": df,
        "topics": topics,
        "ai_analysis": ai_analysis,
        # Preserve the configuration so the user can re-run with the same params.
        "lang": st.session_state.get("sel_lang", "es"),
        "countries": list(st.session_state.get("sel_countries", [])),
        "regions": list(st.session_state.get("sel_regions", [])),
        "sort": st.session_state.get("sel_sort", "Más recientes"),
    })


# ---------- Navbar ----------

current_page = st.session_state["page"]
col_logo, _, col_nav = st.columns([2, 4, 2])
with col_logo:
    logo_url = _logo_data_url()
    if logo_url:
        logo_html = (
            f'<img src="{logo_url}" alt="Play Insights" '
            f'style="width:44px;height:44px;border-radius:14px;object-fit:cover;'
            f'box-shadow:0 3px 10px rgba(124,58,237,0.22);'
            f'border:1px solid var(--violet-light);background:white">'
        )
    else:
        logo_html = (
            '<div style="width:44px;height:44px;'
            'background:linear-gradient(135deg,var(--violet),var(--violet-dark));'
            'border-radius:14px;display:flex;align-items:center;justify-content:center;'
            'box-shadow:0 3px 10px rgba(124,58,237,0.22)">'
            '<span style="color:white;font-size:1rem;font-weight:800">PI</span></div>'
        )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.75rem;padding-top:0">'
        f'{logo_html}'
        f'<span style="font-family:\'Sora\',\'Inter\',sans-serif;font-weight:800;'
        f'font-size:1.18rem;color:var(--violet-dark);letter-spacing:-0.01em">'
        f'Play Insights</span></div>',
        unsafe_allow_html=True,
    )
with col_nav:
    nc1, nc2 = st.columns(2)
    with nc1:
        if st.button("🔍 Buscar", key="nav_search", use_container_width=True,
                     type="primary" if current_page == "home" else "secondary"):
            st.session_state["search_results"] = None
            go_to("home")
    with nc2:
        if st.button("📋 Historial", key="nav_hist", use_container_width=True,
                     type="primary" if current_page == "history" else "secondary"):
            go_to("history")

st.markdown("<hr style='margin:0 0 1rem;border-color:#e2e8f0'>", unsafe_allow_html=True)


# ---------- Page: Home ----------

def render_home():
    st.markdown('<div class="hero-label">Google Play Store</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Análisis de Reseñas<br>con IA</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Busca cualquier app por nombre, URL o package ID '
        '(ej. <code>com.mercadolibre</code>) y obtené un análisis profundo con Gemini AI.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form(key="search_form", clear_on_submit=False):
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            query = st.text_input(
                "Buscar",
                placeholder="Nombre, package ID o URL de Google Play...",
                label_visibility="collapsed",
                key="home_query",
            )
        with col_btn:
            search_clicked = st.form_submit_button(
                "Buscar", type="primary", use_container_width=True,
            )

        col_lang, col_regions, col_sort, col_count = st.columns([1, 2, 1.2, 1.2])
        with col_lang:
            # Idioma visible pero con una sola opción por ahora. El mapa
            # LANGUAGES está listo para sumar más cuando se incorporen.
            st.markdown('<span style="font-size:0.72rem;color:#64748b;font-weight:500">Idioma:</span>',
                        unsafe_allow_html=True)
            lang_name = st.selectbox(
                "Idioma", list(LANGUAGES.keys()), index=0,
                label_visibility="collapsed", key="home_lang",
            )
        with col_regions:
            st.markdown('<span style="font-size:0.72rem;color:#64748b;font-weight:500">Regiones:</span>',
                        unsafe_allow_html=True)
            sel_regions = st.multiselect(
                "Regiones", list(REGIONS.keys()), default=["Sudamérica"],
                label_visibility="collapsed", key="home_regions",
            )
        with col_sort:
            st.markdown('<span style="font-size:0.72rem;color:#64748b;font-weight:500">Ordenar por:</span>',
                        unsafe_allow_html=True)
            sort_order = st.selectbox(
                "Orden", ["Más recientes", "Más relevantes"],
                index=0, label_visibility="collapsed", key="home_sort",
            )
        with col_count:
            # Opciones dependen del sort: "Más relevantes" viene curado por
            # Google Play (<200 reseñas), pedir más no agrega información.
            is_relevant = sort_order == "Más relevantes"
            count_options = [50, 100, MAX_COUNT_MOST_RELEVANT] if is_relevant else [100, 200, 500, 1000]
            count_label = "Reseñas (máx. ~200 en “Más relevantes”):" if is_relevant else "Reseñas:"
            st.markdown(
                f'<span style="font-size:0.72rem;color:#64748b;font-weight:500">{count_label}</span>',
                unsafe_allow_html=True,
            )
            review_count = st.selectbox(
                "Reseñas", count_options, index=0,
                label_visibility="collapsed", format_func=lambda x: f"{x:,} reseñas",
                key=f"home_count_{'rel' if is_relevant else 'rec'}",
            )

    lang_code = LANGUAGES[lang_name]

    # Lista plana de países (display names) derivada de las regiones elegidas.
    sel_countries = regions_to_countries(sel_regions) or ["Argentina"]

    if not query.strip():
        st.session_state["search_results"] = None

    if search_clicked and query.strip():
        country_codes = tuple(COUNTRIES[c] for c in sel_countries)
        fallback_countries = _unique_countries(country_codes, ("us", "ar"))
        results = cached_resolve_query(query.strip(), lang_code, fallback_countries)
        st.session_state["search_results"] = results
        if not results:
            st.markdown(
                '<div style="display:flex;align-items:center;gap:0.5rem;color:#dc2626;'
                'background:#fef2f2;border:1px solid #fecaca;border-radius:8px;'
                'padding:0.8rem 1rem;margin:1rem 0;font-size:0.85rem">'
                '<span>⚠️</span><span>No se encontraron apps. Probá con otro término, '
                'un package ID o la URL de Google Play.</span></div>',
                unsafe_allow_html=True,
            )

    results = st.session_state.get("search_results") or []
    if results:
        st.markdown(
            f'<p style="font-size:0.68rem;color:var(--violet);text-transform:uppercase;'
            f'letter-spacing:1.5px;font-weight:700;margin:1rem 0 0.8rem;text-align:center">'
            f'{len(results)} resultado{"s" if len(results) != 1 else ""} '
            f'encontrado{"s" if len(results) != 1 else ""}</p>',
            unsafe_allow_html=True,
        )
        for idx, app_data in enumerate(results):
            app_id = app_data.get("appId", "")
            if not app_id:
                continue
            title = app_data.get("title") or app_id
            icon = app_data.get("icon", "")
            dev = app_data.get("developer", "")
            score = app_data.get("score", 0) or 0
            installs = app_data.get("installs", "")
            genre = app_data.get("genre", "")

            col_card, col_action = st.columns([5, 1])
            with col_card:
                badge = (
                    f'<span style="font-size:0.65rem;color:#64748b;border:1px solid #e2e8f0;'
                    f'padding:1px 6px;border-radius:3px;margin-left:0.5rem">{esc(genre)}</span>'
                    if genre else ""
                )
                score_html = (
                    f'<span style="font-size:0.7rem;color:#64748b">⭐ {score:.1f}</span>'
                    if score else ""
                )
                inst_html = (
                    f'<span style="font-size:0.7rem;color:#94a3b8">⬇ {esc(installs)}</span>'
                    if installs and installs != "N/A" else ""
                )
                st.markdown(
                    f'<div class="app-card">'
                    f'<img src="{esc(icon)}" style="width:48px;height:48px;border-radius:8px" '
                    f'onerror="this.style.display=\'none\'">'
                    f'<div style="flex:1;min-width:0">'
                    f'<div style="display:flex;align-items:center;gap:0.3rem">'
                    f'<span style="font-weight:600;font-size:0.9rem;color:#0f172a">{esc(title)}</span>{badge}</div>'
                    f'<div style="font-size:0.75rem;color:#64748b">{esc(dev)}</div>'
                    f'<div style="display:flex;gap:0.8rem;margin-top:0.2rem">{score_html}{inst_html}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            with col_action:
                st.markdown("<div style='padding-top:1rem'></div>", unsafe_allow_html=True)
                if st.button("Analizar ›", key=f"analyze_{app_id}_{idx}",
                             use_container_width=True, type="primary"):
                    go_to(
                        "loading",
                        sel_app_id=app_id,
                        sel_app_name=title,
                        sel_app_icon=icon,
                        sel_app_dev=dev,
                        sel_app_genre=genre,
                        sel_app_score=score,
                        sel_app_installs=installs,
                        sel_lang=lang_code,
                        sel_count=review_count,
                        sel_countries=sel_countries or ["Argentina"],
                        sel_regions=sel_regions or ["Sudamérica"],
                        sel_sort=sort_order,
                    )
    else:
        st.markdown(
            '<div style="text-align:center;padding:3rem 0">'
            '<div style="width:48px;height:48px;background:var(--violet-soft);border-radius:12px;'
            'display:flex;align-items:center;justify-content:center;margin:0 auto 1rem">'
            '<span style="font-size:1.2rem;color:var(--violet)">🔍</span></div>'
            '<p style="color:#64748b;font-size:0.9rem">Busca una app para comenzar el análisis</p>'
            '<p style="color:#94a3b8;font-size:0.78rem;margin-top:0.3rem">'
            'Ej: "Spotify", <code>com.mercadolibre</code> o una URL de Google Play</p></div>',
            unsafe_allow_html=True,
        )


# ---------- Page: Loading ----------

def render_loading():
    app_id = st.session_state.get("sel_app_id", "")
    app_name = st.session_state.get("sel_app_name", app_id)
    lang = st.session_state.get("sel_lang", "es")
    count = st.session_state.get("sel_count", 100)
    sel_countries = st.session_state.get("sel_countries", ["Argentina"])
    sort_label = st.session_state.get("sel_sort", "Más recientes")
    api_key = get_api_key()

    if not app_id or not str(app_id).strip():
        st.error("No se seleccionó ninguna app. Volvé al inicio y elegí una.")
        if st.button("← Volver", key="loading_back_empty"):
            go_to("home")
        st.stop()

    clean_id = extract_app_id(app_id) or str(app_id).strip()
    country_codes = tuple(COUNTRIES[c] for c in sel_countries) or ("ar",)
    lookup_countries = _unique_countries(country_codes, ("us", "ar"))

    if not st.session_state.get("sel_app_icon"):
        info = cached_get_app_info(clean_id, lang, lookup_countries)
        if info:
            st.session_state.update({
                "sel_app_name": info.get("title", clean_id),
                "sel_app_icon": info.get("icon", ""),
                "sel_app_dev": info.get("developer", ""),
                "sel_app_genre": info.get("genre", ""),
                "sel_app_score": info.get("score", 0),
                "sel_app_installs": info.get("installs", ""),
            })
            app_name = st.session_state["sel_app_name"]

    st.markdown(
        f'<div style="text-align:center;padding:3rem 0">'
        f'<div style="width:52px;height:52px;border:3px solid {VIOLET};border-top-color:transparent;'
        f'border-radius:50%;margin:0 auto 1.2rem;animation:spin 1s linear infinite"></div>'
        f'<h2 class="section-title" style="margin-top:0">Analizando {esc(app_name)}</h2>'
        f'<p style="color:#64748b;font-size:0.9rem">Esto puede tardar unos minutos...</p>'
        f'</div>'
        f'<style>@keyframes spin {{ 0%{{transform:rotate(0deg)}} 100%{{transform:rotate(360deg)}} }}</style>',
        unsafe_allow_html=True,
    )

    progress = st.progress(0, text="Iniciando...")
    n_countries = max(1, len(sel_countries))
    # Exact distribution: avoid inflating the total when per-country division rounds down.
    base, extra = divmod(count, n_countries)
    per_country_counts = [base + (1 if i < extra else 0) for i in range(n_countries)]

    dfs = []
    for i, country_name in enumerate(sel_countries):
        target = per_country_counts[i]
        if target <= 0:
            continue

        # Inter-country rate limiting: Add delay between countries to avoid
        # hammering Google Play API. Reduces 429 rate limit errors.
        if i > 0:
            import time
            time.sleep(2)

        pct = int(10 + (i / n_countries) * 40)
        progress.progress(pct, text=f"📱 Descargando reseñas de {country_name} ({target})...")
        try:
            df_part = cached_scrape_reviews(
                clean_id, lang, COUNTRIES[country_name], country_name, target, sort_label,
            )
            if not df_part.empty:
                dfs.append(df_part)
        except Exception as e:
            st.warning(f"Error en {country_name}: {e}")

    if not dfs:
        st.error("No se pudieron obtener reseñas. Verificá el package ID o probá otro país.")
        if st.button("← Volver", key="loading_back_empty_results"):
            go_to("home")
        st.stop()

    df = pd.concat(dfs, ignore_index=True)
    df = clean_dataframe(df)

    progress.progress(60, text="🧠 Clasificando sentimientos...")
    sentiment_client = get_sentiment_client()
    df["sentiment"] = sentiment_client.classify(df[CONTENT_COL].tolist())

    topics: list[dict] = []
    ai_analysis = ""
    if api_key:
        progress.progress(75, text="🤖 Extrayendo temas con IA (Gemini)...")
        try:
            gemini = get_gemini_client(api_key)
            topics = extract_topics_gemini(df, gemini, app_name)
        except Exception:
            topics = []

        progress.progress(85, text="📝 Generando análisis profundo...")
        try:
            gemini = get_gemini_client(api_key)
            ai_analysis = generate_deep_analysis(df, gemini, app_name, topics)
        except Exception as e:
            ai_analysis = f"Error al generar análisis: {str(e)[:200]}"

    progress.progress(95, text="💾 Guardando resultados...")
    st.session_state["df_analyzed"] = df
    st.session_state["topics"] = topics
    st.session_state["ai_analysis"] = ai_analysis
    st.session_state["rev_page"] = 1
    push_history(app_name, clean_id, df, topics, ai_analysis)
    progress.progress(100, text="✅ ¡Análisis completado!")
    go_to("dashboard")


# ---------- Page: History ----------

def render_history():
    st.markdown('<div class="hero-label">Análisis guardados</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="section-title" style="margin:0.3rem 0 0.3rem">Historial</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Explorá, descargá y gestioná los análisis previos.</p>',
                unsafe_allow_html=True)

    hist = st.session_state.get("history", [])
    if not hist:
        st.markdown(
            '<div style="text-align:center;padding:4rem 0">'
            '<div style="font-size:2rem;margin-bottom:0.5rem">📋</div>'
            '<p style="color:#94a3b8;font-size:0.9rem">No hay análisis guardados</p></div>',
            unsafe_allow_html=True,
        )
        if st.button("🔍 Buscar primera app", use_container_width=True):
            go_to("home")
        return

    for idx, entry in reversed(list(enumerate(hist))):
        score_txt = f" · ⭐ {entry['score']:.1f}" if entry.get("score") else ""
        has_ai = bool(entry.get("ai_analysis")) and not entry["ai_analysis"].startswith("Error")

        # Icono de la app a la izquierda + expander nativo a la derecha.
        # El expander conserva su caret/colapso estándar; el icono se
        # muestra siempre (colapsado o abierto) porque vive fuera del header.
        hc_icon, hc_exp = st.columns([0.5, 11])
        with hc_icon:
            if entry.get("icon"):
                st.markdown(
                    f'<div class="history-icon-wrap">'
                    f'<img src="{esc(entry["icon"])}" class="history-icon" alt=""></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="history-icon-wrap">'
                    '<div class="history-icon-placeholder">📊</div></div>',
                    unsafe_allow_html=True,
                )
        with hc_exp:
            ai_badge = "  ·  🧠 IA" if has_ai else ""
            header = f"{entry['name']}  ·  {entry['count']:,} reseñas{score_txt}{ai_badge}"
            with st.expander(header, expanded=False):
                dev_genre = " · ".join(
                    p for p in (esc(entry.get("dev", "")), esc(entry.get("genre", ""))) if p
                )
                n_topics = len(entry.get("topics", []))
                topics_html = (
                    f' · <span style="color:#64748b;font-size:0.72rem">'
                    f'{n_topics} temas detectados</span>' if n_topics else ""
                )
                if dev_genre or topics_html:
                    st.markdown(
                        f'<div style="font-size:0.78rem;color:#64748b;margin:0.1rem 0 0.7rem">'
                        f'{dev_genre}{topics_html}</div>',
                        unsafe_allow_html=True,
                    )

                safe = safe_filename(entry["name"])
                ai_text = entry.get("ai_analysis", "")
                df_entry = entry.get("df")

                # Todos los botones en una sola fila, labels cortos para que
                # quepan cómodamente aunque el expander esté en pantalla chica.
                b1, b2, b3, b4, b5 = st.columns([1.2, 1, 1, 1, 1])
                with b1:
                    if st.button("📂 Abrir análisis", key=f"hist_open_{idx}",
                                 use_container_width=True, type="primary"):
                        load_history_entry(idx)
                with b2:
                    if has_ai:
                        st.download_button(
                            "📝 MD",
                            data=ai_text.encode("utf-8"),
                            file_name=f"analisis_{safe}.md",
                            mime="text/markdown",
                            use_container_width=True,
                            key=f"hist_md_{idx}",
                        )
                    else:
                        st.button("📝 MD", key=f"hist_md_dis_{idx}",
                                  use_container_width=True, disabled=True)
                with b3:
                    if has_ai:
                        try:
                            pdf_bytes = build_pdf_bytes(entry["name"], ai_text)
                            st.download_button(
                                "📄 PDF",
                                data=pdf_bytes,
                                file_name=f"analisis_{safe}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                key=f"hist_pdf_{idx}",
                            )
                        except Exception:
                            st.button("📄 PDF", key=f"hist_pdf_err_{idx}",
                                      use_container_width=True, disabled=True)
                    else:
                        st.button("📄 PDF", key=f"hist_pdf_dis_{idx}",
                                  use_container_width=True, disabled=True)
                with b4:
                    if df_entry is not None and not df_entry.empty:
                        export_cols = [c for c in [CONTENT_COL, "sentiment", SCORE_COL, LIKES_COL,
                                                   DATE_COL, VERSION_COL, "userName"]
                                       if c in df_entry.columns]
                        csv_bytes = df_entry[export_cols].to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ CSV",
                            data=csv_bytes,
                            file_name=f"reviews_{safe}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key=f"hist_csv_{idx}",
                        )
                    else:
                        st.button("⬇️ CSV", key=f"hist_csv_dis_{idx}",
                                  use_container_width=True, disabled=True)
                with b5:
                    if st.button("🗑 Eliminar", key=f"hist_del_{idx}", use_container_width=True):
                        st.session_state["history"].pop(idx)
                        st.rerun()


# ---------- Dashboard helpers ----------

def _render_dashboard_header(app_name: str, total: int):
    app_icon = st.session_state.get("sel_app_icon", "")
    app_dev = st.session_state.get("sel_app_dev", "")
    app_genre = st.session_state.get("sel_app_genre", "")
    app_play_score = st.session_state.get("sel_app_score", 0) or 0
    app_installs = st.session_state.get("sel_app_installs", "")

    col_back, col_header, col_rating = st.columns([0.5, 5, 2])
    with col_back:
        if st.button("←", key="dash_back"):
            for k in ["df_analyzed", "topics", "ai_analysis", "sel_app_id", "sel_app_name",
                      "sel_app_icon", "sel_app_dev", "sel_app_genre", "sel_app_score",
                      "sel_app_installs"]:
                st.session_state.pop(k, None)
            st.session_state["rev_page"] = 1
            go_to("home")
    with col_header:
        icon_html = f'<img src="{esc(app_icon)}" style="width:48px;height:48px;border-radius:4px">' if app_icon else ""
        genre_html = f' · {esc(app_genre)}' if app_genre else ""
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:1rem">{icon_html}'
            f'<div><div style="font-weight:700;font-size:1.2rem;color:#0f172a">{esc(app_name)}</div>'
            f'<div style="font-size:0.78rem;color:#64748b">{esc(app_dev)}{genre_html}</div></div></div>',
            unsafe_allow_html=True,
        )
    with col_rating:
        score_html = (
            f'<span style="background:#f8fafc;border:1px solid #e2e8f0;padding:3px 10px;'
            f'border-radius:4px;font-weight:600;font-size:0.85rem">⭐ {app_play_score:.1f}</span>'
            if app_play_score else ""
        )
        inst_html = (
            f'<span style="color:#94a3b8;font-size:0.78rem;margin-left:0.5rem">{esc(app_installs)}</span>'
            if app_installs else ""
        )
        st.markdown(
            f'<div style="text-align:right;padding-top:0.5rem">{score_html}{inst_html}</div>',
            unsafe_allow_html=True,
        )


# Mapeo keyword → sentimiento de las reseñas a mostrar como evidencia
# para cada sección del análisis IA. Los keywords se buscan en el título
# de la sección normalizado a minúscula.
_AI_SECTION_EVIDENCE = {
    "fortaleza": ("positive", "Reseñas que respaldan esta fortaleza"),
    "problema":  ("negative", "Reseñas que ilustran este problema"),
    "recomendac": ("negative", "Quejas frecuentes que motivan estas recomendaciones"),
    "perfil":     (None, "Voces representativas del tipo de usuario"),
    "resumen":    (None, None),  # sin evidencia: es la síntesis
}

_AI_CANONICAL_KEYWORDS = ("resumen", "fortaleza", "problema", "perfil", "recomendac")


def _extract_key_insight(ai_text: str) -> str:
    """Pull the first meaningful sentence of the Resumen Ejecutivo.

    Used as the TL;DR card at the top of the AI tab so the reader gets the
    "qué está pasando" before scrolling through the full breakdown.
    """
    if not ai_text:
        return ""
    # Prefer the Resumen Ejecutivo section if present.
    m = re.search(r"##\s*Resumen Ejecutivo\s*\n(.+?)(?=\n##\s|\Z)", ai_text,
                  flags=re.IGNORECASE | re.DOTALL)
    block = m.group(1).strip() if m else ai_text.strip()
    # Drop any leading bullet markers.
    block = re.sub(r"^[-*•\s]+", "", block)
    # Take the first 1-2 sentences (stop at . ? ! followed by space/newline).
    sentences = re.split(r"(?<=[.!?])\s+", block)
    out = " ".join(sentences[:2]).strip()
    # Trim to a reasonable length for the callout.
    if len(out) > 320:
        out = out[:317].rsplit(" ", 1)[0] + "…"
    return out


def _pick_evidence_reviews(
    df: pd.DataFrame, sentiment: Optional[str], n: int = 2,
) -> list[dict]:
    """Select real review quotes to anchor an AI section.

    Priority: highest-likes reviews (most socially validated) filtered by the
    requested sentiment. If no sentiment is given, just return the top-likes
    reviews overall (used for 'Perfil del usuario').
    """
    if df is None or df.empty or CONTENT_COL not in df.columns:
        return []
    subset = df
    if sentiment and "sentiment" in df.columns:
        subset = df[df["sentiment"] == sentiment]
    if subset.empty:
        return []
    sort_col = LIKES_COL if LIKES_COL in subset.columns else SCORE_COL
    if sort_col in subset.columns:
        subset = subset.sort_values(sort_col, ascending=False)
    picks: list[dict] = []
    for _, row in subset.head(n * 3).iterrows():
        text = str(row.get(CONTENT_COL, "")).strip()
        if len(text) < 20:
            continue
        picks.append({
            "text": text[:260] + ("…" if len(text) > 260 else ""),
            "likes": int(row.get(LIKES_COL, 0) or 0),
            "score": int(row.get(SCORE_COL, 0) or 0),
            "user": str(row.get("userName", "Anónimo")),
        })
        if len(picks) >= n:
            break
    return picks


def _render_ai_evidence(df: pd.DataFrame, section_title: str):
    """Render quote evidence under an AI section, if the section type warrants it."""
    key = next(
        (k for k in _AI_SECTION_EVIDENCE if k in section_title.lower()),
        None,
    )
    if not key:
        return
    sentiment, label = _AI_SECTION_EVIDENCE[key]
    if not label:
        return
    picks = _pick_evidence_reviews(df, sentiment, n=2)
    if not picks:
        return
    html_parts = [
        '<div class="ai-evidence">',
        f'<div class="ai-evidence-label">📎 {esc(label)}</div>',
    ]
    for p in picks:
        meta_bits = []
        if p["score"]:
            meta_bits.append(f'{"★" * p["score"]}')
        meta_bits.append(esc(p["user"]))
        if p["likes"]:
            meta_bits.append(f'👍 {p["likes"]}')
        meta = " · ".join(meta_bits)
        html_parts.append(
            f'<div class="ai-quote">"{esc(p["text"])}"'
            f'<span class="ai-quote-meta">{meta}</span></div>'
        )
    html_parts.append('</div>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def _render_ai_tab(ai_analysis: str, app_name: str, api_key: str,
                   df: Optional[pd.DataFrame] = None):
    app_icon = st.session_state.get("sel_app_icon", "")
    icon_html = (
        f'<img src="{esc(app_icon)}" style="width:54px;height:54px;border-radius:12px;'
        f'border:2px solid rgba(255,255,255,0.3);flex-shrink:0">' if app_icon else ""
    )
    st.markdown(
        f'<div class="ai-hero">'
        f'<div style="display:flex;align-items:center;gap:1rem">'
        f'{icon_html}'
        f'<div style="flex:1;min-width:0">'
        f'<div class="ai-hero-label">🤖 Análisis Profundo · Gemini AI</div>'
        f'<div class="ai-hero-title">{esc(app_name)}</div>'
        f'</div></div></div>',
        unsafe_allow_html=True,
    )

    if not ai_analysis or ai_analysis.startswith("Error"):
        if not api_key:
            st.markdown(
                '<div style="text-align:center;padding:3rem 0">'
                '<div style="font-size:2.5rem;margin-bottom:0.5rem">🔑</div>'
                '<p style="color:#64748b;font-size:0.95rem;margin-bottom:0.3rem">'
                'API Key de Gemini no configurada</p>'
                '<p style="color:#94a3b8;font-size:0.8rem">Configurá <code>GEMINI_API_KEY</code> '
                'en <code>.streamlit/secrets.toml</code> o como variable de entorno.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning(ai_analysis or "No se pudo generar el análisis.")
        return

    safe_name = safe_filename(app_name)
    col_md, col_pdf, _ = st.columns([1.3, 1.3, 2.4])
    with col_md:
        st.download_button(
            "📝 Descargar Markdown",
            data=ai_analysis.encode("utf-8"),
            file_name=f"analisis_{safe_name}.md",
            mime="text/markdown",
            use_container_width=True,
            key="ai_dl_md",
        )
    with col_pdf:
        try:
            pdf_bytes = build_pdf_bytes(app_name, ai_analysis)
            st.download_button(
                "📄 Descargar PDF",
                data=pdf_bytes,
                file_name=f"analisis_{safe_name}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="ai_dl_pdf",
            )
        except Exception:
            pass

    # --- Key Insight callout (TL;DR derivado del Resumen Ejecutivo) ---
    insight = _extract_key_insight(ai_analysis)
    if insight:
        st.markdown(
            f'<div class="ai-key-insight">'
            f'<div class="kicker">💡 Insight principal</div>'
            f'<div class="body">{esc(insight)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # --- Normalizar la salida de la LLM ---
    # A veces Gemini agrega su propio H1 ("# Análisis Profundo de...") o un
    # ## con título genérico antes del Resumen Ejecutivo. Lo promovemos a un
    # título inline del panel y no a un expander, para no duplicar el hero.
    normalized = ai_analysis.strip()
    # 1. Si arranca con H1, lo extraemos y mostramos como título plano.
    h1_match = re.match(r"^#\s+(.+?)\s*$", normalized, flags=re.MULTILINE)
    leading_title: Optional[str] = None
    if h1_match and normalized.startswith("#"):
        leading_title = h1_match.group(1).strip()
        normalized = normalized[h1_match.end():].lstrip()
    # 2. Si el primer ## no está en las secciones canónicas, también lo
    #    promovemos a título inline.
    first_h2 = re.match(r"^##\s+(.+?)\s*\n", normalized)
    if first_h2:
        first_title = first_h2.group(1).strip().lower()
        if not any(k in first_title for k in _AI_CANONICAL_KEYWORDS):
            leading_title = leading_title or first_h2.group(1).strip()
            # Quitamos esa línea y el bloque hasta el próximo ##.
            tail = normalized[first_h2.end():]
            next_h2 = re.search(r"(?=^##\s)", tail, flags=re.MULTILINE)
            if next_h2:
                normalized = tail[next_h2.start():]
            else:
                normalized = tail

    if leading_title:
        st.markdown(
            f'<div class="ai-report-title">{esc(leading_title)}</div>',
            unsafe_allow_html=True,
        )

    # --- Renderizar secciones canónicas como expanders ---
    section_icons = {"resumen": "📋", "fortaleza": "💪", "problema": "⚠️",
                     "perfil": "👥", "recomendac": "🎯"}
    sections = re.split(r"(?=^## )", normalized, flags=re.MULTILINE)
    canonical_idx = 0
    for section in sections:
        section = section.strip()
        if not section:
            continue
        match = re.match(r"^## (.+)", section)
        if not match:
            # Texto previo a cualquier ## (preámbulo) — lo mostramos plano.
            st.markdown(section)
            continue
        sec_title = match.group(1).strip()
        sec_body = section[match.end():].strip()
        is_canonical = any(k in sec_title.lower() for k in _AI_CANONICAL_KEYWORDS)
        if not is_canonical:
            # Secciones extra de la LLM: título inline + cuerpo, sin expander.
            st.markdown(f'<div class="ai-report-title">{esc(sec_title)}</div>',
                        unsafe_allow_html=True)
            st.markdown(sec_body)
            continue
        icon = next(
            (ic for keyword, ic in section_icons.items() if keyword in sec_title.lower()),
            "📊",
        )
        with st.expander(f"{icon}   {sec_title}", expanded=(canonical_idx < 2)):
            st.markdown(sec_body)
            _render_ai_evidence(df, sec_title)
        canonical_idx += 1


def _build_rating_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rating over time, bucketed by data span.

    Does NOT filter by min-points: every bucket with at least 1 review is
    kept so the chart can always tell a story (even with sparse data).
    The x-axis range in the renderer is what enforces the 90-day window.
    """
    if DATE_COL not in df.columns or SCORE_COL not in df.columns:
        return pd.DataFrame()
    df_t = df.dropna(subset=[DATE_COL, SCORE_COL]).copy()
    if df_t.empty:
        return pd.DataFrame()
    span_days = (df_t[DATE_COL].max() - df_t[DATE_COL].min()).days
    if span_days > 120:
        df_t["bucket"] = df_t[DATE_COL].dt.to_period("M").dt.to_timestamp()
        label_fmt = "%Y-%m"
    elif span_days > 21:
        df_t["bucket"] = df_t[DATE_COL].dt.to_period("W").dt.start_time
        label_fmt = "%Y-%m-%d"
    else:
        df_t["bucket"] = df_t[DATE_COL].dt.normalize()
        label_fmt = "%Y-%m-%d"
    agg = (
        df_t.groupby("bucket")
        .agg(avg_score=(SCORE_COL, "mean"), count=(SCORE_COL, "size"))
        .reset_index()
        .sort_values("bucket")
        .tail(24)
    )
    agg["label"] = agg["bucket"].dt.strftime(label_fmt)
    return agg


def _compute_velocity(df: pd.DataFrame) -> Optional[str]:
    """Reviews-per-week estimate. Informative when data spans > 7 days."""
    if DATE_COL not in df.columns or df.empty:
        return None
    dates = df[DATE_COL].dropna()
    if dates.empty:
        return None
    span_days = (dates.max() - dates.min()).days
    if span_days < 7:
        return None
    per_week = len(df) / max(1, span_days / 7)
    return f"~{per_week:.0f} por semana"


def _compute_rating_trend(df: pd.DataFrame) -> Optional[str]:
    """Compare rating of the last 30 days vs the prior 30 days."""
    if DATE_COL not in df.columns or SCORE_COL not in df.columns:
        return None
    d = df.dropna(subset=[DATE_COL, SCORE_COL])
    if d.empty:
        return None
    latest = d[DATE_COL].max()
    if pd.isna(latest):
        return None
    cut1 = latest - pd.Timedelta(days=30)
    cut2 = latest - pd.Timedelta(days=60)
    recent = d[d[DATE_COL] > cut1]
    prior = d[(d[DATE_COL] <= cut1) & (d[DATE_COL] > cut2)]
    if len(recent) < 5 or len(prior) < 5:
        return None
    delta = recent[SCORE_COL].mean() - prior[SCORE_COL].mean()
    if abs(delta) < 0.05:
        return "estable vs. 30d previos"
    arrow = "▲" if delta > 0 else "▼"
    return f"{arrow} {abs(delta):.2f}★ vs. 30d previos"


def _render_metrics_tab(df: pd.DataFrame, total: int, topics: list[dict], avg_score: float,
                        n_pos: int, n_neg: int, n_neu: int):
    velocity = _compute_velocity(df)
    trend = _compute_rating_trend(df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_stat("Total Reseñas", f"{total:,}", velocity or "")
    with c2:
        render_stat("Rating Promedio", f"{avg_score:.2f}/5", trend or "")
    with c3:
        render_stat(
            "Sentimiento Positivo",
            f"{n_pos / total * 100:.1f}%",
            f"{n_pos:,} reseñas",
            tone="positive",
        )
    with c4:
        render_stat(
            "Sentimiento Negativo",
            f"{n_neg / total * 100:.1f}%",
            f"{n_neg:,} reseñas",
            tone="negative",
        )
    st.markdown("<br>", unsafe_allow_html=True)

    col_d, col_r = st.columns(2)
    with col_d:
        st.markdown(
            '<div class="chart-card">'
            '<div class="chart-card-title">Distribución de Sentimiento</div>'
            f'<div class="chart-card-kicker">{total:,} reseñas</div>',
            unsafe_allow_html=True,
        )
        sent_df = pd.DataFrame({
            "Sentimiento": ["Positivo", "Neutro", "Negativo"],
            "Cantidad": [n_pos, n_neu, n_neg],
        })
        fig = px.pie(
            sent_df, values="Cantidad", names="Sentimiento", color="Sentimiento",
            color_discrete_map={
                "Positivo": COLORS["positive"],
                "Neutro": COLORS["neutral"],
                "Negativo": COLORS["negative"],
            },
            hole=0.55,
        )
        fig.update_layout(**CHART_LAYOUT, showlegend=True, height=260,
                          legend=dict(orientation="h", y=-0.1))
        fig.update_traces(
            textinfo="percent", textposition="outside", textfont_size=12,
            marker=dict(line=dict(color="white", width=2)),
            pull=[0.02, 0.02, 0.02],
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        kicker_avg = f"media {avg_score:.2f}★" if avg_score else "sin datos"
        st.markdown(
            '<div class="chart-card">'
            '<div class="chart-card-title">Distribución de Ratings</div>'
            f'<div class="chart-card-kicker">{kicker_avg}</div>',
            unsafe_allow_html=True,
        )
        if SCORE_COL in df.columns:
            sc = df[SCORE_COL].value_counts().sort_index()
            rating_df = pd.DataFrame({"rating": [f"{int(i)}★" for i in sc.index],
                                      "count": sc.values})
            fig2 = px.bar(rating_df, x="rating", y="count",
                          labels={"rating": "", "count": "Cantidad"})
            fig2.update_traces(marker_color=VIOLET, marker_line_color=VIOLET_DARK,
                               marker_line_width=0.5, hovertemplate="%{y} reseñas<extra></extra>")
            fig2.update_layout(**CHART_LAYOUT, showlegend=False, height=260, bargap=0.35)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    timeline = _build_rating_timeline(df)
    # Alcance temporal real de la muestra: nos dice si tiene sentido pedirle
    # tendencia al gráfico o si la data es demasiado reciente/apretada.
    if not timeline.empty:
        data_span = timeline["bucket"].max() - timeline["bucket"].min()
        data_span_days = max(0, data_span.days)
    else:
        data_span, data_span_days = pd.Timedelta(0), 0
    tl_kicker = (
        f"{len(timeline)} puntos · {data_span_days} días cubiertos"
        if not timeline.empty else "sin datos"
    )
    st.markdown(
        '<div class="chart-card">'
        '<div class="chart-card-title">Evolución del Rating</div>'
        f'<div class="chart-card-kicker">{tl_kicker}</div>',
        unsafe_allow_html=True,
    )
    # Aviso cuando la muestra viene concentrada — evita que el usuario
    # se lleve la impresión de que "el gráfico no funciona". Sugiere la
    # combinación que sí va a dar una tendencia legible.
    if not timeline.empty and data_span_days < 14:
        st.markdown(
            '<div style="background:#fef3c7;border:1px solid #fde68a;'
            'border-radius:8px;padding:0.5rem 0.8rem;margin:0 0 0.8rem;'
            'color:#92400e;font-size:0.78rem;text-align:center">'
            '💡 Muestra concentrada en pocos días — probá "Más relevantes" '
            'o aumentá la cantidad de reseñas para ver tendencia histórica.'
            '</div>',
            unsafe_allow_html=True,
        )
    if not timeline.empty:
        # Rating en el eje primario (más importante), reseñas en el eje
        # secundario con barras tenues — el foco es la línea de rating.
        # La ventana temporal visible es SIEMPRE de al menos 90 días, aunque
        # los datos estén concentrados en pocos días (así el gráfico no
        # colapsa a una franja ilegible).
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(
            go.Bar(
                x=timeline["bucket"], y=timeline["count"], name="Reseñas (volumen)",
                marker_color="rgba(124,58,237,0.14)", marker_line_width=0,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y} reseñas<extra></extra>",
            ),
            secondary_y=True,
        )
        fig3.add_trace(
            go.Scatter(
                x=timeline["bucket"], y=timeline["avg_score"],
                mode="lines+markers" if len(timeline) >= 2 else "markers",
                name="Rating promedio",
                line=dict(color=VIOLET, width=3, shape="spline", smoothing=0.6),
                marker=dict(size=9, color=VIOLET, line=dict(color="white", width=2)),
                fill="tozeroy" if len(timeline) >= 2 else None,
                fillcolor="rgba(124,58,237,0.08)",
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f} ★<extra></extra>",
            ),
            secondary_y=False,
        )
        # Línea de media como referencia si hay suficientes puntos.
        if len(timeline) >= 4:
            mean_score = float(timeline["avg_score"].mean())
            fig3.add_hline(
                y=mean_score, line_dash="dot", line_color="rgba(124,58,237,0.35)",
                line_width=1, secondary_y=False,
                annotation_text=f"Media {mean_score:.2f}★",
                annotation_position="top right",
                annotation_font=dict(size=10, color=VIOLET_DARK),
            )
        fig3.update_layout(
            **CHART_LAYOUT, height=310,
            legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
            hovermode="x unified", bargap=0.35,
        )
        # Rating Y: ventana mínima de 1.2★ alrededor del dato para que
        # variaciones chicas sean legibles, sin salirse del rango 1-5.
        y_data_min = float(timeline["avg_score"].min())
        y_data_max = float(timeline["avg_score"].max())
        y_min = max(1.0, y_data_min - 0.4)
        y_max = min(5.0, y_data_max + 0.4)
        if (y_max - y_min) < 1.2:
            mid = (y_data_min + y_data_max) / 2
            y_min, y_max = max(1.0, mid - 0.6), min(5.0, mid + 0.6)
        fig3.update_yaxes(
            title_text="Rating ★", range=[y_min, y_max], secondary_y=False,
            showgrid=True, gridcolor="#f1f5f9", tickfont=dict(color=VIOLET_DARK),
        )
        fig3.update_yaxes(
            title_text="Reseñas", secondary_y=True, showgrid=False,
            tickfont=dict(color="#94a3b8"),
        )
        # Ventana X adaptativa: si los datos cubren X días, la ventana se
        # escala para que el usuario siempre vea contexto alrededor sin que
        # la línea quede aplastada contra un borde. Datos concentrados ⇒
        # ventana chica (no 90 días de vacío); datos amplios ⇒ todo el rango.
        latest = timeline["bucket"].max()
        earliest_data = timeline["bucket"].min()
        if data_span_days < 7:
            window_days = 21
        elif data_span_days < 21:
            window_days = 45
        elif data_span_days < 60:
            window_days = 90
        elif data_span_days < 180:
            window_days = data_span_days + 14
        else:
            window_days = data_span_days + 20
        pad = pd.Timedelta(days=2)
        x_start = min(earliest_data - pad, latest - pd.Timedelta(days=window_days))
        x_end = latest + pad
        fig3.update_xaxes(
            showgrid=False, tickformat="%Y-%m-%d",
            range=[x_start, x_end],
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown(
            '<p style="text-align:center;color:#94a3b8;font-size:0.85rem;padding:1rem 0">'
            'No hay fechas válidas en esta muestra para trazar la evolución.</p>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Distribución de temas: cuando hay temas de la IA, los mostramos junto
    # a la nube de palabras (dos lentes complementarios — uno interpretado,
    # uno crudo). Si NO hay IA disponible, la nube ocupa todo el ancho: no
    # tiene sentido duplicar el mismo input con dos visualizaciones iguales.
    if topics:
        col_topics, col_wc = st.columns([1.2, 1])
        with col_topics:
            n_topics_shown = min(len(topics), 8)
            st.markdown(
                '<div class="chart-card">'
                '<div class="chart-card-title">Temas detectados por IA</div>'
                f'<div class="chart-card-kicker">top {n_topics_shown}</div>',
                unsafe_allow_html=True,
            )
            topics_df = pd.DataFrame(topics[:n_topics_shown])
            if "topic" in topics_df.columns and "count" in topics_df.columns:
                fig4 = px.bar(
                    topics_df, x="count", y="topic", orientation="h",
                    labels={"count": "Menciones aprox.", "topic": ""},
                )
                colors_list = [COLORS.get(t.get("sentiment", ""), VIOLET)
                               for t in topics[:n_topics_shown]]
                fig4.update_traces(
                    marker_color=colors_list,
                    hovertemplate="<b>%{y}</b><br>%{x} menciones<extra></extra>",
                )
                fig4.update_layout(
                    **CHART_LAYOUT, height=290, showlegend=False,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig4, use_container_width=True,
                                config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with col_wc:
            st.markdown(
                '<div class="chart-card">'
                '<div class="chart-card-title">Nube de Palabras</div>'
                '<div class="chart-card-kicker">términos crudos</div>',
                unsafe_allow_html=True,
            )
            render_html_wordcloud(get_word_freq(df, 60))
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="chart-card">'
            '<div class="chart-card-title">Nube de Palabras</div>'
            '<div class="chart-card-kicker">términos más frecuentes</div>',
            unsafe_allow_html=True,
        )
        render_html_wordcloud(get_word_freq(df, 80))
        st.markdown("</div>", unsafe_allow_html=True)


def _reset_rev_page():
    st.session_state["rev_page"] = 1


def _render_reviews_tab(df: pd.DataFrame):
    col_s, col_f1, col_f2, col_cnt = st.columns([3, 1.5, 1.5, 1])
    with col_s:
        rev_search = st.text_input(
            "Buscar", placeholder="Buscar en reseñas...",
            label_visibility="collapsed", key="rev_search",
            on_change=_reset_rev_page,
        )
    with col_f1:
        sent_f = st.selectbox("Sent", ["Todos", "Positivo", "Neutro", "Negativo"],
                              label_visibility="collapsed", key="rev_sent",
                              on_change=_reset_rev_page)
    with col_f2:
        score_f = st.selectbox("Score", ["Todos"] + [f"{i}★" for i in range(5, 0, -1)],
                               label_visibility="collapsed", key="rev_score",
                               on_change=_reset_rev_page)

    filtered = df.copy()
    if rev_search:
        filtered = filtered[
            filtered[CONTENT_COL].str.lower().str.contains(
                rev_search.lower(), na=False, regex=False,
            )
        ]
    if sent_f != "Todos":
        m = {"Positivo": "positive", "Neutro": "neutral", "Negativo": "negative"}
        filtered = filtered[filtered["sentiment"] == m[sent_f]]
    if score_f != "Todos" and SCORE_COL in filtered.columns:
        filtered = filtered[filtered[SCORE_COL] == int(score_f[0])]
    with col_cnt:
        st.markdown(
            f'<div style="padding-top:0.5rem;color:#94a3b8;font-size:0.78rem;text-align:right">'
            f'{len(filtered):,} reseña{"s" if len(filtered) != 1 else ""}</div>',
            unsafe_allow_html=True,
        )

    per_page = 20
    total_p = max(1, (len(filtered) + per_page - 1) // per_page)
    pg = min(st.session_state.get("rev_page", 1), total_p)
    page_df = filtered.iloc[(pg - 1) * per_page: pg * per_page]

    for _, row in page_df.iterrows():
        content = str(row.get(CONTENT_COL, ""))
        score = int(row.get(SCORE_COL, 0) or 0)
        sent = row.get("sentiment", "neutral")
        uname = row.get("userName", "Anónimo")
        date = row.get(DATE_COL, None)
        ver = row.get(VERSION_COL, "")
        likes = int(row.get(LIKES_COL, 0) or 0)

        stars = "".join(
            f'<span style="color:{"#f59e0b" if i < score else "#cbd5e1"};font-size:0.65rem">★</span>'
            for i in range(5)
        )
        badge = f'<span class="{BADGE_MAP.get(sent, "")}">{LABEL_MAP.get(sent, esc(sent))}</span>'
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d") if pd.notna(date) else ""
        ver_html = (
            f'<div style="font-size:0.68rem;color:#cbd5e1;margin-top:0.2rem">v{esc(ver)}</div>'
            if ver else ""
        )
        likes_html = f'<span style="color:#94a3b8;font-size:0.7rem">👍 {likes}</span>' if likes else ""

        preview_raw = content[:500] + ("..." if len(content) > 500 else "")
        preview = esc(preview_raw).replace("\n", "<br>")
        st.markdown(
            f'<div class="review-card">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.4rem">'
            f'<span style="font-size:0.78rem;font-weight:600;color:#334155">{esc(uname)}</span>'
            f'<span>{stars}</span> {badge}'
            f'<span style="margin-left:auto;font-size:0.7rem;color:#94a3b8">{likes_html} {date_str}</span></div>'
            f'<div style="font-size:0.8rem;color:#475569;line-height:1.6">{preview}</div>'
            f'{ver_html}</div>',
            unsafe_allow_html=True,
        )

    if total_p > 1:
        col_pv, col_pi, col_nx = st.columns([1, 3, 1])
        with col_pv:
            if st.button("← Anterior", disabled=pg <= 1, key="pg_prev", use_container_width=True):
                st.session_state["rev_page"] = pg - 1
                st.rerun()
        with col_pi:
            st.markdown(
                f'<div style="text-align:center;padding-top:0.5rem;color:#64748b;font-size:0.8rem">'
                f'Página {pg} de {total_p}</div>',
                unsafe_allow_html=True,
            )
        with col_nx:
            if st.button("Siguiente →", disabled=pg >= total_p, key="pg_next", use_container_width=True):
                st.session_state["rev_page"] = pg + 1
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    export_cols = [c for c in [CONTENT_COL, "sentiment", SCORE_COL, LIKES_COL, DATE_COL,
                               VERSION_COL, "userName"] if c in filtered.columns]
    csv = filtered[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV", data=csv, file_name="reviews.csv", mime="text/csv")


def render_dashboard():
    df = st.session_state.get("df_analyzed")
    if df is None or df.empty:
        st.warning("No hay un análisis cargado. Volvé al inicio para analizar una app.")
        if st.button("← Inicio"):
            go_to("home")
        return

    topics = st.session_state.get("topics", [])
    ai_analysis = st.session_state.get("ai_analysis", "")
    api_key = get_api_key()

    total = len(df)
    counts = df["sentiment"].value_counts()
    n_pos = counts.get("positive", 0)
    n_neg = counts.get("negative", 0)
    n_neu = counts.get("neutral", 0)
    avg_score = df[SCORE_COL].mean() if SCORE_COL in df.columns else 0
    app_name = st.session_state.get("sel_app_name", "")

    _render_dashboard_header(app_name, total)

    tab_met, tab_ia, tab_rev = st.tabs(
        ["📊 Métricas", "🤖 Análisis IA", f"💬 Reseñas ({total:,})"]
    )
    # Tab label is fixed at render time; the filtered count is shown inside the tab.
    with tab_met:
        _render_metrics_tab(df, total, topics, avg_score, n_pos, n_neg, n_neu)
    with tab_ia:
        _render_ai_tab(ai_analysis, app_name, api_key, df=df)
    with tab_rev:
        _render_reviews_tab(df)


# ---------- Router ----------

PAGES = {
    "home": render_home,
    "loading": render_loading,
    "history": render_history,
    "dashboard": render_dashboard,
}

PAGES.get(st.session_state["page"], render_home)()
