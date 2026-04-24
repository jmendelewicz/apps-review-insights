from __future__ import annotations

import os
import re
import sys
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
VIOLET_LIGHT = "#ede9fe"
VIOLET_SOFT = "#f5f3ff"

COUNTRIES = {
    "Argentina": "ar", "México": "mx", "España": "es", "Colombia": "co",
    "Chile": "cl", "Perú": "pe", "Uruguay": "uy", "Paraguay": "py",
    "Estados Unidos": "us", "Brasil": "br",
}
LANGUAGES = {"Español": "es", "English": "en", "Português": "pt"}

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
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    h1,h2,h3,h4 { font-family: 'Sora', 'Inter', sans-serif !important; font-weight: 700 !important; color: var(--violet-dark) !important; }
    .block-container { padding-top: 3rem; max-width: 1180px; margin: 0 auto; }
    header[data-testid="stHeader"] { background: white; border-bottom: 1px solid #e2e8f0; }
    div[data-testid="stSidebar"] { display: none; }

    button[kind="primary"], .stButton button[kind="primary"] {
        background: var(--violet) !important; border-color: var(--violet) !important;
    }
    button[kind="primary"]:hover { background: var(--violet-dark) !important; }

    .stat-card {
        background: linear-gradient(135deg, var(--violet) 0%, var(--violet-dark) 100%);
        border: none; border-radius: 16px; padding: 1.3rem 1.3rem 1.4rem;
        box-shadow: 0 4px 14px rgba(124,58,237,0.18);
        min-height: 112px; display: flex; flex-direction: column; justify-content: center;
        color: white;
    }
    .stat-card .label { font-size: 0.68rem; color: rgba(255,255,255,0.8); text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 0.3rem; }
    .stat-card .value { font-size: 1.7rem; font-weight: 700; color: white; font-family: 'Sora', 'Inter', sans-serif; line-height: 1.1; }
    .stat-card .sub { font-size: 0.75rem; color: rgba(255,255,255,0.85); margin-top: 0.3rem; }

    .chart-card { background: white; border: 1px solid #e2e8f0; border-radius: 14px; padding: 1.2rem 1.3rem; margin-bottom: 0.8rem; box-shadow: 0 1px 2px rgba(15,23,42,0.03); }
    .chart-card-title {
        font-family: 'Sora', 'Inter', sans-serif;
        font-size: 0.85rem; color: var(--violet-dark); font-weight: 700;
        letter-spacing: 0.5px; margin-bottom: 0.9rem; text-align: center;
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

    .hero-label { font-size: 0.72rem; color: var(--violet); text-transform: uppercase; letter-spacing: 2.5px; font-weight: 700; text-align: center; }
    .hero-title {
        font-family: 'Sora', 'Inter', sans-serif;
        font-size: 2.7rem; font-weight: 800; color: var(--violet-dark);
        line-height: 1.1; margin: 0.5rem 0; text-align: center;
    }
    .hero-sub { font-size: 0.95rem; color: #64748b; text-align: center; }

    .section-title {
        font-family: 'Sora', 'Inter', sans-serif;
        font-size: 1.4rem; font-weight: 700; color: var(--violet-dark);
        text-align: center; margin: 1.2rem 0 0.4rem;
    }
    .section-sub { text-align: center; color: #64748b; font-size: 0.88rem; margin-bottom: 1.3rem; }

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

    div[data-baseweb="tab-list"] { gap: 0.3rem; }
    button[data-baseweb="tab"] { font-family: 'Sora', 'Inter', sans-serif; font-weight: 600; }
    button[data-baseweb="tab"][aria-selected="true"] { color: var(--violet) !important; }
    div[data-baseweb="tab-highlight"] { background: var(--violet) !important; }
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
    return _secret("GEMINI_API_KEY")


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

def render_stat(label: str, value: str, sub: str = "", color: str = "#64748b"):
    sub_html = f'<div class="sub" style="color:{color}">{sub}</div>' if sub else ""
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
    })


# ---------- Navbar ----------

current_page = st.session_state["page"]
col_logo, _, col_nav = st.columns([2, 4, 2])
with col_logo:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0.5rem;padding-top:0.2rem">'
        '<div style="width:28px;height:28px;background:#0f172a;border-radius:4px;'
        'display:flex;align-items:center;justify-content:center">'
        '<span style="color:white;font-size:0.8rem">📊</span></div>'
        '<span style="font-weight:700;font-size:1rem;color:#0f172a">PlayInsights</span></div>',
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

        col_lang, col_count, col_countries, col_sort = st.columns(4)
        with col_lang:
            st.markdown('<span style="font-size:0.72rem;color:#64748b;font-weight:500">Idioma:</span>',
                        unsafe_allow_html=True)
            lang_name = st.selectbox("Idioma", list(LANGUAGES.keys()), index=0,
                                     label_visibility="collapsed", key="home_lang")
        with col_count:
            st.markdown('<span style="font-size:0.72rem;color:#64748b;font-weight:500">Reseñas:</span>',
                        unsafe_allow_html=True)
            review_count = st.selectbox(
                "Reseñas", [100, 200, 500, 1000], index=0,
                label_visibility="collapsed", format_func=lambda x: f"{x:,} reseñas",
                key="home_count",
            )
        with col_countries:
            st.markdown('<span style="font-size:0.72rem;color:#64748b;font-weight:500">Países:</span>',
                        unsafe_allow_html=True)
            sel_countries = st.multiselect(
                "Países", list(COUNTRIES.keys()), default=["Argentina"],
                label_visibility="collapsed", key="home_countries",
            )
        with col_sort:
            st.markdown('<span style="font-size:0.72rem;color:#64748b;font-weight:500">Ordenar por:</span>',
                        unsafe_allow_html=True)
            sort_order = st.selectbox("Orden", ["Más recientes", "Más relevantes"],
                                      index=0, label_visibility="collapsed", key="home_sort")

    if not query.strip():
        st.session_state["search_results"] = None

    if search_clicked and query.strip():
        lang_code = LANGUAGES[lang_name]
        country_codes = tuple(COUNTRIES[c] for c in sel_countries) or ("us",)
        fallback_countries = tuple(dict.fromkeys(country_codes + ("us", "ar")))
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
                    f'padding:1px 6px;border-radius:3px;margin-left:0.5rem">{genre}</span>'
                    if genre else ""
                )
                score_html = (
                    f'<span style="font-size:0.7rem;color:#64748b">⭐ {score:.1f}</span>'
                    if score else ""
                )
                inst_html = (
                    f'<span style="font-size:0.7rem;color:#94a3b8">⬇ {installs}</span>'
                    if installs and installs != "N/A" else ""
                )
                st.markdown(
                    f'<div class="app-card">'
                    f'<img src="{icon}" style="width:48px;height:48px;border-radius:8px" '
                    f'onerror="this.style.display=\'none\'">'
                    f'<div style="flex:1;min-width:0">'
                    f'<div style="display:flex;align-items:center;gap:0.3rem">'
                    f'<span style="font-weight:600;font-size:0.9rem;color:#0f172a">{title}</span>{badge}</div>'
                    f'<div style="font-size:0.75rem;color:#64748b">{dev}</div>'
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
                        sel_lang=LANGUAGES[lang_name],
                        sel_count=review_count,
                        sel_countries=sel_countries or ["Argentina"],
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

    clean_id = extract_app_id(app_id) or str(app_id).strip()
    country_codes = tuple(COUNTRIES[c] for c in sel_countries)
    lookup_countries = tuple(dict.fromkeys(country_codes + ("us", "ar")))

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
        f'<h2 class="section-title" style="margin-top:0">Analizando {app_name}</h2>'
        f'<p style="color:#64748b;font-size:0.9rem">Esto puede tardar unos minutos...</p>'
        f'</div>'
        f'<style>@keyframes spin {{ 0%{{transform:rotate(0deg)}} 100%{{transform:rotate(360deg)}} }}</style>',
        unsafe_allow_html=True,
    )

    progress = st.progress(0, text="Iniciando...")
    n_countries = len(sel_countries) or 1
    per_country = max(50, count // n_countries)

    dfs = []
    for i, country_name in enumerate(sel_countries):
        pct = int(10 + (i / n_countries) * 40)
        progress.progress(pct, text=f"📱 Descargando reseñas de {country_name} ({per_country})...")
        try:
            df_part = cached_scrape_reviews(
                clean_id, lang, COUNTRIES[country_name], country_name, per_country, sort_label,
            )
            if not df_part.empty:
                dfs.append(df_part)
        except Exception as e:
            st.warning(f"Error en {country_name}: {e}")

    if not dfs:
        st.error("No se pudieron obtener reseñas. Verificá el package ID o probá otro país.")
        if st.button("← Volver"):
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
        header = f"📊   {entry['name']}   ·   {entry['count']:,} reseñas{score_txt}"
        with st.expander(header, expanded=False):
            col_meta, col_open = st.columns([3, 1.2])
            with col_meta:
                icon_html = (
                    f'<img src="{entry["icon"]}" style="width:44px;height:44px;border-radius:8px">'
                    if entry.get("icon") else ""
                )
                n_topics = len(entry.get("topics", []))
                ai_flag = (
                    '<span style="background:var(--violet-light);color:var(--violet-dark);'
                    'padding:2px 8px;border-radius:4px;font-size:0.7rem;font-weight:600">🧠 Análisis IA</span>'
                    if entry.get("ai_analysis") and not entry["ai_analysis"].startswith("Error") else ""
                )
                topics_html = (
                    f'<span style="color:#64748b;font-size:0.75rem;margin-left:0.5rem">'
                    f'{n_topics} temas detectados</span>' if n_topics else ""
                )
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:0.9rem;padding:0.4rem 0">'
                    f'{icon_html}'
                    f'<div style="flex:1">'
                    f'<div style="font-size:0.78rem;color:#64748b">{entry.get("dev", "")} · {entry.get("genre", "")}</div>'
                    f'<div style="margin-top:0.4rem">{ai_flag}{topics_html}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            with col_open:
                st.markdown("<div style='padding-top:0.6rem'></div>", unsafe_allow_html=True)
                if st.button("📂 Abrir análisis", key=f"hist_open_{idx}",
                             use_container_width=True, type="primary"):
                    load_history_entry(idx)

            st.markdown(
                '<hr style="margin:0.8rem 0 0.6rem;border:none;border-top:1px solid var(--violet-light)">',
                unsafe_allow_html=True,
            )

            safe = safe_filename(entry["name"])
            ai_text = entry.get("ai_analysis", "")
            has_ai = bool(ai_text) and not ai_text.startswith("Error")
            df_entry = entry.get("df")

            b1, b2, b3, b4 = st.columns(4)
            with b1:
                if has_ai:
                    st.download_button(
                        "📝 Análisis (MD)",
                        data=ai_text.encode("utf-8"),
                        file_name=f"analisis_{safe}.md",
                        mime="text/markdown",
                        use_container_width=True,
                        key=f"hist_md_{idx}",
                    )
                else:
                    st.button("📝 Análisis (MD)", key=f"hist_md_dis_{idx}",
                              use_container_width=True, disabled=True)
            with b2:
                if has_ai:
                    try:
                        pdf_bytes = build_pdf_bytes(entry["name"], ai_text)
                        st.download_button(
                            "📄 Análisis (PDF)",
                            data=pdf_bytes,
                            file_name=f"analisis_{safe}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key=f"hist_pdf_{idx}",
                        )
                    except Exception:
                        st.button("📄 Análisis (PDF)", key=f"hist_pdf_err_{idx}",
                                  use_container_width=True, disabled=True)
                else:
                    st.button("📄 Análisis (PDF)", key=f"hist_pdf_dis_{idx}",
                              use_container_width=True, disabled=True)
            with b3:
                if df_entry is not None and not df_entry.empty:
                    export_cols = [c for c in [CONTENT_COL, "sentiment", SCORE_COL, LIKES_COL,
                                               DATE_COL, VERSION_COL, "userName"]
                                   if c in df_entry.columns]
                    csv_bytes = df_entry[export_cols].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Reseñas (CSV)",
                        data=csv_bytes,
                        file_name=f"reviews_{safe}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key=f"hist_csv_{idx}",
                    )
                else:
                    st.button("⬇️ Reseñas (CSV)", key=f"hist_csv_dis_{idx}",
                              use_container_width=True, disabled=True)
            with b4:
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
        icon_html = f'<img src="{app_icon}" style="width:48px;height:48px;border-radius:4px">' if app_icon else ""
        genre_html = f' · {app_genre}' if app_genre else ""
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:1rem">{icon_html}'
            f'<div><div style="font-weight:700;font-size:1.2rem;color:#0f172a">{app_name}</div>'
            f'<div style="font-size:0.78rem;color:#64748b">{app_dev}{genre_html}</div></div></div>',
            unsafe_allow_html=True,
        )
    with col_rating:
        score_html = (
            f'<span style="background:#f8fafc;border:1px solid #e2e8f0;padding:3px 10px;'
            f'border-radius:4px;font-weight:600;font-size:0.85rem">⭐ {app_play_score:.1f}</span>'
            if app_play_score else ""
        )
        inst_html = (
            f'<span style="color:#94a3b8;font-size:0.78rem;margin-left:0.5rem">{app_installs}</span>'
            if app_installs else ""
        )
        st.markdown(
            f'<div style="text-align:right;padding-top:0.5rem">{score_html}{inst_html}</div>',
            unsafe_allow_html=True,
        )


def _render_ai_tab(ai_analysis: str, app_name: str, api_key: str):
    app_icon = st.session_state.get("sel_app_icon", "")
    icon_html = (
        f'<img src="{app_icon}" style="width:54px;height:54px;border-radius:12px;'
        f'border:2px solid rgba(255,255,255,0.3);flex-shrink:0">' if app_icon else ""
    )
    st.markdown(
        f'<div class="ai-hero">'
        f'<div style="display:flex;align-items:center;gap:1rem">'
        f'{icon_html}'
        f'<div style="flex:1;min-width:0">'
        f'<div class="ai-hero-label">🤖 Análisis Profundo · Gemini AI</div>'
        f'<div class="ai-hero-title">{app_name}</div>'
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

    st.markdown("<br>", unsafe_allow_html=True)

    section_icons = {"resumen": "📋", "fortaleza": "💪", "problema": "⚠️",
                     "perfil": "👥", "recomendac": "🎯"}
    sections = re.split(r"(?=^## )", ai_analysis, flags=re.MULTILINE)
    section_idx = 0
    for section in sections:
        section = section.strip()
        if not section:
            continue
        match = re.match(r"^## (.+)", section)
        if match:
            sec_title = match.group(1).strip()
            sec_body = section[match.end():].strip()
            icon = next(
                (ic for keyword, ic in section_icons.items() if keyword in sec_title.lower()),
                "📊",
            )
            with st.expander(f"{icon}   {sec_title}", expanded=(section_idx < 2)):
                st.markdown(sec_body)
            section_idx += 1
        else:
            st.markdown(section)


def _build_rating_timeline(df: pd.DataFrame) -> pd.DataFrame:
    if DATE_COL not in df.columns or SCORE_COL not in df.columns:
        return pd.DataFrame()
    df_t = df.dropna(subset=[DATE_COL, SCORE_COL]).copy()
    if df_t.empty:
        return pd.DataFrame()
    span_days = (df_t[DATE_COL].max() - df_t[DATE_COL].min()).days
    if span_days > 120:
        df_t["bucket"] = df_t[DATE_COL].dt.to_period("M").dt.to_timestamp()
        min_points, label_fmt = 3, "%Y-%m"
    elif span_days > 28:
        df_t["bucket"] = df_t[DATE_COL].dt.to_period("W").dt.start_time
        min_points, label_fmt = 2, "%Y-%m-%d"
    else:
        df_t["bucket"] = df_t[DATE_COL].dt.normalize()
        min_points, label_fmt = 1, "%Y-%m-%d"
    agg = (
        df_t.groupby("bucket")
        .agg(avg_score=(SCORE_COL, "mean"), count=(SCORE_COL, "size"))
        .reset_index()
    )
    agg = agg[agg["count"] >= min_points].sort_values("bucket").tail(24)
    agg["label"] = agg["bucket"].dt.strftime(label_fmt)
    return agg


def _render_metrics_tab(df: pd.DataFrame, total: int, topics: list[dict], avg_score: float,
                        n_pos: int, n_neg: int, n_neu: int):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_stat("Total Reseñas", f"{total:,}")
    with c2:
        render_stat("Rating Promedio", f"{avg_score:.2f}/5")
    with c3:
        render_stat("Sentimiento Positivo", f"{n_pos / total * 100:.1f}%", f"{n_pos:,} reseñas")
    with c4:
        render_stat("Sentimiento Negativo", f"{n_neg / total * 100:.1f}%", f"{n_neg:,} reseñas")
    st.markdown("<br>", unsafe_allow_html=True)

    col_d, col_r = st.columns(2)
    with col_d:
        st.markdown('<div class="chart-card"><div class="chart-card-title">Distribución de Sentimiento</div>',
                    unsafe_allow_html=True)
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
        st.markdown('<div class="chart-card"><div class="chart-card-title">Distribución de Ratings</div>',
                    unsafe_allow_html=True)
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
    st.markdown('<div class="chart-card"><div class="chart-card-title">Evolución del Rating en el Tiempo</div>',
                unsafe_allow_html=True)
    if len(timeline) >= 2:
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(
            go.Bar(
                x=timeline["bucket"], y=timeline["count"], name="Reseñas",
                marker_color="rgba(124,58,237,0.22)",
                hovertemplate="%{x|%Y-%m-%d}<br>%{y} reseñas<extra></extra>",
            ),
            secondary_y=False,
        )
        fig3.add_trace(
            go.Scatter(
                x=timeline["bucket"], y=timeline["avg_score"], mode="lines+markers",
                name="Rating promedio",
                line=dict(color=VIOLET, width=2.5),
                marker=dict(size=7, color=VIOLET, line=dict(color="white", width=1.5)),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f} ★<extra></extra>",
            ),
            secondary_y=True,
        )
        fig3.update_layout(
            **CHART_LAYOUT, height=280,
            legend=dict(orientation="h", y=1.1, x=0),
            hovermode="x unified",
        )
        fig3.update_yaxes(title_text="Reseñas", secondary_y=False, showgrid=False)
        fig3.update_yaxes(title_text="Rating", range=[1, 5.2], secondary_y=True,
                          showgrid=True, gridcolor="#f1f5f9")
        fig3.update_xaxes(showgrid=False, tickformat="%Y-%m-%d")
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown(
            '<p style="text-align:center;color:#94a3b8;font-size:0.85rem;padding:1rem 0">'
            'No hay suficiente variación temporal en esta muestra para trazar una evolución confiable.</p>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    col_topics, col_wc = st.columns(2)
    with col_topics:
        st.markdown('<div class="chart-card"><div class="chart-card-title">Temas Más Mencionados</div>',
                    unsafe_allow_html=True)
        if topics:
            topics_df = pd.DataFrame(topics[:8])
            if "topic" in topics_df.columns and "count" in topics_df.columns:
                fig4 = px.bar(
                    topics_df, x="count", y="topic", orientation="h",
                    labels={"count": "Menciones", "topic": ""},
                )
                colors_list = [COLORS.get(t.get("sentiment", ""), VIOLET) for t in topics[:8]]
                fig4.update_traces(marker_color=colors_list)
                fig4.update_layout(**CHART_LAYOUT, height=260, showlegend=False,
                                   yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
        else:
            wf = get_word_freq(df, 10)
            if wf:
                wdf = pd.DataFrame(wf, columns=["Palabra", "Frecuencia"])
                fig4 = px.bar(wdf, x="Frecuencia", y="Palabra", orientation="h")
                fig4.update_traces(marker_color=VIOLET)
                fig4.update_layout(**CHART_LAYOUT, height=260, showlegend=False,
                                   yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_wc:
        st.markdown('<div class="chart-card"><div class="chart-card-title">Nube de Palabras</div>',
                    unsafe_allow_html=True)
        render_html_wordcloud(get_word_freq(df, 60))
        st.markdown("</div>", unsafe_allow_html=True)


def _render_reviews_tab(df: pd.DataFrame):
    col_s, col_f1, col_f2, col_cnt = st.columns([3, 1.5, 1.5, 1])
    with col_s:
        rev_search = st.text_input(
            "Buscar", placeholder="Buscar en reseñas...",
            label_visibility="collapsed", key="rev_search",
        )
    with col_f1:
        sent_f = st.selectbox("Sent", ["Todos", "Positivo", "Neutro", "Negativo"],
                              label_visibility="collapsed", key="rev_sent")
    with col_f2:
        score_f = st.selectbox("Score", ["Todos"] + [f"{i}★" for i in range(5, 0, -1)],
                               label_visibility="collapsed", key="rev_score")

    filtered = df.copy()
    if rev_search:
        filtered = filtered[filtered[CONTENT_COL].str.lower().str.contains(rev_search.lower(), na=False)]
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
        badge = f'<span class="{BADGE_MAP.get(sent, "")}">{LABEL_MAP.get(sent, sent)}</span>'
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d") if pd.notna(date) else ""
        ver_html = (
            f'<div style="font-size:0.68rem;color:#cbd5e1;margin-top:0.2rem">v{ver}</div>'
            if ver else ""
        )
        likes_html = f'<span style="color:#94a3b8;font-size:0.7rem">👍 {likes}</span>' if likes else ""

        preview = content[:500] + ("..." if len(content) > 500 else "")
        st.markdown(
            f'<div class="review-card">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.4rem">'
            f'<span style="font-size:0.78rem;font-weight:600;color:#334155">{uname}</span>'
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
    with tab_met:
        _render_metrics_tab(df, total, topics, avg_score, n_pos, n_neg, n_neu)
    with tab_ia:
        _render_ai_tab(ai_analysis, app_name, api_key)
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
