"""
app.py  —  Metadata Cleaning Pipeline
Warner Chappell Music Intelligence · Task 1

Run: streamlit run app.py
"""

import io
import os
import csv
import json
import time

import streamlit as st
import pandas as pd

from pipeline import (
    EMAIL_A, EMAIL_B,
    run_pipeline,
    normalize_isrc, normalize_date,
)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC Metadata Cleaner",
    page_icon="♪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette (matches warnerchappell.com) ───────────────────────────────────────
BLACK    = "#000000"
BLACK2   = "#0d0d0d"
BLACK3   = "#1a1a1a"
GOLD     = "#F5C518"
GOLD2    = "#c9a010"
GOLD_DIM = "rgba(245,197,24,0.12)"
WHITE    = "#FFFFFF"
GREY1    = "#e0e0e0"
GREY2    = "#a0a0a0"
GREY3    = "#555555"
RED      = "#c0392b"
GREEN    = "#27ae60"
ORANGE   = "#e67e22"

# ── Global Styles ──────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=DM+Mono:wght@300;400&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    color: {WHITE};
}}
.stApp {{ background: {BLACK}; color: {WHITE}; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {BLACK2} !important;
    border-right: 1px solid {BLACK3} !important;
}}
section[data-testid="stSidebar"] > div {{ padding-top: 0 !important; }}

/* ── Brand block ── */
.wc-brand {{
    background: {BLACK};
    border-bottom: 2px solid {GOLD};
    padding: 20px 20px 16px;
    margin: -1rem -1rem 20px;
    position: relative;
    overflow: hidden;
}}
.wc-brand::before {{
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {GOLD}, transparent);
}}
.wc-eyebrow {{
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: {GOLD};
    margin-bottom: 6px;
}}
.wc-logo-text {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 28px;
    letter-spacing: 0.08em;
    line-height: 1;
    color: {WHITE};
    text-transform: uppercase;
}}
.wc-logo-text span {{ color: {GOLD}; }}
.wc-tagline {{
    font-size: 11px;
    font-style: italic;
    color: {GREY2};
    margin-top: 6px;
    font-weight: 300;
    letter-spacing: 0.02em;
}}

/* ── Section labels ── */
.wc-section {{
    font-family: 'DM Mono', monospace;
    font-size: 8px;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: {GREY3};
    margin: 20px 0 8px;
    padding-bottom: 5px;
    border-bottom: 1px solid {BLACK3};
}}

/* ── Main header ── */
.main-header {{
    border-bottom: 2px solid {GOLD};
    padding-bottom: 14px;
    margin-bottom: 32px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}}
.main-title {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 42px;
    letter-spacing: 0.08em;
    color: {WHITE};
    line-height: 1;
}}
.main-title span {{ color: {GOLD}; }}
.main-sub {{
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: {GREY3};
    letter-spacing: 0.15em;
    text-transform: uppercase;
}}
.task-badge {{
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.2em;
    color: {BLACK};
    background: {GOLD};
    padding: 3px 10px;
    text-transform: uppercase;
    margin-bottom: 6px;
    display: inline-block;
}}

/* ── Pipeline steps ── */
.step-card {{
    background: {BLACK2};
    border: 1px solid {BLACK3};
    border-top: 3px solid {GOLD};
    padding: 16px 20px;
    margin-bottom: 4px;
}}
.step-num {{
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.2em;
    color: {GOLD};
    text-transform: uppercase;
    margin-bottom: 4px;
}}
.step-title {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 18px;
    letter-spacing: 0.06em;
    color: {WHITE};
    margin-bottom: 6px;
}}
.step-desc {{
    font-size: 12px;
    color: {GREY2};
    font-weight: 300;
    line-height: 1.6;
}}

/* ── Source tags ── */
.source-pdf {{ color: {GOLD}; font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: 0.1em; text-transform: uppercase; background: rgba(245,197,24,0.08); padding: 2px 7px; }}
.source-email {{ color: #7fb3d3; font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: 0.1em; text-transform: uppercase; background: rgba(127,179,211,0.08); padding: 2px 7px; }}
.source-both {{ color: #82c596; font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: 0.1em; text-transform: uppercase; background: rgba(130,197,150,0.08); padding: 2px 7px; }}

/* ── Info / success / error boxes ── */
.info-box {{
    background: {BLACK2};
    border-left: 3px solid {GOLD};
    padding: 12px 18px;
    margin: 8px 0;
    font-size: 13px;
    color: {GREY1};
    font-weight: 300;
    line-height: 1.65;
}}
.success-box {{
    background: rgba(39,174,96,0.07);
    border-left: 3px solid {GREEN};
    padding: 12px 18px;
    margin: 8px 0;
    color: #a8e6c1;
    font-size: 13px;
    font-weight: 300;
}}
.warn-box {{
    background: rgba(230,126,34,0.07);
    border-left: 3px solid {ORANGE};
    padding: 12px 18px;
    margin: 8px 0;
    color: #f5c28a;
    font-size: 13px;
    font-weight: 300;
}}
.error-box {{
    background: rgba(192,57,43,0.08);
    border-left: 3px solid {RED};
    padding: 12px 18px;
    margin: 8px 0;
    color: #e07060;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    line-height: 1.6;
}}

/* ── Code / JSON blocks ── */
.code-block {{
    background: #050505;
    border: 1px solid {BLACK3};
    border-left: 3px solid {GREY3};
    padding: 14px 16px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 300;
    color: {GREY2};
    white-space: pre-wrap;
    line-height: 1.7;
    letter-spacing: 0.02em;
    max-height: 260px;
    overflow-y: auto;
}}

/* ── Stats strip ── */
.stats-strip {{
    display: flex;
    gap: 0;
    margin: 18px 0;
    border: 1px solid {BLACK3};
}}
.stat-item {{
    flex: 1;
    padding: 16px;
    text-align: center;
    border-right: 1px solid {BLACK3};
}}
.stat-item:last-child {{ border-right: none; }}
.stat-num {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 36px;
    color: {GOLD};
    line-height: 1;
}}
.stat-label {{
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: {GREY3};
    margin-top: 4px;
}}

/* ── Validation badge ── */
.badge-ok {{ display: inline-block; background: rgba(39,174,96,0.15); color: #82c596; font-family: 'DM Mono', monospace; font-size: 9px; padding: 2px 8px; letter-spacing: 0.1em; }}
.badge-warn {{ display: inline-block; background: rgba(230,126,34,0.15); color: #f5c28a; font-family: 'DM Mono', monospace; font-size: 9px; padding: 2px 8px; letter-spacing: 0.1em; }}
.badge-err {{ display: inline-block; background: rgba(192,57,43,0.15); color: #e07060; font-family: 'DM Mono', monospace; font-size: 9px; padding: 2px 8px; letter-spacing: 0.1em; }}

/* ── Empty state ── */
.empty-state {{
    text-align: center;
    padding: 80px 20px;
    color: {GREY3};
}}
.empty-mark {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 80px;
    color: {GOLD};
    line-height: 1;
    margin-bottom: 12px;
    opacity: 0.25;
}}
.empty-title {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 36px;
    letter-spacing: 0.08em;
    color: {WHITE};
    margin-bottom: 10px;
}}
.empty-sub {{
    font-size: 13px;
    line-height: 1.9;
    font-weight: 300;
    max-width: 400px;
    margin: 0 auto;
    color: {GREY2};
}}

/* ── Buttons ── */
.stButton > button {{
    background: transparent !important;
    border: 1px solid {GOLD} !important;
    border-radius: 0 !important;
    color: {GOLD} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 10px 24px !important;
    transition: all 0.15s ease !important;
}}
.stButton > button:hover {{
    background: {GOLD_DIM} !important;
}}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {{
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: {GREY3} !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 16px !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: {GOLD} !important;
    border-bottom-color: {GOLD} !important;
}}

/* ── Data table ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BLACK3} !important;
    border-top: 2px solid {GOLD} !important;
}}
[data-testid="stDataFrame"] th {{
    background: {BLACK2} !important;
    color: {GOLD} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}}
[data-testid="stDataFrame"] td {{
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    color: {GREY1} !important;
}}

/* ── Text area ── */
[data-testid="stTextArea"] textarea {{
    background: {BLACK2} !important;
    border: 1px solid {BLACK3} !important;
    border-bottom: 2px solid {GREY3} !important;
    border-radius: 0 !important;
    color: {GREY1} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    line-height: 1.6 !important;
}}
[data-testid="stTextArea"] textarea:focus {{
    border-bottom-color: {GOLD} !important;
    box-shadow: none !important;
}}

/* ── File uploader ── */
div[data-testid="stFileUploader"] {{
    border: 1px dashed {GREY3} !important;
    border-radius: 0 !important;
    background: {BLACK2} !important;
}}

/* ── Progress bar ── */
.stProgress > div > div {{ background-color: {GOLD} !important; }}

/* ── Select / radio ── */
[data-testid="stSelectbox"] div, [data-testid="stRadio"] label {{
    color: {GREY1} !important;
    font-size: 13px !important;
}}

/* ── Expander ── */
details summary {{
    color: {GREY2} !important;
    font-size: 12px !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
}}
details summary:hover {{ color: {GOLD} !important; }}

/* ── Spinner ── */
.stSpinner > div {{ border-top-color: {GOLD} !important; }}

/* ── Divider ── */
hr {{ border: none !important; border-top: 1px solid {BLACK3} !important; margin: 12px 0 !important; }}

/* ── Sidebar text ── */
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown {{ color: {GREY1} !important; }}

/* ── Sidebar input fields ── */
[data-testid="stSidebar"] input {{
    background: {BLACK3} !important;
    border: 1px solid {GREY3} !important;
    border-radius: 0 !important;
    color: {WHITE} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}}
</style>
""", unsafe_allow_html=True)

LOGO_URL = "https://music-row-website-assets.s3.amazonaws.com/wp-content/uploads/2019/05/10144300/WCM_Lockup_Black_Gold-TEX.png"


# ── Helper: CSV builder ────────────────────────────────────────────────────────
def build_csv(clean_records: list) -> str:
    output_cols = ["Song Title", "Writers", "Recording Artist", "ISRC", "Release Date"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=output_cols, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(clean_records)
    return buf.getvalue()


# ── Helper: LLM client factory ─────────────────────────────────────────────────
def get_client(provider: str, api_key: str, base_url: str = None):
    from openai import OpenAI
    if provider == "OpenAI":
        return OpenAI(api_key=api_key)
    elif provider == "Groq":
        return OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    elif provider == "Custom / Other":
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="wc-brand">
      <img src="{LOGO_URL}"
           alt="Warner Chappell Music"
           style="width:100%;max-width:190px;height:auto;margin-bottom:6px;display:block;" />
      <div class="wc-tagline">Metadata Intelligence Pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="wc-section">LLM Configuration</div>', unsafe_allow_html=True)

    provider = st.selectbox(
        "Provider",
        ["OpenAI", "Groq", "Custom / Other"],
        index=1,
        help="Groq is recommended (fast + free tier). OpenAI also works.",
    )

    api_key = st.text_input(
        "API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
        help="Enter your API key. Not stored between sessions.",
    )

    if provider == "OpenAI":
        model_default = "gpt-4o-mini"
        model_help = "gpt-4o-mini is fast and cheap. gpt-4o for best accuracy."
    elif provider == "Groq":
        model_default = "llama-3.3-70b-versatile"
        model_help = "llama-3.3-70b-versatile recommended. Also: mixtral-8x7b-32768"
    else:
        model_default = ""
        model_help = "Enter your model name"

    model = st.text_input("Model", value=model_default, help=model_help)

    if provider == "Custom / Other":
        base_url = st.text_input("Base URL", placeholder="https://api.example.com/v1")
    else:
        base_url = None

    st.markdown('<div class="wc-section">PDF Agreement</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:11px;color:{GREY3};font-weight:300;margin-bottom:8px;line-height:1.6">'
        f'Upload the Schedule A PDF agreement. This is the source of truth for conflicting data.'
        f'<br><br>No PDF? Run <code style="color:{GOLD};font-size:10px">python create_sample_pdf.py</code>'
        f' to generate a test file.</div>',
        unsafe_allow_html=True,
    )
    uploaded_pdf = st.file_uploader("", type=["pdf"], label_visibility="collapsed",
                                    key="pdf_upload")

    st.markdown('<div class="wc-section">Pipeline Options</div>', unsafe_allow_html=True)
    show_raw_extraction = st.checkbox("Show raw LLM extractions", value=False)
    show_validation_log = st.checkbox("Show validation log", value=True)
    show_merge_debug = st.checkbox("Show merge JSON debug", value=False)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↺  Reset", use_container_width=True):
        for key in ["pipeline_result", "email_a_edited", "email_b_edited"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
  <div>
    <div class="task-badge">Task 1</div>
    <div class="main-title">Metadata <span>Cleaning</span></div>
  </div>
  <div class="main-sub">Ingest · Extract · Merge · Validate · Deliver</div>
</div>
""", unsafe_allow_html=True)

# ── Pipeline overview cards ────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
steps = [
    ("01", "PDF Text Extract",  "pdfplumber pulls raw Schedule A page text — avoids column mis-detection"),
    ("02", "AI Parses PDF",     "LLM reads raw text with track-number anchors — handles wrapped cells correctly"),
    ("03", "AI Parses Emails",  "Schema-locked prompt extracts songs from both emails"),
    ("04", "Python Merge",      "ISRC-first matching · PDF wins every conflict · emails fill gaps only"),
]
for col, (num, title, desc) in zip([col1, col2, col3, col4], steps):
    with col:
        st.markdown(f"""
        <div class="step-card">
          <div class="step-num">Step {num}</div>
          <div class="step-title">{title}</div>
          <div class="step-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Source inputs ──────────────────────────────────────────────────────────────
st.markdown(f'<div class="wc-section">Source Documents</div>', unsafe_allow_html=True)

tab_pdf, tab_email_a, tab_email_b = st.tabs(["PDF Agreement", "Email A — Creative Dept", "Email B — Artist Management"])

with tab_pdf:
    if uploaded_pdf:
        st.markdown(
            f'<div class="success-box">✓ PDF loaded: <strong>{uploaded_pdf.name}</strong> '
            f'({uploaded_pdf.size:,} bytes)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="warn-box">⚠ No PDF uploaded. Upload via the sidebar → '
            f'Run <code style="color:{GOLD}">python create_sample_pdf.py</code> to generate a test file.</div>',
            unsafe_allow_html=True,
        )

with tab_email_a:
    email_a_text = st.text_area(
        "Email A content (editable)",
        value=st.session_state.get("email_a_edited", EMAIL_A),
        height=280,
        key="email_a_input",
        label_visibility="collapsed",
    )
    st.session_state["email_a_edited"] = email_a_text

with tab_email_b:
    email_b_text = st.text_area(
        "Email B content (editable)",
        value=st.session_state.get("email_b_edited", EMAIL_B),
        height=280,
        key="email_b_input",
        label_visibility="collapsed",
    )
    st.session_state["email_b_edited"] = email_b_text

st.markdown("<br>", unsafe_allow_html=True)

# ── Run button ─────────────────────────────────────────────────────────────────
col_run, col_info = st.columns([1, 3])
with col_run:
    run_clicked = st.button("▶  Run Pipeline", use_container_width=True)
with col_info:
    if not api_key:
        st.markdown(
            f'<div class="warn-box" style="margin:4px 0">Set your API key in the sidebar to run.</div>',
            unsafe_allow_html=True,
        )
    elif not uploaded_pdf:
        st.markdown(
            f'<div class="warn-box" style="margin:4px 0">Upload a PDF via the sidebar, or run '
            f'<code style="color:{GOLD}">python create_sample_pdf.py</code> first.</div>',
            unsafe_allow_html=True,
        )

# ── Run pipeline ───────────────────────────────────────────────────────────────
if run_clicked:
    if not api_key:
        st.error("Please enter an API key in the sidebar.")
        st.stop()
    if not uploaded_pdf:
        st.error("Please upload a PDF agreement in the sidebar.")
        st.stop()
    if not model:
        st.error("Please enter a model name in the sidebar.")
        st.stop()

    # Build client
    try:
        client = get_client(provider, api_key, base_url)
    except Exception as e:
        st.error(f"Failed to initialise LLM client: {e}")
        st.stop()

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    def on_progress(step, total, message):
        progress_bar.progress(step / total)
        status_text.markdown(
            f'<div class="info-box">⟳ <strong>Step {step}/{total}</strong> — {message}</div>',
            unsafe_allow_html=True,
        )

    # Reset PDF position
    uploaded_pdf.seek(0)

    with st.spinner(""):
        try:
            result = run_pipeline(
                pdf_file=uploaded_pdf,
                email_a_text=email_a_text,
                email_b_text=email_b_text,
                client=client,
                model=model,
                progress_callback=on_progress,
            )
            st.session_state["pipeline_result"] = result
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

    progress_bar.progress(1.0)
    status_text.empty()

# ── Results ────────────────────────────────────────────────────────────────────
if "pipeline_result" in st.session_state:
    result = st.session_state["pipeline_result"]
    clean_records = result.get("clean_records", [])
    validation_log = result.get("validation_log", [])
    errors = result.get("errors", [])

    # ── Error summary ──────────────────────────────────────────────────────────
    if errors:
        for err in errors:
            st.markdown(f'<div class="error-box">⚠ {err}</div>', unsafe_allow_html=True)

    # ── Stats strip ───────────────────────────────────────────────────────────
    n_pdf = len(result.get("pdf_songs", []))
    n_ea  = len(result.get("email_a_songs", []))
    n_eb  = len(result.get("email_b_songs", []))
    n_out = len(clean_records)
    n_issues = sum(1 for v in validation_log if v.get("issues"))

    st.markdown(f"""
    <div class="stats-strip">
      <div class="stat-item">
        <div class="stat-num">{n_pdf}</div>
        <div class="stat-label">PDF (table)</div>
      </div>
      <div class="stat-item">
        <div class="stat-num">{n_ea}</div>
        <div class="stat-label">Email A (LLM)</div>
      </div>
      <div class="stat-item">
        <div class="stat-num">{n_eb}</div>
        <div class="stat-label">Email B (LLM)</div>
      </div>
      <div class="stat-item">
        <div class="stat-num" style="color:{'#82c596' if n_issues==0 else GOLD}">{n_out}</div>
        <div class="stat-label">Output Rows</div>
      </div>
      <div class="stat-item">
        <div class="stat-num" style="color:{'#82c596' if n_issues==0 else '#f5c28a'}">{n_issues}</div>
        <div class="stat-label">Flags</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Raw extractions (optional) ─────────────────────────────────────────────
    if show_raw_extraction:
        st.markdown(f'<div class="wc-section">PDF Raw Text (Schedule A page)</div>', unsafe_allow_html=True)
        with st.expander("View extracted PDF text (input to LLM)", expanded=False):
            pdf_raw = result.get("pdf_raw_text", "")
            st.markdown(
                f'<div class="code-block">{pdf_raw}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(f'<div class="wc-section">LLM Parsed Output — PDF + Emails</div>', unsafe_allow_html=True)
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown(f'<span class="source-pdf">PDF (LLM parsed)</span>', unsafe_allow_html=True)
            with st.expander(f"{n_pdf} songs", expanded=False):
                st.markdown(
                    f'<div class="code-block">{json.dumps(result.get("pdf_songs",[]), indent=2)}</div>',
                    unsafe_allow_html=True,
                )
        with rc2:
            st.markdown(f'<span class="source-email">Email A</span>', unsafe_allow_html=True)
            with st.expander(f"{n_ea} songs", expanded=False):
                st.markdown(
                    f'<div class="code-block">{json.dumps(result.get("email_a_songs",[]), indent=2)}</div>',
                    unsafe_allow_html=True,
                )
        with rc3:
            st.markdown(f'<span class="source-email">Email B</span>', unsafe_allow_html=True)
            with st.expander(f"{n_eb} songs", expanded=False):
                st.markdown(
                    f'<div class="code-block">{json.dumps(result.get("email_b_songs",[]), indent=2)}</div>',
                    unsafe_allow_html=True,
                )

    # ── Conflict + Fill log ────────────────────────────────────────────────────
    conflict_log = result.get("conflict_log", [])
    fill_log = result.get("fill_log", [])

    if conflict_log:
        st.markdown(f'<div class="wc-section">Conflict Resolution Log — PDF Won</div>', unsafe_allow_html=True)
        for line in conflict_log:
            st.markdown(
                f'<div style="font-family:DM Mono,monospace;font-size:11px;color:{ORANGE};'
                f'padding:4px 0;border-bottom:1px solid {BLACK3};line-height:1.5">{line.strip()}</div>',
                unsafe_allow_html=True,
            )

    if fill_log and show_merge_debug:
        st.markdown(f'<div class="wc-section">Gap-Fill Log — Email Data Used</div>', unsafe_allow_html=True)
        for line in fill_log:
            color = GREY2 if "MISSING" not in line else RED
            st.markdown(
                f'<div style="font-family:DM Mono,monospace;font-size:11px;color:{color};'
                f'padding:4px 0;border-bottom:1px solid {BLACK3};line-height:1.5">{line.strip()}</div>',
                unsafe_allow_html=True,
            )

    # ── Main output table ──────────────────────────────────────────────────────
    st.markdown(f'<div class="wc-section">Clean Output — Schedule A Unified</div>', unsafe_allow_html=True)

    if clean_records:
        display_cols = ["Song Title", "Writers", "Recording Artist", "ISRC", "Release Date", "_source_note"]
        df = pd.DataFrame(clean_records)[display_cols].rename(
            columns={"_source_note": "Source"}
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Download
        csv_data = build_csv(clean_records)
        st.markdown("<br>", unsafe_allow_html=True)
        col_dl, col_copy = st.columns([1, 3])
        with col_dl:
            st.download_button(
                label="⬇  Download CSV",
                data=csv_data,
                file_name="schedule_a_clean.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_copy:
            st.markdown(
                f'<div class="info-box" style="margin:0;padding:10px 16px">'
                f'Output file: <code style="color:{GOLD}">schedule_a_clean.csv</code> · '
                f'{len(clean_records)} songs · '
                f'All ISRCs normalized · All dates in YYYY-MM-DD</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="error-box">No clean records to display.</div>', unsafe_allow_html=True)

    # ── Validation log ─────────────────────────────────────────────────────────
    if show_validation_log and validation_log:
        st.markdown(f'<div class="wc-section">Validation Log</div>', unsafe_allow_html=True)
        for entry in validation_log:
            issues = entry.get("issues", [])
            badge_html = (
                '<span class="badge-ok">CLEAN</span>' if not issues
                else '<span class="badge-warn">FLAGGED</span>'
            )
            issue_text = (
                " · ".join(issues) if issues
                else "All fields valid"
            )
            source_display = entry.get("source", "unknown")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:12px;padding:6px 0;'
                f'border-bottom:1px solid {BLACK3};font-size:12px">'
                f'<span style="font-family:DM Mono,monospace;font-size:10px;color:{GREY3};width:24px">'
                f'{entry["row"]:02d}</span>'
                f'{badge_html}'
                f'<span style="color:{GREY1};flex:1">{entry["song_title"]}</span>'
                f'<span style="color:{GREY3};font-size:11px;font-family:DM Mono,monospace">{issue_text}</span>'
                f'<span style="color:{GREY3};font-size:10px;font-family:DM Mono,monospace;min-width:180px;text-align:right">{source_display}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── ISRC + Date reference ──────────────────────────────────────────────────
    with st.expander("ISRC & Date Normalization Reference", expanded=False):
        st.markdown(
            f'<div class="info-box">'
            f'<strong style="color:{GOLD}">ISRC Format:</strong> CC-RRR-YY-NNNNN<br>'
            f'&nbsp;&nbsp;CC = 2-letter country code &nbsp;·&nbsp; RRR = 3-char registrant &nbsp;·&nbsp; '
            f'YY = 2-digit year &nbsp;·&nbsp; NNNNN = 5-digit designation<br>'
            f'<strong>Example:</strong> USWB1240432 → US-WB1-24-00432<br><br>'
            f'<strong style="color:{GOLD}">Date Format:</strong> YYYY-MM-DD (ISO 8601)<br>'
            f'<strong>Examples:</strong> "Nov 15, 2024" → 2024-11-15 &nbsp;·&nbsp; '
            f'"15/11/2024" → 2024-11-15 &nbsp;·&nbsp; "December 12, 2024" → 2024-12-12'
            f'</div>',
            unsafe_allow_html=True,
        )

else:
    # ── Empty state ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="empty-state">
      <div class="empty-mark">♪</div>
      <div class="empty-title">Ready to Clean</div>
      <div class="empty-sub">
        Upload the Schedule A PDF via the sidebar,<br>
        review the email inputs, configure your LLM,<br>
        then click <strong style="color:{GOLD}">▶ Run Pipeline</strong> to begin.<br><br>
        No PDF yet? Generate a test file:<br>
        <code style="color:{GOLD}">python create_sample_pdf.py</code>
      </div>
    </div>
    """, unsafe_allow_html=True)
