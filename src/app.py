"""
app.py — Streamlit UI for Review-of-Reviews
Run from project root: streamlit run src/app.py
"""

import os
import sys
import streamlit as st
import pandas as pd

# Allow imports from src/ regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

from utils import load_reviews
from logic_engine import ReviewLogic
from trust_score import TrustAnalyzer
from csp_solver import CSPSolver
from summarizer import annotate, astar_search

# -----------------------------------------------------------------------
# Page config — must be first Streamlit call
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Review-of-Reviews",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------
# Custom CSS — grey / white / blue palette, clean academic feel
# -----------------------------------------------------------------------
st.markdown("""
<style>
  /* ---- Google Font ---- */
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

  /* ---- Base ---- */
  html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
  }

  /* ---- Background ---- */
  .stApp {
      background-color: #f4f6f9;
  }

  /* ---- Sidebar ---- */
  [data-testid="stSidebar"] {
      background-color: #1e2d45;
  }
  [data-testid="stSidebar"] * {
      color: #d6e4f7 !important;
  }
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stNumberInput label {
      color: #a8c4e0 !important;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
  }

  /* ---- Header banner ---- */
  .ror-header {
      background: linear-gradient(135deg, #1e2d45 0%, #2b4a72 100%);
      border-radius: 12px;
      padding: 28px 36px;
      margin-bottom: 24px;
      color: white;
  }
  .ror-header h1 {
      font-size: 1.9rem;
      font-weight: 600;
      margin: 0 0 4px 0;
      color: white;
  }
  .ror-header p {
      font-size: 0.92rem;
      color: #a8c4e0;
      margin: 0;
  }

  /* ---- Stat cards ---- */
  .stat-row {
      display: flex;
      gap: 14px;
      margin-bottom: 20px;
  }
  .stat-card {
      flex: 1;
      background: white;
      border-radius: 10px;
      padding: 18px 20px;
      border-left: 4px solid #2b6cb0;
      box-shadow: 0 1px 4px rgba(0,0,0,0.07);
  }
  .stat-card .stat-value {
      font-size: 1.6rem;
      font-weight: 600;
      color: #1e2d45;
      line-height: 1;
  }
  .stat-card .stat-label {
      font-size: 0.75rem;
      color: #718096;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-top: 4px;
  }

  /* ---- Section headers ---- */
  .section-title {
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: #4a6fa5;
      margin: 20px 0 10px 0;
      padding-bottom: 6px;
      border-bottom: 1px solid #dee4ef;
  }

  /* ---- CSP report box ---- */
  .csp-box {
      background: white;
      border-radius: 10px;
      border: 1px solid #dee4ef;
      padding: 18px 22px;
      font-family: 'DM Mono', monospace;
      font-size: 0.78rem;
      color: #2d3748;
      line-height: 1.7;
      box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }

  /* ---- Review cards ---- */
  .review-card {
      background: white;
      border-radius: 10px;
      border: 1px solid #dee4ef;
      padding: 20px 22px;
      margin-bottom: 14px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.05);
      transition: box-shadow 0.2s;
  }
  .review-card:hover {
      box-shadow: 0 3px 12px rgba(0,0,0,0.10);
  }
  .review-number {
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #4a6fa5;
      margin-bottom: 8px;
  }
  .review-text {
      font-size: 0.90rem;
      color: #2d3748;
      line-height: 1.55;
      margin-bottom: 14px;
      font-style: italic;
  }
  .review-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
  }

  /* ---- Pill badges ---- */
  .pill {
      display: inline-block;
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 0.72rem;
      font-weight: 500;
  }
  .pill-topic    { background: #ebf4ff; color: #2b6cb0; }
  .pill-pos      { background: #e6fffa; color: #276749; }
  .pill-neg      { background: #fff5f5; color: #c53030; }
  .pill-neu      { background: #f7fafc; color: #718096; border: 1px solid #e2e8f0; }
  .pill-flag     { background: #fffbeb; color: #975a16; }

  /* ---- Trust bar ---- */
  .trust-bar-wrap {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 6px;
  }
  .trust-bar-track {
      flex: 1;
      height: 6px;
      background: #edf2f7;
      border-radius: 3px;
      overflow: hidden;
  }
  .trust-bar-fill {
      height: 100%;
      border-radius: 3px;
  }
  .trust-bar-label {
      font-size: 0.75rem;
      font-weight: 600;
      color: #2d3748;
      white-space: nowrap;
  }

  /* ---- Log expander ---- */
  .log-line {
      font-family: 'DM Mono', monospace;
      font-size: 0.75rem;
      color: #4a5568;
      padding: 2px 0;
      border-bottom: 1px solid #f7fafc;
  }

  /* ---- No-result message ---- */
  .no-result {
      background: #fff5f5;
      border: 1px solid #fed7d7;
      border-radius: 10px;
      padding: 20px;
      color: #c53030;
      font-size: 0.88rem;
  }

  /* ---- Run button override ---- */
  div.stButton > button {
      background: #2b6cb0;
      color: white;
      border: none;
      border-radius: 8px;
      padding: 10px 28px;
      font-size: 0.88rem;
      font-weight: 500;
      width: 100%;
      transition: background 0.2s;
  }
  div.stButton > button:hover {
      background: #2c5282;
  }

  /* Hide Streamlit default chrome */
  #MainMenu, footer { visibility: hidden; }
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def trust_color(trust: float) -> str:
    if trust >= 0.80:
        return "#38a169"   # green
    elif trust >= 0.60:
        return "#d69e2e"   # amber
    else:
        return "#e53e3e"   # red


def sentiment_pill(label: str) -> str:
    cls = {"Positive": "pill-pos", "Negative": "pill-neg"}.get(label, "pill-neu")
    return f'<span class="pill {cls}">{label}</span>'


def topic_pill(topic: str) -> str:
    return f'<span class="pill pill-topic">{topic}</span>'


def flag_pill(flag: str) -> str:
    short = flag[:45] + "…" if len(flag) > 45 else flag
    return f'<span class="pill pill-flag">⚠ {short}</span>'


def render_review_card(rank: int, review: dict):
    trust = review["trust"]
    bar_color = trust_color(trust)
    bar_pct = int(trust * 100)

    topics_html = " ".join(topic_pill(t) for t in review["topics"])
    sentiments_html = " ".join(
        f'<span style="font-size:0.72rem;color:#718096;">{t}:</span> {sentiment_pill(s)}'
        for t, s in review["sentiment_map"].items()
    )
    flags_html = " ".join(flag_pill(f) for f in review.get("trust_reasons", []))

    preview = str(review["text"])[:220].replace('"', '&quot;')

    st.markdown(f"""
    <div class="review-card">
      <div class="review-number">Review #{rank}</div>
      <div class="review-text">"{preview}{"…" if len(str(review["text"])) > 220 else ""}"</div>

      <div class="review-meta">
        {topics_html}
      </div>

      <div style="margin-top:10px; display:flex; flex-wrap:wrap; gap:6px; align-items:center;">
        {sentiments_html}
      </div>

      <div class="trust-bar-wrap" style="margin-top:12px;">
        <span style="font-size:0.72rem;color:#718096;white-space:nowrap;">Trust score</span>
        <div class="trust-bar-track">
          <div class="trust-bar-fill"
               style="width:{bar_pct}%; background:{bar_color};"></div>
        </div>
        <span class="trust-bar-label">{trust:.2f}</span>
      </div>

      {"<div style='margin-top:10px;'>" + flags_html + "</div>" if flags_html else ""}
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------
# Data loading (cached so it doesn't re-run on every widget interaction)
# -----------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_and_annotate(data_path: str, sample_n: int, text_col: str):
    df = load_reviews(data_path, text_col=text_col, sample_n=sample_n)
    df = annotate(df, text_col=text_col)
    return df


# -----------------------------------------------------------------------
# Sidebar — controls
# -----------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    st.markdown("**Dataset**")
    sample_n = st.slider("Reviews to sample", 20, 200, 60, step=10,
                         help="How many reviews to load from the dataset before filtering.")

    st.markdown("---")
    st.markdown("**CSP Constraints**")

    min_trust = st.slider("Minimum trust score", 0.0, 1.0, 0.55, step=0.05,
                          help="Reviews below this trust threshold are excluded by the CSP solver.")

    min_words = st.slider("Minimum word count", 1, 30, 8, step=1,
                          help="Reviews shorter than this are excluded.")

    topic_options = ["Any", "Logistics", "Quality", "Fit", "General"]
    required_topic = st.selectbox("Required topic", topic_options,
                                  help="Force at least one selected review to cover this topic.")

    sentiment_options = ["Any", "Positive", "Negative", "Neutral"]
    sentiment_filter = st.selectbox("Sentiment filter", sentiment_options,
                                    help="Only consider reviews that contain this sentiment.")

    st.markdown("---")
    st.markdown("**A* Search**")

    k = st.slider("Reviews to select (k)", 1, 10, 5, step=1,
                  help="Number of reviews the A* optimizer will choose.")

    topic_weight = st.slider("Topic coverage weight", 0.0, 1.0, 0.7, step=0.05,
                             help="How much the A* objective values covering diverse topics.")
    trust_weight = round(1.0 - topic_weight, 2)
    st.caption(f"Trust weight (auto): **{trust_weight}**")

    st.markdown("---")
    run = st.button("▶  Run Analysis")

# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------

st.markdown("""
<div class="ror-header">
  <h1>🔍 Review-of-Reviews</h1>
  <p>Logic-based topic inference · CSP constraint solving · A* search optimization</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# Locate dataset — look in ./data/ relative to project root
# -----------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data",
                         "Womens Clothing E-Commerce Reviews.csv")
TEXT_COL = "Review Text"

if not os.path.exists(DATA_PATH):
    st.error(
        f"Dataset not found at `{DATA_PATH}`.  \n"
        "Place `Womens Clothing E-Commerce Reviews.csv` in the `/data` folder "
        "and restart the app."
    )
    st.stop()

# -----------------------------------------------------------------------
# Main logic — runs when button is pressed (or on first load with defaults)
# -----------------------------------------------------------------------

if run or "last_results" not in st.session_state:

    with st.spinner("Loading and annotating reviews…"):
        df = load_and_annotate(DATA_PATH, sample_n, TEXT_COL)

    # Build candidate list
    candidates = [
        {
            "text": row[TEXT_COL],
            "topics": row["topics"],
            "sentiment_map": row["sentiment_map"],
            "trust": row["trust"],
            "trust_reasons": row["trust_reasons"],
        }
        for _, row in df.iterrows()
    ]

    # Resolve "Any" → None for the solver
    req_topic = None if required_topic == "Any" else required_topic
    sent_filter = None if sentiment_filter == "Any" else sentiment_filter

    constraints = {
        "min_trust": min_trust,
        "required_topic": req_topic,
        "sentiment_filter": sent_filter,
        "min_words": min_words,
        "coverage_goal": True,
    }

    # CSP solve
    with st.spinner("Running CSP solver…"):
        solver = CSPSolver(candidates, k, constraints)
        domain_size = len(solver.initial_domain)
        solution, stats, forward_log = solver.solve()

    csp_report_lines = [
        f"Candidates after unary filtering : {domain_size} / {len(candidates)}",
        f"Slots to fill (k)                : {k}",
        f"Nodes expanded                   : {stats.get('nodes_expanded', 0)}",
        f"Backtracks                       : {stats.get('backtracks', 0)}",
        f"Forward prunes                   : {stats.get('forward_prunes', 0)}",
        f"Result                           : "
        f"{'Valid assignment found ✓' if solution else 'No solution found ✗'}",
    ]

    # A* optimize
    if solution:
        with st.spinner("Running A* search…"):
            selected = astar_search(
                solution,
                k=k,
                topic_weight=topic_weight,
                trust_weight=trust_weight,
                beam_width=120,
                max_expansions=4000,
            )
    else:
        selected = []

    # Store results in session state so they persist across reruns
    st.session_state["last_results"] = {
        "selected": selected,
        "csp_report_lines": csp_report_lines,
        "forward_log": forward_log,
        "stats": stats,
        "total_loaded": len(candidates),
        "domain_size": domain_size,
        "constraints": constraints,
    }

# -----------------------------------------------------------------------
# Render results (always, from session state)
# -----------------------------------------------------------------------

res = st.session_state.get("last_results", {})
selected = res.get("selected", [])
csp_report_lines = res.get("csp_report_lines", [])
forward_log = res.get("forward_log", [])
stats = res.get("stats", {})
total_loaded = res.get("total_loaded", 0)
domain_size = res.get("domain_size", 0)

# Stat cards
all_topics_covered = set()
for r in selected:
    all_topics_covered.update(r["topics"])

avg_trust = (sum(r["trust"] for r in selected) / len(selected)) if selected else 0.0

st.markdown(f"""
<div class="stat-row">
  <div class="stat-card">
    <div class="stat-value">{total_loaded}</div>
    <div class="stat-label">Reviews loaded</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{domain_size}</div>
    <div class="stat-label">Passed CSP filter</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{len(selected)}</div>
    <div class="stat-label">Selected by A*</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{len(all_topics_covered)}</div>
    <div class="stat-label">Topics covered</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{avg_trust:.2f}</div>
    <div class="stat-label">Avg trust score</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Two-column layout: CSP report left, reviews right
col_left, col_right = st.columns([1, 1.6])

with col_left:
    st.markdown('<div class="section-title">CSP Solver Report</div>', unsafe_allow_html=True)
    report_text = "\n".join(csp_report_lines)
    st.markdown(f'<div class="csp-box">{report_text}</div>', unsafe_allow_html=True)

    if forward_log:
        with st.expander(f"Forward checking log ({len(forward_log)} entries)"):
            for line in forward_log[:30]:
                st.markdown(f'<div class="log-line">{line}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Topic coverage</div>', unsafe_allow_html=True)
    if all_topics_covered:
        pills = " ".join(topic_pill(t) for t in sorted(all_topics_covered))
        st.markdown(pills, unsafe_allow_html=True)
    else:
        st.caption("No topics covered.")

with col_right:
    st.markdown('<div class="section-title">Selected Reviews</div>', unsafe_allow_html=True)

    if not selected:
        st.markdown("""
        <div class="no-result">
          <strong>No reviews selected.</strong><br>
          The CSP solver found no valid assignment with the current constraints.
          Try lowering the minimum trust score or removing the required topic filter.
        </div>
        """, unsafe_allow_html=True)
    else:
        for i, review in enumerate(selected, 1):
            render_review_card(i, review)