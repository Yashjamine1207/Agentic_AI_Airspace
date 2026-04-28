"""
core/__init__.py
Shared constants, CSS injection, and Plotly layout helpers
for the Agentic Airspace Streamlit dashboard.
"""

import plotly.graph_objects as go

# ── Colour Palette ──────────────────────────────────────────────────────────
C = {
    "bg":     "#0D1117",
    "card":   "#161B22",
    "border": "#30363D",
    "blue":   "#2196F3",
    "cyan":   "#00BCD4",
    "green":  "#4CAF50",
    "red":    "#F44336",
    "amber":  "#FF9800",
    "purple": "#9C27B0",
    "text":   "#E6EDF3",
    "muted":  "#8B949E",
    "white":  "#FFFFFF",
}

# ── TRACON constants ─────────────────────────────────────────────────────────
TRACON = {
    "lat_min": 36.5, "lat_max": 38.5,
    "lon_min": -123.0, "lon_max": -121.0,
    "alt_min": 0, "alt_max": 18000,
}
WAYPOINTS = {
    "SFO": (37.619, -122.375, 0),
    "OAK": (37.721, -122.221, 0),
    "SJC": (37.363, -121.929, 0),
    "DUMBA": (37.85,  -122.50, 8000),
    "BRIXX": (37.45,  -122.10, 10000),
    "VPJAZ": (37.70,  -121.80, 12000),
}

# ── Actual simulation results from notebook run ──────────────────────────────
REAL_RESULTS = {
    "astar_fuel_kg":      109.0,
    "ppo_fuel_kg":        28.8,
    "fuel_reduction_pct": 73.6,
    "fuel_saved_kg":      80.3,
    "co2_saved_t":        0.254,
    "collision_rate":     0.0,
    "notam_avoidance_pct": 82.0,
    "astar_steps_avg":    32,
    "ppo_steps_avg":      11,
    # Surrogate / RAG (from PDF — these don't change between runs)
    "surrogate_mape":     3.4,
    "surrogate_latency_ms": 28,
    "rag_accuracy_pct":   98.4,
    "rag_latency_ms":     185,
    "rl_reroute_ms":      42,
    "total_latency_ms":   227,
    "speedup_vs_cfd":     400,
}

# ── CSS Injection ────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
/* ── Base ── */
.stApp { background-color: #0D1117; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #30363D;
}
[data-testid="stSidebar"] * { color: #E6EDF3 !important; }

/* ── KPI Card ── */
.kpi-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: border-color .2s;
}
.kpi-card:hover { border-color: #2196F3; }
.kpi-value  { font-size: 2.2rem; font-weight: 700; line-height: 1.1; }
.kpi-label  { font-size: 0.78rem; color: #8B949E; margin-top: .25rem; letter-spacing:.05em; text-transform:uppercase; }
.kpi-delta  { font-size: 0.78rem; margin-top: .3rem; }

/* ── Section header ── */
.section-header {
    border-left: 4px solid #2196F3;
    padding-left: .75rem;
    margin: 1.5rem 0 .8rem 0;
    font-size: 1.1rem; font-weight: 600; color: #E6EDF3;
}

/* ── Status badges ── */
.badge-pass   { background:#1a3a1a; color:#4CAF50; border:1px solid #4CAF50; border-radius:4px; padding:2px 10px; font-size:.8rem; font-weight:600; }
.badge-warn   { background:#3a2a0a; color:#FF9800; border:1px solid #FF9800; border-radius:4px; padding:2px 10px; font-size:.8rem; font-weight:600; }
.badge-fail   { background:#3a0a0a; color:#F44336; border:1px solid #F44336; border-radius:4px; padding:2px 10px; font-size:.8rem; font-weight:600; }
.badge-info   { background:#0a1a3a; color:#2196F3; border:1px solid #2196F3; border-radius:4px; padding:2px 10px; font-size:.8rem; font-weight:600; }

/* ── JSON display ── */
.json-box {
    background:#161B22; border:1px solid #30363D; border-radius:8px;
    padding:1rem; font-family:monospace; font-size:.82rem; color:#4CAF50;
    white-space:pre; overflow-x:auto;
}

/* ── Info box ── */
.info-box {
    background:#0a1a3a; border:1px solid #2196F3; border-radius:8px;
    padding:.8rem 1rem; color:#E6EDF3; font-size:.88rem;
}

/* ── Metric row ── */
.metric-row {
    display:flex; gap:1rem; flex-wrap:wrap; margin:.8rem 0;
}
.metric-chip {
    background:#161B22; border:1px solid #30363D; border-radius:6px;
    padding:.4rem .9rem; font-size:.82rem; color:#8B949E;
}
.metric-chip b { color:#E6EDF3; }

/* ── Divider ── */
.fancy-divider {
    height:1px; background:linear-gradient(90deg,#2196F3,transparent);
    margin:1.5rem 0; border:none;
}

/* ── Latency bar ── */
.latency-bar { margin:.5rem 0; }
.latency-label { font-size:.8rem; color:#8B949E; margin-bottom:.2rem; }
.latency-fill  { height:8px; border-radius:4px; }

/* ── Hide Streamlit default elements ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
"""


def inject_css():
    """Call at top of every page to apply global CSS."""
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def plotly_dark_layout(**kwargs):
    """
    Return a consistent dark Plotly layout dict.
    NOTE: xaxis/yaxis are intentionally excluded here — pass them
    as separate fig.update_layout(xaxis=..., yaxis=...) calls to
    avoid duplicate-keyword errors in Plotly 6.
    """
    base = dict(
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["card"],
        font=dict(color=C["text"], family="Inter, sans-serif", size=12),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(bgcolor=C["card"], bordercolor=C["border"], borderwidth=1),
    )
    base.update(kwargs)
    return base


def axis_style(**kwargs):
    """Return axis dict with dark grid defaults merged with caller overrides."""
    base = dict(gridcolor=C["border"], zerolinecolor=C["border"])
    base.update(kwargs)
    return base


def kpi_card(value: str, label: str, delta: str = "", color: str = "#2196F3") -> str:
    """Return HTML for a KPI card."""
    delta_html = f'<div class="kpi-delta" style="color:{color}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>"""


def section_header(text: str) -> str:
    return f'<div class="section-header">{text}</div>'
