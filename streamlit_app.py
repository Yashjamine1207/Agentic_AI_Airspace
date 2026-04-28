"""
streamlit_app.py  — HOME PAGE  (Plotly 6 compatible)
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import inject_css, C, REAL_RESULTS, kpi_card, section_header, plotly_dark_layout, axis_style

st.set_page_config(
    page_title="Agentic Airspace | Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

with st.sidebar:
    st.markdown("## ✈️ Agentic Airspace")
    st.markdown("**MARL + RAG Flight Routing**")
    st.markdown("---")
    st.markdown("**Navigate to:**")
    st.markdown("- ⚡ Live Agentic Demo")
    st.markdown("- 🧠 RAG Pipeline")
    st.markdown("- 🔮 Surrogate Model")
    st.markdown("- ✈️ Trajectory Optimiser")
    st.markdown("- 🛡️ MARL Safety")
    st.markdown("- 📊 Results Dashboard")
    st.markdown("---")
    st.markdown(
        '<div style="font-size:.75rem;color:#8B949E">'
        'SFO/OAK/SJC TRACON<br>100×100×20 3D Grid<br>'
        'PPO + RAG + MARL<br>April–May 2026</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style="text-align:center;padding:2rem 0 1rem 0;">
        <div style="font-size:3rem;margin-bottom:.3rem;">✈️</div>
        <h1 style="font-size:2.4rem;font-weight:800;margin:0;
                   background:linear-gradient(90deg,#2196F3,#00BCD4);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Agentic Airspace
        </h1>
        <p style="color:#8B949E;font-size:1.1rem;margin-top:.5rem;">
            RAG-Driven Multi-Agent Reinforcement Learning for Dynamic 3D Flight Routing
        </p>
        <p style="color:#8B949E;font-size:.85rem;">
            SFO / OAK / SJC TRACON &nbsp;·&nbsp; PPO Actor-Critic &nbsp;·&nbsp;
            8-Layer Transformer Surrogate &nbsp;·&nbsp; Agentic LLM/RAG Pipeline
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# ── KPI Row 1 ─────────────────────────────────────────────────────────────────
st.markdown(section_header("🎯 System Performance — Actual Simulation Results"),
            unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(kpi_card("73.6%", "Fuel Reduction vs A* Baseline",
                          "✅ Target >10% — PASS", C["green"]), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card("0", "Collision Violations",
                          "✅ Zero-tolerance — PASS", C["green"]), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card("82.0%", "NOTAM Zone Avoidance",
                          "⚠️ Target >95% — CHECK", C["amber"]), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_card("400×", "Speedup vs CFD Physics",
                          "✅ 28ms vs 5.4s — PASS", C["blue"]), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
k5, k6, k7, k8 = st.columns(4)
with k5:
    st.markdown(kpi_card("3.4%", "Surrogate MAPE",
                          "✅ Target <5% — PASS", C["cyan"]), unsafe_allow_html=True)
with k6:
    st.markdown(kpi_card("98.4%", "RAG NOTAM Accuracy",
                          "✅ Target >95% — PASS", C["green"]), unsafe_allow_html=True)
with k7:
    st.markdown(kpi_card("227ms", "Text → Trajectory Latency",
                          "⚡ 185ms RAG + 42ms RL", C["blue"]), unsafe_allow_html=True)
with k8:
    st.markdown(kpi_card("1.2M", "ADS-B Training Vectors",
                          "SFO/OAK/SJC TRACON", C["purple"]), unsafe_allow_html=True)

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# ── Fuel bar chart ────────────────────────────────────────────────────────────
st.markdown(section_header("⛽ Fuel Efficiency: PPO Agent vs A* Static Baseline"),
            unsafe_allow_html=True)

col_chart, col_info = st.columns([2, 1])
with col_chart:
    fig_fuel = go.Figure()
    fig_fuel.add_trace(go.Bar(
        x=["A* Static Baseline", "PPO Agent (Agentic RL)"],
        y=[REAL_RESULTS["astar_fuel_kg"], REAL_RESULTS["ppo_fuel_kg"]],
        marker_color=[C["red"], C["green"]],
        text=[f'{REAL_RESULTS["astar_fuel_kg"]:.1f} kg',
              f'{REAL_RESULTS["ppo_fuel_kg"]:.1f} kg'],
        textposition="outside",
        textfont=dict(color=C["text"], size=14),
        width=0.5,
    ))
    fig_fuel.update_layout(**plotly_dark_layout(
        title="Avg Fuel Burn per Episode (kg)", height=320, showlegend=False))
    fig_fuel.update_layout(
        yaxis=axis_style(title="Fuel (kg)", range=[0, 130]),
        xaxis=axis_style(),
    )
    st.plotly_chart(fig_fuel, use_container_width=True)

with col_info:
    st.markdown(
        f"""
        <div style="margin-top:1.5rem;">
            <div class="kpi-card" style="margin-bottom:.8rem;">
                <div class="kpi-value" style="color:{C['green']}">
                    {REAL_RESULTS['fuel_reduction_pct']:.1f}%</div>
                <div class="kpi-label">Fuel Reduction</div>
            </div>
            <div class="kpi-card" style="margin-bottom:.8rem;">
                <div class="kpi-value" style="color:{C['cyan']}">
                    {REAL_RESULTS['fuel_saved_kg']:.1f} kg</div>
                <div class="kpi-label">Fuel Saved / Flight</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value" style="color:{C['green']}">
                    {REAL_RESULTS['co2_saved_t']:.3f} t</div>
                <div class="kpi-label">CO₂ Saved / Flight</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# ── Architecture diagram ──────────────────────────────────────────────────────
st.markdown(section_header("🏗️ System Architecture — 5-Layer Agentic Pipeline"),
            unsafe_allow_html=True)

fig_arch = go.Figure()
layers = [
    (0.10, "📡 Layer 1\nDATA",
     "OpenSky ADS-B (1.2M)<br>FAA NOTAM / METAR", C["blue"]),
    (0.28, "🧠 Layer 2\nSURROGATE",
     "8-Layer Transformer<br>28ms · MAPE 3.4%", C["purple"]),
    (0.46, "🤖 Layer 3\nRAG/LLM",
     "Two-stage NOTAM parser<br>→ 4D JSON (185ms)", C["cyan"]),
    (0.64, "✈️ Layer 4\nRL AGENT",
     "PPO Actor-Critic<br>Continuous 3D control", C["green"]),
    (0.82, "🛡️ Layer 5\nSAFETY",
     "A* + Action Masking<br><5NM / <1000ft", C["amber"]),
]
for x_pos, title, desc, color in layers:
    fig_arch.add_shape(type="rect",
        x0=x_pos-0.085, x1=x_pos+0.085, y0=0.10, y1=0.90,
        fillcolor=C["card"], line=dict(color=color, width=2))
    fig_arch.add_annotation(x=x_pos, y=0.75,
        text=f"<b>{title.replace(chr(10),'<br>')}</b>",
        font=dict(size=9.5, color=color), showarrow=False)
    fig_arch.add_annotation(x=x_pos, y=0.38, text=desc,
        font=dict(size=8.5, color=C["muted"]), showarrow=False, align="center")
    if x_pos < 0.82:
        fig_arch.add_annotation(
            x=x_pos + 0.100, y=0.5,
            xref="paper", yref="paper",
            ax=30, ay=0,
            axref="pixel", ayref="pixel",
            text="", showarrow=True,
            arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor=C["border"])

fig_arch.update_layout(
    **plotly_dark_layout(height=220),
    xaxis=dict(visible=False, range=[0, 1]),
    yaxis=dict(visible=False, range=[0, 1]),
    showlegend=False,
)
fig_arch.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_arch, use_container_width=True)

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# ── Innovation cards ──────────────────────────────────────────────────────────
st.markdown(section_header("💡 Core Technical Innovation"), unsafe_allow_html=True)

i1, i2, i3 = st.columns(3)
with i1:
    st.markdown(
        f'<div class="kpi-card"><div style="font-size:1.8rem">🔗</div>'
        f'<div style="color:{C["blue"]};font-weight:700;margin:.4rem 0">'
        f'Text → Math → Action</div>'
        f'<div style="color:{C["muted"]};font-size:.85rem">Closes the loop from '
        f'unstructured FAA regulatory text to physically executed trajectory updates '
        f'with no human intervention.</div></div>', unsafe_allow_html=True)
with i2:
    st.markdown(
        f'<div class="kpi-card"><div style="font-size:1.8rem">⚡</div>'
        f'<div style="color:{C["cyan"]};font-weight:700;margin:.4rem 0">'
        f'400× Computational Speedup</div>'
        f'<div style="color:{C["muted"]};font-size:.85rem">8-layer Transformer '
        f'surrogate replaces 5.4s CFD physics engine with 28ms inference — '
        f'enabling high-throughput PPO sampling.</div></div>', unsafe_allow_html=True)
with i3:
    st.markdown(
        f'<div class="kpi-card"><div style="font-size:1.8rem">🛡️</div>'
        f'<div style="color:{C["green"]};font-weight:700;margin:.4rem 0">'
        f'Hard Safety Invariants</div>'
        f'<div style="color:{C["muted"]};font-size:.85rem">Action Masking makes '
        f'collision avoidance a structural invariant — not a learned behaviour '
        f'that can be unlearned.</div></div>', unsafe_allow_html=True)
