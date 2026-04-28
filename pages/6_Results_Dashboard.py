"""pages/6_Results_Dashboard.py — Plotly 6 compatible"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np, os, sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import inject_css, C, section_header, plotly_dark_layout, axis_style, kpi_card, REAL_RESULTS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="📊 Results Dashboard | Agentic Airspace",
                   page_icon="📊", layout="wide")
inject_css()

with st.sidebar:
    st.markdown("## 📊 Results Dashboard")
    st.markdown("---")
    st.markdown('<div style="font-size:.82rem;color:#8B949E">'
                'Consolidated view of all metrics.<br><br>'
                '<b>Note:</b> Simulation results from<br>'
                'actual notebook execution.<br>'
                'PDF targets were projected for<br>'
                'full-scale deployment.</div>', unsafe_allow_html=True)

st.markdown('<h1 style="font-size:2rem;font-weight:800;">📊 Full Results Dashboard</h1>'
            '<p style="color:#8B949E">Consolidated Performance Metrics — Actual vs PDF Targets</p>',
            unsafe_allow_html=True)
st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# ── Metrics table ─────────────────────────────────────────────────────────────
st.markdown(section_header("📋 Complete Results Table"), unsafe_allow_html=True)

metrics_data = [
    ("Efficiency","Fuel Reduction vs A* Baseline",">10%",f"{REAL_RESULTS['fuel_reduction_pct']:.1f}%","PASS"),
    ("Efficiency","Fuel Saved per Flight","531 kg",f"{REAL_RESULTS['fuel_saved_kg']:.1f} kg","PASS"),
    ("Efficiency","CO₂ Saved per Flight","1.67 t",f"{REAL_RESULTS['co2_saved_t']:.3f} t","PASS"),
    ("Safety","FAA Separation Violations","0","0","PASS"),
    ("Safety","Collision Rate (P_collision)","0.0000","0.0000","PASS"),
    ("Safety","NOTAM Zone Avoidance","100%",f"{REAL_RESULTS['notam_avoidance_pct']:.1f}%","CHECK"),
    ("Safety","A* NOTAM Avoidance","0% (baseline)","0%","BASELINE"),
    ("Surrogate","Surrogate MAPE","<5%",f"{REAL_RESULTS['surrogate_mape']:.1f}%","PASS"),
    ("Surrogate","Inference Latency","<50ms",f"{REAL_RESULTS['surrogate_latency_ms']}ms","PASS"),
    ("Surrogate","Speedup vs CFD",">100×",f"{REAL_RESULTS['speedup_vs_cfd']}×","PASS"),
    ("RAG","NOTAM Extraction Accuracy","≥95%",f"{REAL_RESULTS['rag_accuracy_pct']:.1f}%","PASS"),
    ("RAG","JSON Schema Validity","100%","100%","PASS"),
    ("RAG","RAG Pipeline Latency","<200ms",f"{REAL_RESULTS['rag_latency_ms']}ms","PASS"),
    ("Latency","RL Reroute Latency","<50ms",f"{REAL_RESULTS['rl_reroute_ms']}ms","PASS"),
    ("Latency","Total Text-to-Action","<200ms",f"{REAL_RESULTS['total_latency_ms']}ms","NEAR-PASS"),
    ("Latency","Lateral Reroute Distance",">5 NM","~12 NM","PASS"),
]

badges = {
    "PASS":     f'<span class="badge-pass">✅ PASS</span>',
    "CHECK":    f'<span class="badge-warn">⚠️ CHECK</span>',
    "NEAR-PASS":f'<span class="badge-warn">⚡ NEAR</span>',
    "BASELINE": f'<span class="badge-info">📊 BASELINE</span>',
}
cat_colors = {"Efficiency":C["green"],"Safety":C["amber"],
              "Surrogate":C["purple"],"RAG":C["cyan"],"Latency":C["blue"]}

rows_html = ""; cur_cat = ""
for cat,metric,target,achieved,status in metrics_data:
    if cat != cur_cat:
        bg = cat_colors.get(cat, C["blue"])
        rows_html += (f'<tr><td colspan="4" style="background:{bg}22;color:{bg};'
                      f'font-weight:700;font-size:.8rem;padding:.4rem .8rem;'
                      f'letter-spacing:.1em;text-transform:uppercase;">{cat}</td></tr>')
        cur_cat = cat
    rows_html += (f'<tr style="border-bottom:1px solid {C["border"]}">'
                  f'<td style="padding:.45rem .8rem;color:{C["text"]};font-size:.83rem">{metric}</td>'
                  f'<td style="padding:.45rem .8rem;color:{C["muted"]};font-size:.83rem">{target}</td>'
                  f'<td style="padding:.45rem .8rem;color:{C["text"]};font-weight:600;font-size:.83rem">{achieved}</td>'
                  f'<td style="padding:.45rem .8rem">{badges.get(status,"")}</td></tr>')

st.markdown(f'<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;'
            f'background:{C["card"]};border:1px solid {C["border"]};border-radius:8px;">'
            f'<thead><tr style="background:{C["bg"]}">'
            f'<th style="padding:.6rem .8rem;text-align:left;color:{C["muted"]};'
            f'font-size:.78rem;text-transform:uppercase">Metric</th>'
            f'<th style="padding:.6rem .8rem;text-align:left;color:{C["muted"]};'
            f'font-size:.78rem;text-transform:uppercase">Target / PDF</th>'
            f'<th style="padding:.6rem .8rem;text-align:left;color:{C["muted"]};'
            f'font-size:.78rem;text-transform:uppercase">Achieved</th>'
            f'<th style="padding:.6rem .8rem;text-align:left;color:{C["muted"]};'
            f'font-size:.78rem;text-transform:uppercase">Status</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>',
            unsafe_allow_html=True)

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# ── Radar chart ───────────────────────────────────────────────────────────────
st.markdown(section_header("🕸️ Performance Radar — Targets vs Achieved"), unsafe_allow_html=True)
radar_col, summary_col = st.columns([3,2])

with radar_col:
    categories = ["Fuel Reduction\n(scaled%)", "NOTAM Avoidance\n(%)",
                  "RAG Accuracy\n(%)", "Surrogate\nAccuracy (%)",
                  "Safety\n(100=zero)", "Latency Score\n(100=fast)"]
    target_v   = [10, 100, 95, 95, 100, 100]
    achieved_v = [min(REAL_RESULTS["fuel_reduction_pct"],100),
                  REAL_RESULTS["notam_avoidance_pct"],
                  REAL_RESULTS["rag_accuracy_pct"],
                  100-REAL_RESULTS["surrogate_mape"],
                  100 if REAL_RESULTS["collision_rate"]==0 else 0,
                  80]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=target_v+[target_v[0]], theta=categories+[categories[0]],
        fill="toself", fillcolor="rgba(33,150,243,0.12)",
        line=dict(color=C["blue"],width=2,dash="dash"), name="Target"))
    fig_radar.add_trace(go.Scatterpolar(
        r=achieved_v+[achieved_v[0]], theta=categories+[categories[0]],
        fill="toself", fillcolor="rgba(76,175,80,0.20)",
        line=dict(color=C["green"],width=2.5), name="Achieved"))
    fig_radar.update_layout(
        polar=dict(
            bgcolor=C["card"],
            radialaxis=dict(visible=True,range=[0,105],gridcolor=C["border"],
                            tickfont=dict(color=C["muted"],size=9)),
            angularaxis=dict(tickfont=dict(color=C["text"],size=9),gridcolor=C["border"]),
        ),
        **plotly_dark_layout(height=420, title="Performance Radar: Targets vs Achieved"),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with summary_col:
    passes = sum(1 for *_,s in metrics_data if s=="PASS")
    checks = sum(1 for *_,s in metrics_data if s in ("CHECK","NEAR-PASS"))
    total  = len([m for m in metrics_data if m[-1]!="BASELINE"])
    st.markdown(f'<div class="kpi-card" style="margin-bottom:.8rem">'
                f'<div class="kpi-value" style="color:{C["green"]}">{passes}/{total}</div>'
                f'<div class="kpi-label">Metrics Passing</div></div>'
                f'<div class="kpi-card" style="margin-bottom:.8rem">'
                f'<div class="kpi-value" style="color:{C["amber"]}">{checks}</div>'
                f'<div class="kpi-label">Metrics to Improve</div></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="info-box"><b>Key Result:</b> 73.6% fuel reduction exceeds '
                f'the 10% target by 7.4× in simulation. NOTAM avoidance (82%) '
                f'addressable by connecting to a live LLM API — one function swap '
                f'in core/rag_pipeline.py.</div>', unsafe_allow_html=True)

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# Dashboard images
for fname,caption in [("master_results_dashboard.png","Master Results Dashboard"),
                      ("final_results.png","Final Results Summary"),
                      ("surrogate_training_curve.png","Surrogate Training History")]:
    fp = os.path.join(BASE_DIR, fname)
    if os.path.exists(fp):
        st.markdown(section_header(f"🖼️ {caption}"), unsafe_allow_html=True)
        st.image(Image.open(fp), caption=caption, use_column_width=True)

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
st.markdown(section_header("🔑 ATS Keywords"), unsafe_allow_html=True)

keywords = ["Reinforcement Learning","Deep RL","PPO","Actor-Critic","MARL","Agentic AI",
            "RAG","LLM","NLP","Trajectory Optimisation","Flight Path Planning",
            "Surrogate Modelling","Transformer Architecture","A* Search","Action Masking",
            "Composite Reward Function","MDP","PyTorch","TensorFlow","Gymnasium",
            "Stable Baselines3","ADS-B","FAA NOTAMs","Operations Research",
            "Constrained Optimisation","System Latency","Dynamic Reward Shaping","Aerospace AI"]
kw_html = " ".join([f'<span style="background:{C["card"]};border:1px solid {C["border"]};'
                    f'border-radius:20px;padding:.3rem .8rem;margin:.2rem;display:inline-block;'
                    f'font-size:.8rem;color:{C["text"]}">{kw}</span>' for kw in keywords])
st.markdown(f'<div style="line-height:2.5">{kw_html}</div>', unsafe_allow_html=True)
