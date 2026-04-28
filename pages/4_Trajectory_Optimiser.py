"""pages/4_Trajectory_Optimiser.py — Plotly 6 compatible"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import inject_css, C, section_header, plotly_dark_layout, axis_style, kpi_card, REAL_RESULTS, WAYPOINTS
from core.astar import astar_3d, path_to_latlon, latlon_to_grid, constraint_to_forbidden

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="✈️ Trajectory Optimiser | Agentic Airspace",
                   page_icon="✈️", layout="wide")
inject_css()

with st.sidebar:
    st.markdown("## ✈️ Trajectory Optimiser")
    st.markdown("---")
    st.markdown('<div style="font-size:.82rem;color:#8B949E">'
                '<b>A* Baseline</b><br>Manhattan heuristic, 3D 26-connected<br>'
                'Fuel-agnostic (geometry only)<br><br>'
                '<b>PPO Agent</b><br>Actor-Critic, 3×256, tanh<br>'
                'Composite reward: fuel+safety+NOTAM</div>', unsafe_allow_html=True)

st.markdown('<h1 style="font-size:2rem;font-weight:800;">✈️ Trajectory Optimiser</h1>'
            '<p style="color:#8B949E">A* Static Baseline vs PPO Agentic Agent — Fuel Efficiency Head-to-Head</p>',
            unsafe_allow_html=True)
st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# Load fuel data
try:
    ppo_fuels   = np.load(os.path.join(BASE_DIR,"ppo_fuels.npy"))
    astar_fuels = np.load(os.path.join(BASE_DIR,"astar_fuels.npy"))
    data_loaded = True
except Exception:
    np.random.seed(42)
    astar_fuels = np.random.normal(REAL_RESULTS["astar_fuel_kg"],12,50)
    ppo_fuels   = np.random.normal(REAL_RESULTS["ppo_fuel_kg"],5,50)
    data_loaded = False

avg_astar  = float(np.mean(astar_fuels))
avg_ppo    = float(np.mean(ppo_fuels))
reduction  = (avg_astar-avg_ppo)/avg_astar*100
saved_kg   = avg_astar-avg_ppo
co2_saved  = saved_kg*3.16/1000

st.markdown(section_header("🎯 Head-to-Head Results — 50 Evaluation Episodes"), unsafe_allow_html=True)
h1,h2,h3,h4,h5 = st.columns(5)
for col,(val,lbl,delta,clr) in zip([h1,h2,h3,h4,h5],[
    (f"{avg_astar:.1f} kg","A* Avg Fuel","Static routing",C["red"]),
    (f"{avg_ppo:.1f} kg","PPO Avg Fuel","Agentic routing",C["green"]),
    (f"{reduction:.1f}%","Fuel Reduction","✅ Target >10%",C["green"]),
    (f"{saved_kg:.1f} kg","Fuel Saved/Flight","Per episode",C["cyan"]),
    (f"{co2_saved:.3f} t","CO₂ Saved/Flight","Direct Scope 1",C["green"]),
]):
    with col:
        st.markdown(kpi_card(val,lbl,delta,clr), unsafe_allow_html=True)

if not data_loaded:
    st.caption("⚠️ Using representative values — run notebook to generate .npy files.")

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# Distribution charts
dist_col, box_col = st.columns([3,2])
with dist_col:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=astar_fuels, nbinsx=15, name="A* Baseline",
                                    marker_color=C["red"], opacity=0.75))
    fig_hist.add_trace(go.Histogram(x=ppo_fuels, nbinsx=15, name="PPO Agent",
                                    marker_color=C["green"], opacity=0.75))
    fig_hist.add_vline(x=avg_astar, line=dict(color=C["red"],dash="dash",width=2),
                       annotation_text=f"A* {avg_astar:.1f}kg",
                       annotation_font_color=C["red"])
    fig_hist.add_vline(x=avg_ppo, line=dict(color=C["green"],dash="dash",width=2),
                       annotation_text=f"PPO {avg_ppo:.1f}kg",
                       annotation_font_color=C["green"])
    fig_hist.update_layout(**plotly_dark_layout(
        title="Fuel Burn Distribution — 50 Episodes (kg)", height=320, barmode="overlay"))
    fig_hist.update_layout(
        xaxis=axis_style(title="Fuel (kg)"),
        yaxis=axis_style(title="Episode Count"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with box_col:
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=astar_fuels, name="A* Baseline", marker_color=C["red"],
                             boxmean=True, fillcolor="rgba(244,67,54,0.2)",
                             line=dict(color=C["red"])))
    fig_box.add_trace(go.Box(y=ppo_fuels, name="PPO Agent", marker_color=C["green"],
                             boxmean=True, fillcolor="rgba(76,175,80,0.2)",
                             line=dict(color=C["green"])))
    fig_box.update_layout(**plotly_dark_layout(title="Fuel Boxplot", height=320, showlegend=False))
    fig_box.update_layout(yaxis=axis_style(title="Fuel (kg)"))
    st.plotly_chart(fig_box, use_container_width=True)

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# Reward function
st.markdown(section_header("⚙️ Composite Reward Function"), unsafe_allow_html=True)
st.markdown(f'<div style="background:{C["card"]};border:1px solid {C["border"]};'
            f'border-radius:10px;padding:1.2rem;font-family:monospace;font-size:.95rem;">'
            f'<span style="color:{C["cyan"]}">R</span><span style="color:{C["muted"]}">t</span>'
            f' = <span style="color:{C["red"]}">−(α·DL_fuel)</span>'
            f' − <span style="color:{C["amber"]}">−(β·P_collision)</span>'
            f' − <span style="color:{C["purple"]}">−(γ·LLM_penalty)</span>'
            f' + <span style="color:{C["green"]}">dest_bonus</span></div>',
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
rw1,rw2,rw3,rw4 = st.columns(4)
for col,(term,weight,desc,clr) in zip([rw1,rw2,rw3,rw4],[
    ("α·DL_fuel","α = 1.0","Primary fuel objective<br>Proportional to fuel cost",C["red"]),
    ("β·P_collision","β = 1,000","Terminal collision penalty<br>Binary <5NM/<1000ft",C["amber"]),
    ("γ·LLM_penalty","γ = 500","Dynamic NOTAM penalty<br>Proximity-scaled",C["purple"]),
    ("Dest Bonus","+500","Terminal reward<br>for reaching waypoint",C["green"]),
]):
    with col:
        st.markdown(f'<div class="kpi-card">'
                    f'<div style="color:{clr};font-weight:700;font-size:.9rem">{term}</div>'
                    f'<div style="font-size:1.1rem;font-weight:800;color:{C["text"]}">{weight}</div>'
                    f'<div style="color:{C["muted"]};font-size:.78rem;margin-top:.4rem;'
                    f'line-height:1.6">{desc}</div></div>', unsafe_allow_html=True)

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# A* Visualiser
st.markdown(section_header("🗺️ A* Path Visualiser"), unsafe_allow_html=True)
wp_names = list(WAYPOINTS.keys())
sc1,sc2 = st.columns(2)
with sc1:
    start_wp = st.selectbox("Start Waypoint:", wp_names, index=0)
with sc2:
    end_wp   = st.selectbox("End Waypoint:", wp_names, index=2)

if st.button("🔍 Run A* Path", type="secondary"):
    with st.spinner("Running A* on 100×100×20 grid…"):
        s_lat,s_lon,s_alt = WAYPOINTS[start_wp]; s_alt = max(s_alt,8000)
        e_lat,e_lon,e_alt = WAYPOINTS[end_wp];   e_alt = max(e_alt,8000)
        path = astar_3d(latlon_to_grid(s_lat,s_lon,s_alt),
                        latlon_to_grid(e_lat,e_lon,e_alt))
    if path:
        coords = path_to_latlon(path)
        fig_as = go.Figure()
        fig_as.add_trace(go.Scatter(x=[c[1] for c in coords], y=[c[0] for c in coords],
            mode="lines+markers", line=dict(color=C["blue"],width=2.5),
            marker=dict(size=3), name=f"A* Path ({len(path)} steps)"))
        for wp,(wlat,wlon,_) in WAYPOINTS.items():
            fig_as.add_trace(go.Scatter(x=[wlon],y=[wlat], mode="markers+text",
                marker=dict(size=10,color=C["cyan"]),
                text=[wp], textposition="top center",
                textfont=dict(color=C["cyan"],size=9), showlegend=False))
        fig_as.update_layout(**plotly_dark_layout(
            title=f"A* Path: {start_wp} → {end_wp} | {len(path)} steps", height=380))
        fig_as.update_layout(
            xaxis=axis_style(title="Longitude", range=[-123.1,-121.7]),
            yaxis=axis_style(title="Latitude",  range=[37.0,38.3]),
        )
        st.plotly_chart(fig_as, use_container_width=True)
    else:
        st.warning("No path found.")
else:
    fig_wp = go.Figure()
    for wp,(wlat,wlon,_) in WAYPOINTS.items():
        fig_wp.add_trace(go.Scatter(x=[wlon],y=[wlat], mode="markers+text",
            marker=dict(size=12,color=C["cyan"]), text=[wp],
            textposition="top center", textfont=dict(color=C["cyan"],size=10),
            showlegend=False))
    fig_wp.add_shape(type="rect", x0=-123.0,x1=-121.0,y0=36.5,y1=38.5,
                     line=dict(color=C["border"],dash="dot",width=1.5))
    fig_wp.update_layout(**plotly_dark_layout(title="SFO/OAK/SJC TRACON — Key Waypoints",height=380))
    fig_wp.update_layout(
        xaxis=axis_style(title="Longitude", range=[-123.2,-121.5]),
        yaxis=axis_style(title="Latitude",  range=[36.3,38.6]),
    )
    st.plotly_chart(fig_wp, use_container_width=True)
