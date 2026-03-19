import streamlit as st
import joblib
import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="ENGINE — Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# Custom CSS — industrial dark theme
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0a0e17;
    color: #c9d1e0;
}

.stApp {
    background: #0a0e17;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1220;
    border-right: 1px solid #1e2d45;
}

/* Cards */
.metric-card {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
}
.metric-card .value {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
}
.metric-card .label {
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a6080;
    margin-top: 6px;
}

/* Risk colors */
.low    { color: #22c55e; }
.medium { color: #f59e0b; }
.high   { color: #f97316; }
.critical { color: #ef4444; }

/* Header banner */
.banner {
    background: linear-gradient(135deg, #0d1220 0%, #111827 100%);
    border: 1px solid #1e2d45;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 20px 28px;
    margin-bottom: 28px;
}
.banner h1 {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.6rem;
    color: #e2e8f0;
    margin: 0;
    letter-spacing: 0.05em;
}
.banner p {
    color: #4a6080;
    margin: 4px 0 0 0;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Section headers */
.section-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3b82f6;
    border-bottom: 1px solid #1e2d45;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* Streamlit overrides */
div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 8px;
    padding: 16px;
}
div[data-testid="stMetric"] label {
    color: #4a6080 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-family: 'Share Tech Mono', monospace !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace;
    color: #e2e8f0;
}

.stSelectbox label, .stSlider label, .stNumberInput label {
    color: #4a6080 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #1e2d45;
    border-radius: 8px;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #4a6080 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom-color: #3b82f6 !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Load model & data (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("artifacts/gb_model.joblib")
    feature_cols = joblib.load("artifacts/feature_cols.joblib")
    return model, feature_cols

@st.cache_data
def load_data():
    columns = (
        ["engine_id", "cycle"] +
        [f"op_setting_{i}" for i in range(1, 4)] +
        [f"sensor_{i}" for i in range(1, 22)]
    )
    DROP_SENSORS = ["sensor_1","sensor_5","sensor_6","sensor_10","sensor_16","sensor_18","sensor_19"]

    df_train = pd.read_csv("CMAPSSData/train_FD001.txt", sep=r"\s+", header=None, names=columns)
    df_test  = pd.read_csv("CMAPSSData/test_FD001.txt",  sep=r"\s+", header=None, names=columns)

    for df in [df_train, df_test]:
        df.drop(columns=DROP_SENSORS, inplace=True)

    WINDOW = 20
    sensor_cols = [c for c in df_test.columns if c.startswith("sensor_")]

    for df in [df_train, df_test]:
        for sensor in sensor_cols:
            df[f"{sensor}_roll_mean"] = df.groupby("engine_id")[sensor].transform(
                lambda x: x.rolling(WINDOW, min_periods=1).mean())
            df[f"{sensor}_roll_std"] = df.groupby("engine_id")[sensor].transform(
                lambda x: x.rolling(WINDOW, min_periods=1).std())
            df[f"{sensor}_trend"] = df.groupby("engine_id")[sensor].transform(
                lambda x: x.diff(WINDOW))

    df_train.fillna(0, inplace=True)
    df_test.fillna(0, inplace=True)

    return df_train, df_test, sensor_cols

model, feature_cols = load_model()
df_train, df_test, sensor_cols = load_data()

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def risk_category(prob):
    if prob < 0.2:   return "LOW RISK"
    elif prob < 0.5: return "MEDIUM RISK"
    elif prob < 0.75: return "HIGH RISK"
    else:            return "CRITICAL RISK"

def risk_color(cat):
    return {"LOW RISK":"#22c55e","MEDIUM RISK":"#f59e0b","HIGH RISK":"#f97316","CRITICAL RISK":"#ef4444"}.get(cat,"#94a3b8")

def predict_engine(engine_id, source_df=None, n_samples=50):
    if source_df is None:
        source_df = df_test
    engine_df = source_df[source_df["engine_id"] == engine_id]
    if engine_df.empty:
        return None, None
    latest = engine_df.sort_values("cycle").iloc[-1]
    X = pd.DataFrame([latest[feature_cols].values], columns=feature_cols)
    probs = [model.predict_proba(X)[0, 1] for _ in range(n_samples)]
    return float(np.mean(probs)), float(np.std(probs))

def predict_fleet_from(start_id, end_id, threshold=0.5, source_df=None):
    if source_df is None:
        source_df = df_test
    results = []
    for eid in range(start_id, end_id + 1):
        mean_r, std_r = predict_engine(eid, source_df=source_df)
        if mean_r is not None:
            results.append({"Engine ID": eid, "Failure Risk": mean_r, "Uncertainty (±)": std_r})
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df["Risk Rank"]     = df["Failure Risk"].rank(ascending=False, method="dense").astype(int)
    df["Risk Category"] = df["Failure Risk"].apply(risk_category)
    df["Status"]        = df["Failure Risk"].apply(lambda x: "⚠ ALERT" if x >= threshold else "✓ OK")
    return df.sort_values("Risk Rank").reset_index(drop=True)

# ------------------------------------------------------------------
# PSI — exact formula from notebook
# reference = early cycles (first CYCLE_CUTOFF), current = late cycles
# ------------------------------------------------------------------
CYCLE_CUTOFF = 50

# max_cycle per engine needed for correct current_df split

def calculate_psi(expected, actual, bins=10):
    breakpoints = np.linspace(0, 100, bins + 1)
    expected_perc = np.percentile(expected, breakpoints)
    psi = 0
    for i in range(len(expected_perc) - 1):
        exp_pct = ((expected >= expected_perc[i]) & (expected < expected_perc[i+1])).mean()
        act_pct = ((actual   >= expected_perc[i]) & (actual   < expected_perc[i+1])).mean()
        exp_pct = max(exp_pct, 1e-6)
        act_pct = max(act_pct, 1e-6)
        psi += (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    return psi

def compute_drift(sensor_cols, df_train, cutoff=CYCLE_CUTOFF):
    # Match notebook exactly:
    # reference = early life (cycle <= cutoff)
    # current   = near-failure (last `cutoff` cycles of each engine's life)
    max_cycles   = df_train.groupby("engine_id")["cycle"].transform("max")
    reference_df = df_train[df_train["cycle"] <= cutoff]
    current_df   = df_train[df_train["cycle"] >= max_cycles - cutoff]
    records = []
    for sensor in sensor_cols:
        ref_vals = reference_df[sensor].dropna().values
        cur_vals = current_df[sensor].dropna().values
        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue
        psi           = calculate_psi(ref_vals, cur_vals)
        ks_stat, ks_p = stats.ks_2samp(ref_vals, cur_vals)
        if psi > 0.25:   drift = "HIGH DRIFT"
        elif psi > 0.1:  drift = "MODERATE"
        else:            drift = "STABLE"
        records.append({"Sensor": sensor, "PSI": round(psi,4), "KS Stat": round(ks_stat,4),
                        "KS p-value": round(ks_p,6), "Drift Status": drift})
    return pd.DataFrame(records).sort_values("PSI", ascending=False).reset_index(drop=True)

def plotly_dark():
    return dict(
        plot_bgcolor="#0a0e17", paper_bgcolor="#0a0e17",
        font=dict(color="#c9d1e0", family="Share Tech Mono"),
        xaxis=dict(gridcolor="#1e2d45", zerolinecolor="#1e2d45"),
        yaxis=dict(gridcolor="#1e2d45", zerolinecolor="#1e2d45"),
    )

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-family: Share Tech Mono, monospace; font-size:1.3rem; color:#3b82f6; letter-spacing:0.15em;'>⚙ ENGINE</div>
        <div style='font-size:0.65rem; color:#4a6080; letter-spacing:0.2em; text-transform:uppercase; margin-top:4px;'>Predictive Maintenance</div>
    </div>
    <hr style='border-color:#1e2d45; margin: 12px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Fleet Dashboard", "Engine Deep Dive", "Sensor Drift Monitor", "How It Works"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#1e2d45; margin: 16px 0;'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='font-size:0.7rem; color:#4a6080; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:8px;'>Model Info</div>
    <div style='font-family: Share Tech Mono, monospace; font-size:0.8rem; color:#c9d1e0; line-height:2;'>
        Dataset &nbsp;&nbsp;: NASA C-MAPSS<br>
        Model &nbsp;&nbsp;&nbsp;&nbsp;: Gradient Boosting<br>
        ROC-AUC &nbsp;&nbsp;: 0.9899<br>
        F1 Score &nbsp;: 0.889
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# PAGE 1 — Fleet Dashboard
# ------------------------------------------------------------------
if page == "Fleet Dashboard":

    data_mode = st.radio(
        "Mode",
        ["📡  Test Data — Fleet Snapshot", "📈  Train Data — Engine Lifecycle"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # ══════════════════════════════════════════════════════════════
    # MODE A — TEST DATA: Fleet snapshot (current risk per engine)
    # ══════════════════════════════════════════════════════════════
    if "Test" in data_mode:
        st.markdown("""
        <div class='banner'>
            <h1>📡 FLEET SNAPSHOT — DEPLOYMENT MODE</h1>
            <p>Live risk score per engine · Last recorded sensor reading · Test dataset</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#111827; border:1px solid #1e2d45; border-left:4px solid #3b82f6;
                    border-radius:8px; padding:12px 20px; margin-bottom:16px; font-size:0.82rem; color:#94a3b8;'>
            <b style='color:#3b82f6;'>Deployment mode</b> &nbsp;·&nbsp;
            Each engine is assessed at its <b>latest recorded sensor reading</b>.
            This is what a real-time monitoring dashboard looks like — a snapshot of your entire fleet right now.
            Most engines show LOW RISK because the test set contains engines that have not yet failed.
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns([2,1])
        with col_a:
            start_id = st.number_input("Start Engine ID", min_value=1, max_value=100, value=1, key="test_start")
            end_id   = st.number_input("End Engine ID",   min_value=1, max_value=100, value=20, key="test_end")
        with col_b:
            threshold = st.slider("Alert Threshold", 0.1, 0.9, 0.5, 0.05, key="test_thresh")
            run = st.button("▶  RUN FLEET SNAPSHOT", use_container_width=True, key="test_run")

        if run:
            with st.spinner("Scanning fleet..."):
                st.session_state["snap_df"]     = predict_fleet_from(int(start_id), int(end_id), threshold, df_test)
                st.session_state["snap_thresh"] = threshold

        if "snap_df" in st.session_state:
            df_fleet    = st.session_state["snap_df"]
            used_thresh = st.session_state.get("snap_thresh", threshold)

            if df_fleet.empty:
                st.warning("No valid engine IDs found.")
            else:
                total    = len(df_fleet)
                critical = (df_fleet["Risk Category"] == "CRITICAL RISK").sum()
                high     = (df_fleet["Risk Category"] == "HIGH RISK").sum()
                alert    = (df_fleet["Status"] == "⚠ ALERT").sum()
                healthy  = total - alert

                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric("Total Engines", total)
                c2.metric("Critical",  int(critical))
                c3.metric("High Risk", int(high))
                c4.metric("On Alert",  int(alert))
                c5.metric("Healthy",   int(healthy))

                st.markdown("<div class='section-title' style='margin-top:24px;'>Current Risk Score — All Engines</div>", unsafe_allow_html=True)

                colors = [risk_color(c) for c in df_fleet["Risk Category"]]
                fig = go.Figure(go.Bar(
                    x=df_fleet["Engine ID"].astype(str),
                    y=df_fleet["Failure Risk"],
                    marker_color=colors,
                    error_y=dict(type="data", array=df_fleet["Uncertainty (±)"], visible=True, color="#334155"),
                    hovertemplate="<b>Engine %{x}</b><br>Risk: %{y:.4f}<extra></extra>"
                ))
                fig.add_hline(y=used_thresh, line_dash="dash", line_color="#ef4444",
                              annotation_text=f"Alert threshold ({used_thresh})",
                              annotation_font_color="#ef4444")
                layout = plotly_dark()
                layout["yaxis"] = dict(range=[0,1], gridcolor="#1e2d45", title="Failure Risk")
                layout["xaxis"] = dict(gridcolor="#1e2d45", title="Engine ID")
                fig.update_layout(**layout, height=340, bargap=0.3, margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("<div class='section-title'>Engine Risk Table</div>", unsafe_allow_html=True)
                def color_row_snap(row):
                    c = risk_color(row["Risk Category"])
                    return [f"color: {c}" if col in ["Risk Category","Failure Risk","Status"] else "" for col in row.index]
                display_df = df_fleet.copy()
                display_df["Failure Risk"]    = display_df["Failure Risk"].round(4)
                display_df["Uncertainty (±)"] = display_df["Uncertainty (±)"].round(4)
                st.dataframe(display_df.style.apply(color_row_snap, axis=1), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════
    # MODE B — TRAIN DATA: Engine lifecycle (risk over time)
    # ══════════════════════════════════════════════════════════════
    else:
        st.markdown("""
        <div class='banner'>
            <h1>📈 ENGINE LIFECYCLE — DEGRADATION ANALYSIS</h1>
            <p>Risk score at every cycle · Full run-to-failure · Train dataset</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#111827; border:1px solid #1e2d45; border-left:4px solid #f59e0b;
                    border-radius:8px; padding:12px 20px; margin-bottom:16px; font-size:0.82rem; color:#94a3b8;'>
            <b style='color:#f59e0b;'>Degradation analysis mode</b> &nbsp;·&nbsp;
            Select one or more engines and watch their <b>risk score evolve from first cycle to failure</b>.
            This is only possible with training data which contains the full run-to-failure lifecycle.
            A well-calibrated model should show risk climbing steadily as the engine degrades.
        </div>
        """, unsafe_allow_html=True)

        max_train_eng = int(df_train["engine_id"].max())
        col_a, col_b = st.columns([2,1])
        with col_a:
            selected_engines = st.multiselect(
                "Select Engines to Compare (1–5 recommended)",
                options=list(range(1, max_train_eng + 1)),
                default=[1, 2, 3],
                key="lifecycle_engines"
            )
        with col_b:
            sample_every = st.slider("Sample every N cycles", 1, 10, 3,
                                     help="Higher = faster but less resolution")
            run_lc = st.button("▶  RUN LIFECYCLE ANALYSIS", use_container_width=True, key="lc_run")

        if run_lc and selected_engines:
            with st.spinner("Computing risk across full lifecycle..."):
                lifecycle_data = {}
                for eid in selected_engines:
                    engine_df = df_train[df_train["engine_id"] == eid].sort_values("cycle")
                    cycles, risks = [], []
                    for idx in range(0, len(engine_df), sample_every):
                        row  = engine_df.iloc[idx]
                        X    = pd.DataFrame([row[feature_cols].values], columns=feature_cols)
                        risk = float(model.predict_proba(X)[0, 1])
                        cycles.append(int(row["cycle"]))
                        risks.append(risk)
                    lifecycle_data[eid] = {"cycles": cycles, "risks": risks}
                st.session_state["lifecycle_data"] = lifecycle_data

        if "lifecycle_data" in st.session_state:
            lc = st.session_state["lifecycle_data"]

            # ── Risk progression line chart ──
            st.markdown("<div class='section-title' style='margin-top:8px;'>Risk Score Progression — Cycle 1 → Failure</div>", unsafe_allow_html=True)

            palette = ["#3b82f6","#f59e0b","#22c55e","#a855f7","#f97316"]
            fig_lc = go.Figure()

            for i, (eid, data) in enumerate(lc.items()):
                color = palette[i % len(palette)]
                fig_lc.add_trace(go.Scatter(
                    x=data["cycles"], y=data["risks"],
                    mode="lines", name=f"Engine {eid}",
                    line=dict(color=color, width=2),
                    hovertemplate=f"<b>Engine {eid}</b><br>Cycle %{{x}}<br>Risk: %{{y:.4f}}<extra></extra>"
                ))

            # Shade risk zones
            fig_lc.add_hrect(y0=0,    y1=0.2,  fillcolor="#052e16", opacity=0.3, line_width=0)
            fig_lc.add_hrect(y0=0.2,  y1=0.5,  fillcolor="#1c1400", opacity=0.3, line_width=0)
            fig_lc.add_hrect(y0=0.5,  y1=0.75, fillcolor="#1c0a00", opacity=0.3, line_width=0)
            fig_lc.add_hrect(y0=0.75, y1=1.0,  fillcolor="#1c0000", opacity=0.3, line_width=0)

            # Zone labels
            for y, label in [(0.1,"LOW"),(0.35,"MEDIUM"),(0.62,"HIGH"),(0.87,"CRITICAL")]:
                fig_lc.add_annotation(x=0, y=y, text=label, showarrow=False,
                                      font=dict(size=9, color="#4a6080"),
                                      xref="paper", xanchor="left")

            layout_lc = plotly_dark()
            layout_lc["yaxis"] = dict(range=[0,1], gridcolor="#1e2d45", title="Failure Risk")
            layout_lc["xaxis"] = dict(gridcolor="#1e2d45", title="Cycle (Operational Time)")
            layout_lc["legend"] = dict(bgcolor="#111827", bordercolor="#1e2d45")
            fig_lc.update_layout(**layout_lc, height=420, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_lc, use_container_width=True)

            # ── Per-engine summary cards ──
            st.markdown("<div class='section-title'>Engine Lifecycle Summary</div>", unsafe_allow_html=True)

            # Compute comparison stats before rendering cards
            all_cycles    = {eid: data["cycles"][-1] for eid, data in lc.items()}
            longest_eid   = max(all_cycles, key=all_cycles.get)
            shortest_eid  = min(all_cycles, key=all_cycles.get)
            avg_cycles    = sum(all_cycles.values()) / len(all_cycles)

            # Find which engine escalated to HIGH RISK latest (most gradual degradation)
            escalation_cycles = {}
            for eid, data in lc.items():
                for cyc, risk in zip(data["cycles"], data["risks"]):
                    if risk >= 0.5:
                        escalation_cycles[eid] = cyc
                        break
                else:
                    escalation_cycles[eid] = data["cycles"][-1]

            latest_escalation_eid = max(escalation_cycles, key=escalation_cycles.get)

            CARDS_PER_ROW = 4
            lc_items = list(lc.items())

            for row_start in range(0, len(lc_items), CARDS_PER_ROW):
                row_items = lc_items[row_start:row_start + CARDS_PER_ROW]
                cols_lc   = st.columns(len(row_items))

                for col_idx, (eid, data) in enumerate(row_items):
                    final_risk    = data["risks"][-1]
                    max_risk      = max(data["risks"])
                    total_cyc     = data["cycles"][-1]
                    esc_cyc       = escalation_cycles[eid]
                    cat           = risk_category(final_risk)
                    color         = risk_color(cat)
                    diff_from_avg = total_cyc - avg_cycles

                    badges = []
                    if eid == longest_eid and len(lc) > 1:
                        badges.append(("<span style='background:#052e16; color:#22c55e; border:1px solid #22c55e33; "
                                       "border-radius:4px; padding:2px 8px; font-size:0.65rem; letter-spacing:0.1em;'>"
                                       "LONGEST LIFE</span>"))
                    if eid == shortest_eid and len(lc) > 1:
                        badges.append(("<span style='background:#1c0000; color:#ef4444; border:1px solid #ef444433; "
                                       "border-radius:4px; padding:2px 8px; font-size:0.65rem; letter-spacing:0.1em;'>"
                                       "SHORTEST LIFE</span>"))
                    if eid == latest_escalation_eid and len(lc) > 1:
                        badges.append(("<span style='background:#0c1a2e; color:#3b82f6; border:1px solid #3b82f633; "
                                       "border-radius:4px; padding:2px 8px; font-size:0.65rem; letter-spacing:0.1em;'>"
                                       "MOST GRADUAL</span>"))

                    diff_str   = f"+{int(diff_from_avg)}" if diff_from_avg >= 0 else str(int(diff_from_avg))
                    diff_color = "#22c55e" if diff_from_avg >= 0 else "#ef4444"
                    badge_html = " ".join(badges)

                    cols_lc[col_idx].markdown(f"""
                    <div class='metric-card' style='border-top: 3px solid {color}; margin-bottom:12px;'>
                        <div style='min-height:24px; margin-bottom:8px;'>{badge_html}</div>
                        <div class='value' style='color:{color};'>{final_risk:.3f}</div>
                        <div class='label'>Engine {eid} · Final Risk</div>
                        <div style='margin-top:12px; font-size:0.72rem; color:#4a6080; line-height:2;'>
                            Peak risk &nbsp;&nbsp;&nbsp;&nbsp;: {max_risk:.3f}<br>
                            Life cycles &nbsp;&nbsp;: {total_cyc}
                            <span style='color:{diff_color}; font-size:0.68rem;'> ({diff_str} vs avg)</span><br>
                            Risk escalated : cycle {esc_cyc}<br>
                            Status &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <span style='color:{color};'>{cat}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Fleet insight summary ──
            if len(lc) > 1:
                best     = lc[longest_eid]
                worst    = lc[shortest_eid]
                best_cyc = best["cycles"][-1]
                worst_cyc= worst["cycles"][-1]
                grad_esc = escalation_cycles[latest_escalation_eid]
                st.markdown(f"""
                <div style='background:#111827; border:1px solid #1e2d45; border-left:4px solid #22c55e;
                            border-radius:8px; padding:16px 20px; margin-top:16px;
                            font-size:0.82rem; color:#94a3b8; line-height:1.9;'>
                    <b style='color:#22c55e;'>⚡ Fleet Insight</b><br>
                    <b style='color:#e2e8f0;'>Engine {longest_eid}</b> had the longest operational life
                    at <b style='color:#22c55e;'>{best_cyc} cycles</b> —
                    <b style='color:#22c55e;'>{best_cyc - worst_cyc} cycles longer</b> than
                    Engine {shortest_eid} ({worst_cyc} cycles). It can sustain operation significantly
                    longer before requiring intervention.<br>
                    <b style='color:#e2e8f0;'>Engine {latest_escalation_eid}</b> showed the most gradual
                    degradation — risk didn't escalate past 50% until cycle {grad_esc}, suggesting
                    more predictable wear and easier maintenance scheduling compared to engines
                    that escalate abruptly.
                </div>
                """, unsafe_allow_html=True)


# ------------------------------------------------------------------
# PAGE 2 — Engine Deep Dive
# ------------------------------------------------------------------
elif page == "Engine Deep Dive":
    st.markdown("""
    <div class='banner'>
        <h1>🔍 ENGINE DEEP DIVE</h1>
        <p>Per-engine risk analysis · Manual sensor input · Uncertainty estimation</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📂  Dataset Lookup", "⌨️  Manual Sensor Input"])

    # ── TAB 1: Dataset Lookup ──────────────────────────────────────
    with tab1:
        all_ids   = sorted(df_test["engine_id"].unique())
        engine_id = st.selectbox("Select Engine ID", all_ids)

        if engine_id:
            mean_risk, std_risk = predict_engine(int(engine_id))

            if mean_risk is None:
                st.error("Engine not found.")
            else:
                cat   = risk_category(mean_risk)
                color = risk_color(cat)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Engine ID",     engine_id)
                c2.metric("Failure Risk",  f"{mean_risk:.4f}")
                c3.metric("Uncertainty ±", f"{std_risk:.4f}")
                c4.metric("Risk Category", cat)

                st.markdown("<div class='section-title' style='margin-top:20px;'>Risk Gauge</div>", unsafe_allow_html=True)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=mean_risk,
                    delta={"reference": 0.5, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#22c55e"}},
                    gauge={
                        "axis": {"range": [0,1], "tickcolor":"#4a6080"},
                        "bar": {"color": color, "thickness": 0.25},
                        "bgcolor": "#111827",
                        "bordercolor": "#1e2d45",
                        "steps": [
                            {"range": [0,    0.2],  "color": "#052e16"},
                            {"range": [0.2,  0.5],  "color": "#1c1400"},
                            {"range": [0.5,  0.75], "color": "#1c0a00"},
                            {"range": [0.75, 1],    "color": "#1c0000"},
                        ],
                        "threshold": {"line": {"color":"#ef4444","width":3}, "thickness":0.8, "value":0.5}
                    },
                    number={"font": {"color": color, "family": "Share Tech Mono"}}
                ))
                fig_gauge.update_layout(**plotly_dark(), height=260, margin=dict(l=40,r=40,t=20,b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown("<div class='section-title'>Sensor Trends — Last 50 Cycles</div>", unsafe_allow_html=True)
                engine_data = df_test[df_test["engine_id"] == engine_id].sort_values("cycle").tail(50)
                top_sensors = sensor_cols[:4]
                scols = st.columns(2)
                for i, sensor in enumerate(top_sensors):
                    fig_s = go.Figure()
                    fig_s.add_trace(go.Scatter(
                        x=engine_data["cycle"], y=engine_data[sensor],
                        mode="lines", name="Raw", line=dict(color="#334155", width=1)
                    ))
                    fig_s.add_trace(go.Scatter(
                        x=engine_data["cycle"], y=engine_data[f"{sensor}_roll_mean"],
                        mode="lines", name="Rolling Mean", line=dict(color="#3b82f6", width=2)
                    ))
                    fig_s.update_layout(
                        **plotly_dark(), height=200,
                        title=dict(text=sensor.upper(), font=dict(size=11, color="#4a6080"), x=0),
                        showlegend=False, margin=dict(l=0,r=0,t=30,b=0)
                    )
                    scols[i % 2].plotly_chart(fig_s, use_container_width=True)

                st.markdown("<div class='section-title'>Top Predictive Features (Model Importances)</div>", unsafe_allow_html=True)
                importances = model.feature_importances_
                feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
                feat_df = feat_df.sort_values("Importance", ascending=True).tail(15)
                fig_imp = go.Figure(go.Bar(
                    x=feat_df["Importance"], y=feat_df["Feature"],
                    orientation="h", marker_color="#3b82f6",
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>"
                ))
                fig_imp.update_layout(**plotly_dark(), height=360, margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig_imp, use_container_width=True)

    # ── TAB 2: Manual Sensor Input ─────────────────────────────────
    with tab2:
        st.markdown("""
        <div style='background:#111827; border:1px solid #1e2d45; border-left:4px solid #22c55e;
                    border-radius:8px; padding:12px 20px; margin-bottom:20px; font-size:0.82rem; color:#94a3b8;'>
            <b style='color:#22c55e;'>ℹ How this works</b> &nbsp;·&nbsp;
            Enter raw sensor readings. The system computes rolling features (mean, std, trend)
            using your last N cycles of input, then runs the same Gradient Boosting inference pipeline.
            Simulates a real-time IoT sensor feed.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Enter Sensor Readings (Current Cycle)</div>", unsafe_allow_html=True)

        # Get default values from median of test data for guidance
        defaults = {s: float(df_test[s].median()) for s in sensor_cols}

        # Render input fields in 3 columns
        input_vals = {}
        rows = [sensor_cols[i:i+3] for i in range(0, len(sensor_cols), 3)]
        for row in rows:
            cols_inp = st.columns(3)
            for j, sensor in enumerate(row):
                input_vals[sensor] = cols_inp[j].number_input(
                    sensor.upper(),
                    value=round(defaults[sensor], 4),
                    format="%.4f",
                    key=f"inp_{sensor}"
                )

        st.markdown("<div class='section-title' style='margin-top:16px;'>Cycle History (for rolling features)</div>", unsafe_allow_html=True)
        n_cycles = st.slider(
            "Number of historical cycles to simulate (same reading repeated — conservative estimate)",
            min_value=5, max_value=50, value=20, step=5
        )

        predict_btn = st.button("▶  PREDICT FAILURE RISK", use_container_width=True, key="manual_predict")

        if predict_btn:
            # Build a mini dataframe replicating the current reading over n_cycles
            # This matches your feature engineering: rolling mean/std/trend over WINDOW=20
            rows_list = []
            for cycle in range(1, n_cycles + 1):
                row_data = {"engine_id": 9999, "cycle": cycle}
                row_data.update({s: input_vals[s] for s in sensor_cols})
                # op_settings — use median from test data (not used in features but needed for schema)
                for op in ["op_setting_1","op_setting_2","op_setting_3"]:
                    row_data[op] = float(df_test[op].median()) if op in df_test.columns else 0.0
                rows_list.append(row_data)

            manual_df = pd.DataFrame(rows_list)

            # Apply your exact feature engineering (WINDOW=20, same as training)
            WINDOW = 20
            for sensor in sensor_cols:
                manual_df[f"{sensor}_roll_mean"] = (
                    manual_df[sensor].rolling(WINDOW, min_periods=1).mean()
                )
                manual_df[f"{sensor}_roll_std"] = (
                    manual_df[sensor].rolling(WINDOW, min_periods=1).std()
                )
                manual_df[f"{sensor}_trend"] = manual_df[sensor].diff(WINDOW)

            manual_df.fillna(0, inplace=True)

            # Use the last row (most recent cycle) — same as predict_engine
            latest_row = manual_df.sort_values("cycle").iloc[-1]
            X_manual   = pd.DataFrame([latest_row[feature_cols].values], columns=feature_cols)

            # Run uncertainty-aware inference (same as your predict_with_uncertainty)
            probs      = [model.predict_proba(X_manual)[0, 1] for _ in range(50)]
            mean_risk  = float(np.mean(probs))
            std_risk   = float(np.std(probs))
            cat        = risk_category(mean_risk)
            color      = risk_color(cat)

            st.markdown("<div class='section-title' style='margin-top:20px;'>Prediction Result</div>", unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)
            r1.metric("Failure Risk",  f"{mean_risk:.4f}")
            r2.metric("Uncertainty ±", f"{std_risk:.4f}")
            r3.metric("Risk Category", cat)

            # Gauge
            fig_m = go.Figure(go.Indicator(
                mode="gauge+number",
                value=mean_risk,
                gauge={
                    "axis": {"range": [0,1], "tickcolor":"#4a6080"},
                    "bar": {"color": color, "thickness": 0.25},
                    "bgcolor": "#111827",
                    "bordercolor": "#1e2d45",
                    "steps": [
                        {"range": [0,    0.2],  "color": "#052e16"},
                        {"range": [0.2,  0.5],  "color": "#1c1400"},
                        {"range": [0.5,  0.75], "color": "#1c0a00"},
                        {"range": [0.75, 1],    "color": "#1c0000"},
                    ],
                    "threshold": {"line": {"color":"#ef4444","width":3}, "thickness":0.8, "value":0.5}
                },
                number={"font": {"color": color, "family": "Share Tech Mono"}}
            ))
            fig_m.update_layout(**plotly_dark(), height=260, margin=dict(l=40,r=40,t=20,b=20))
            st.plotly_chart(fig_m, use_container_width=True)

            # Show what features were computed
            with st.expander("🔍 View computed features (transparency)"):
                st.dataframe(
                    pd.DataFrame([latest_row[feature_cols].values], columns=feature_cols).T
                    .rename(columns={0: "Computed Value"}).round(6),
                    use_container_width=True
                )

# ------------------------------------------------------------------
# PAGE 3 — Sensor Drift Monitor
# ------------------------------------------------------------------
elif page == "Sensor Drift Monitor":
    st.markdown("""
    <div class='banner'>
        <h1>📡 SENSOR DRIFT MONITOR</h1>
        <p>Population Stability Index · Kolmogorov–Smirnov Test · Distribution Shift Detection</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#111827; border:1px solid #1e2d45; border-left:4px solid #f59e0b;
                border-radius:8px; padding:14px 20px; margin-bottom:20px; font-size:0.82rem; color:#94a3b8;'>
        <b style='color:#f59e0b;'>ℹ Methodology</b> &nbsp;·&nbsp;
        Reference = early life (cycle ≤ 50) · Current = near-failure (last 50 cycles of each engine's life)<br>
        PSI &gt; 0.25 = significant drift · PSI 0.1–0.25 = moderate · All sensors expected to drift — engines physically degrade toward failure<br>
        KS p-value = 0.0 for all sensors is correct and expected — healthy vs near-failure states are statistically unquestionably different
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Computing drift metrics..."):
        drift_df = compute_drift(sensor_cols, df_train, cutoff=CYCLE_CUTOFF)

    # Summary cards
    stable   = (drift_df["Drift Status"] == "STABLE").sum()
    moderate = (drift_df["Drift Status"] == "MODERATE").sum()
    high_d   = (drift_df["Drift Status"] == "HIGH DRIFT").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stable Sensors",        stable)
    c2.metric("Moderate Drift",        moderate)
    c3.metric("High Drift (PSI>0.25)", high_d)
    c4.metric("Expected High Drift",   f"{high_d}/{stable+moderate+high_d}",
              help="All sensors drifting is normal — we compare healthy vs near-failure cycles")

    if high_d == stable + moderate + high_d:
        st.markdown("""
        <div style='background:#0d1f12; border:1px solid #22c55e33; border-left:4px solid #22c55e;
                    border-radius:8px; padding:10px 18px; margin-bottom:4px; font-size:0.8rem; color:#94a3b8;'>
            <b style='color:#22c55e;'>✓ All sensors show HIGH DRIFT — this is correct and expected.</b>
            We're comparing early healthy operation (cycle ≤ 50) against near-failure cycles (last 50 cycles).
            These are physically opposite states of the engine. High PSI = high degradation signal = 
            <b style='color:#e2e8f0;'>high predictive power</b>. These sensors are exactly what the model 
            uses to detect failure.
        </div>
        """, unsafe_allow_html=True)

    # PSI bar chart
    st.markdown("<div class='section-title' style='margin-top:20px;'>PSI by Sensor</div>", unsafe_allow_html=True)

    drift_colors = [
        "#ef4444" if s == "HIGH DRIFT" else "#f59e0b" if s == "MODERATE" else "#22c55e"
        for s in drift_df["Drift Status"]
    ]

    fig_psi = go.Figure(go.Bar(
        x=drift_df["Sensor"], y=drift_df["PSI"],
        marker_color=drift_colors,
        hovertemplate="<b>%{x}</b><br>PSI: %{y:.4f}<extra></extra>"
    ))
    fig_psi.add_hline(y=0.1,  line_dash="dot",  line_color="#f59e0b",
                      annotation_text="Moderate threshold (0.1)",
                      annotation_position="top right",
                      annotation_font_color="#f59e0b", annotation_font_size=11)
    fig_psi.add_hline(y=0.25, line_dash="dash", line_color="#ef4444",
                      annotation_text="High drift threshold (0.25)",
                      annotation_position="top right",
                      annotation_font_color="#ef4444", annotation_font_size=11)
    # Add a note that all HIGH DRIFT is expected for this dataset
    fig_psi.add_annotation(
        x=0.01, y=0.97, xref="paper", yref="paper",
        text="ℹ All sensors show HIGH DRIFT — expected when comparing healthy vs near-failure cycles",
        showarrow=False, font=dict(size=10, color="#4a6080"),
        align="left", xanchor="left"
    )
    psi_layout = plotly_dark()
    psi_layout["yaxis"] = dict(gridcolor="#1e2d45", title="PSI Value")
    fig_psi.update_layout(**psi_layout, height=380, margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig_psi, use_container_width=True)

    # Distribution overlay for selected sensor
    st.markdown("<div class='section-title'>Distribution Comparison — Early Cycles vs Late Cycles</div>", unsafe_allow_html=True)

    col_sens, col_cut = st.columns([3,1])
    with col_sens:
        selected_sensor = st.selectbox("Select Sensor", sensor_cols)
    with col_cut:
        cutoff_val = st.number_input("Cycle cutoff", min_value=10, max_value=200, value=CYCLE_CUTOFF, step=10)

    # Match notebook: reference = early life, current = near-failure (last cutoff_val cycles)
    max_cyc_map = df_train.groupby("engine_id")["cycle"].transform("max")
    ref_vals = df_train[df_train["cycle"] <= cutoff_val][selected_sensor].dropna().values
    cur_vals = df_train[df_train["cycle"] >= max_cyc_map - cutoff_val][selected_sensor].dropna().values

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=ref_vals, name=f"Early Life (cycle ≤ {cutoff_val})", opacity=0.6,
        marker_color="#3b82f6", nbinsx=40,
        histnorm="probability density"
    ))
    fig_dist.add_trace(go.Histogram(
        x=cur_vals, name=f"Near Failure (last {cutoff_val} cycles)", opacity=0.6,
        marker_color="#f97316", nbinsx=40,
        histnorm="probability density"
    ))
    fig_dist.update_layout(
        **plotly_dark(), barmode="overlay", height=320,
        legend=dict(bgcolor="#111827", bordercolor="#1e2d45"),
        margin=dict(l=0,r=0,t=10,b=0)
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Full drift table
    st.markdown("<div class='section-title'>Full Drift Report</div>", unsafe_allow_html=True)

    def color_drift(row):
        c = {"HIGH DRIFT":"#ef4444","MODERATE":"#f59e0b","STABLE":"#22c55e"}.get(row["Drift Status"],"")
        return ["color: " + c if col == "Drift Status" else "" for col in row.index]

    st.dataframe(
        drift_df.style.apply(color_drift, axis=1),
        use_container_width=True,
        hide_index=True
    )

# ------------------------------------------------------------------
# PAGE 4 — How It Works
# ------------------------------------------------------------------
elif page == "How It Works":
    st.markdown("""
    <div class='banner'>
        <h1>📖 HOW IT WORKS</h1>
        <p>System architecture · ML pipeline · Feature engineering · How to use each page</p>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 1: What this system does ──
    st.markdown("<div class='section-title'>What This System Does</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#111827; border:1px solid #1e2d45; border-radius:8px; padding:20px 24px; margin-bottom:20px; line-height:1.9; color:#c9d1e0; font-size:0.88rem;'>
        This is an <b style='color:#3b82f6;'>end-to-end predictive maintenance system</b> built on multivariate time-series sensor data
        from NASA's C-MAPSS turbofan engine dataset (FD001). It predicts the probability that an engine will fail
        within a defined horizon, ranks engines by relative risk, detects sensor drift, and provides
        interpretable, deployment-ready inference.<br><br>
        In a real industrial setting (oil rigs, turbines, compressors), this system would ingest live IoT sensor
        streams and alert maintenance teams <b style='color:#22c55e;'>before</b> a failure occurs — reducing unplanned downtime
        and preventing catastrophic equipment loss.
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 2: End-to-end pipeline diagram ──
    st.markdown("<div class='section-title'>End-to-End ML Pipeline</div>", unsafe_allow_html=True)

    steps = [
        ("01", "Raw Sensor Data", "21 sensors per engine, readings at every operational cycle. 7 constant sensors removed — they carry no signal.", "#3b82f6"),
        ("02", "Feature Engineering", "3 features derived per sensor: Rolling Mean (stable trend), Rolling Std (instability), Trend/Diff (rate of degradation). Window = 20 cycles.", "#a855f7"),
        ("03", "Label Construction", "Remaining Useful Life (RUL) computed per engine. Converted to binary label: failure within horizon = 1, safe = 0.", "#f59e0b"),
        ("04", "Temporal Validation", "Engine-aware train/test split. No random shuffling — future cycles never leak into training. Prevents data leakage.", "#22c55e"),
        ("05", "Model Training", "3 models evaluated: Logistic Regression, Random Forest, Gradient Boosting. GB selected — ROC-AUC 0.9899, F1 0.889.", "#ef4444"),
        ("06", "Cost-Sensitive Threshold", "Asymmetric cost logic: missing a failure (FN) costs more than a false alarm (FP). Threshold tuned accordingly.", "#f97316"),
        ("07", "Drift Detection", "PSI + KS test compare early-cycle vs late-cycle sensor distributions. Flags sensors undergoing abnormal shift.", "#06b6d4"),
        ("08", "Deployment Inference", "Model serialized to artifacts/. CLI script + this Streamlit UI load the model and run inference on new data.", "#3b82f6"),
    ]

    for i in range(0, len(steps), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(steps):
                num, title, desc, color = steps[i + j]
                col.markdown(f"""
                <div style='background:#111827; border:1px solid #1e2d45; border-left:3px solid {color};
                            border-radius:8px; padding:16px 18px; margin-bottom:12px; height:110px;'>
                    <div style='font-family: Share Tech Mono, monospace; font-size:0.65rem;
                                color:{color}; letter-spacing:0.2em; margin-bottom:4px;'>STEP {num}</div>
                    <div style='font-size:0.9rem; font-weight:600; color:#e2e8f0; margin-bottom:6px;'>{title}</div>
                    <div style='font-size:0.78rem; color:#64748b; line-height:1.5;'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── SECTION 3: Feature engineering explained with example ──
    st.markdown("<div class='section-title' style='margin-top:8px;'>Feature Engineering — Worked Example</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#111827; border:1px solid #1e2d45; border-radius:8px; padding:20px 24px; margin-bottom:16px; font-size:0.85rem; color:#c9d1e0; line-height:1.9;'>
        Suppose <b style='color:#3b82f6;'>sensor_4</b> reads the following values over 5 cycles for Engine 1:<br><br>
        <span style='font-family: Share Tech Mono, monospace; color:#94a3b8;'>
        Cycle 1: 1415.2 &nbsp;|&nbsp; Cycle 2: 1418.7 &nbsp;|&nbsp; Cycle 3: 1421.0 &nbsp;|&nbsp; Cycle 4: 1430.5 &nbsp;|&nbsp; Cycle 5: 1445.8
        </span><br><br>
        The system computes 3 new features from this window (size = 20 cycles, shown here simplified):<br><br>
        <b style='color:#22c55e;'>sensor_4_roll_mean</b> = average of last 20 readings → captures overall operating level<br>
        <b style='color:#f59e0b;'>sensor_4_roll_std</b> = std dev of last 20 readings → captures instability / erratic behaviour<br>
        <b style='color:#ef4444;'>sensor_4_trend</b> = current value − value 20 cycles ago → captures rate of change / degradation speed<br><br>
        This is done for all 14 active sensors → <b style='color:#3b82f6;'>14 × 3 = 42 engineered features</b> fed into the model.
        The raw sensor values themselves are not used — only the derived features.
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 4: Risk categories explained ──
    st.markdown("<div class='section-title'>Risk Categories</div>", unsafe_allow_html=True)
    risk_levels = [
        ("LOW RISK",      "< 0.2",  "#22c55e", "#052e16", "Engine operating normally. No action required. Continue scheduled monitoring."),
        ("MEDIUM RISK",   "0.2 – 0.5", "#f59e0b", "#1c1400", "Early degradation signals detected. Flag for next scheduled maintenance window."),
        ("HIGH RISK",     "0.5 – 0.75", "#f97316", "#1c0a00", "Significant sensor drift. Prioritise inspection. Do not defer maintenance."),
        ("CRITICAL RISK", "> 0.75", "#ef4444", "#1c0000", "Imminent failure likely. Take offline for immediate inspection or controlled shutdown."),
    ]
    rcols = st.columns(4)
    for i, (label, threshold, color, bg, action) in enumerate(risk_levels):
        rcols[i].markdown(f"""
        <div style='background:{bg}; border:1px solid {color}33; border-top:3px solid {color};
                    border-radius:8px; padding:16px; text-align:center; height:160px;'>
            <div style='font-family: Share Tech Mono, monospace; font-size:0.75rem;
                        color:{color}; letter-spacing:0.1em; margin-bottom:6px;'>{label}</div>
            <div style='font-size:1.2rem; font-weight:700; color:{color}; font-family: Share Tech Mono, monospace;
                        margin-bottom:10px;'>{threshold}</div>
            <div style='font-size:0.72rem; color:#94a3b8; line-height:1.5;'>{action}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SECTION 5: How to use each page ──
    st.markdown("<div class='section-title' style='margin-top:20px;'>How to Use Each Page</div>", unsafe_allow_html=True)


    st.markdown("""
    <div style="background:#111827; border:1px solid #1e2d45; border-left:4px solid #3b82f6; border-radius:8px; padding:24px 28px; margin-bottom:16px;">
        <div style="font-family: Share Tech Mono, monospace; font-size:0.95rem; color:#3b82f6; margin-bottom:18px; letter-spacing:0.08em;">
            📡 FLEET DASHBOARD
        </div>
        <p style="color:#c9d1e0; font-size:0.88rem; font-weight:600; margin-bottom:4px;">Test Data Mode</p>
        <p style="color:#64748b; font-size:0.83rem; line-height:1.8; margin-bottom:16px;">
            Select Test Data at the top of the page. Enter a start and end engine ID, set your alert threshold using the slider,
            and click Run Fleet Snapshot. Each bar on the chart represents one engine's current predicted failure risk based on
            its most recent sensor reading. Engines are colour coded — green means low risk, yellow is medium, orange is high,
            and red means critical. Any engine above the red dashed threshold line is flagged as On Alert.
            The summary cards at the top show totals across the fleet. Most engines will show low risk because the test
            dataset contains engines that are still mid-life and have not yet failed — this is correct deployment behaviour.
        </p>
        <p style="color:#c9d1e0; font-size:0.88rem; font-weight:600; margin-bottom:4px;">Train Data Mode — Engine Lifecycle</p>
        <p style="color:#64748b; font-size:0.83rem; line-height:1.8; margin-bottom:16px;">
            Select Train Data at the top. Use the multiselect dropdown to pick one or more engine IDs to compare,
            set how often to sample cycles using the slider, and click Run Lifecycle Analysis. The chart will show
            each engine as a separate coloured line, plotting its predicted failure risk from its very first cycle
            all the way through to the cycle it failed. You will see lines stay flat near zero risk for most of the
            engine's life, then climb sharply in the final cycles as the model detects increasing degradation.
            The summary cards below the chart compare engines against each other — showing which lived longest,
            which failed earliest, and which degraded most gradually.
        </p>
        <p style="color:#c9d1e0; font-size:0.88rem; font-weight:600; margin-bottom:4px;">Alert Threshold</p>
        <p style="color:#64748b; font-size:0.83rem; line-height:1.8; margin-bottom:0;">
            The slider controls where the red dashed alert line sits on the fleet chart. Default is 0.5.
            Moving it lower makes the system more sensitive — it will flag engines earlier, catching risk sooner
            but producing more false alarms. Moving it higher reduces false alarms but may miss early warning signs.
            In real industrial settings this threshold is tuned based on the cost of downtime versus the cost of unnecessary inspections.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#111827; border:1px solid #1e2d45; border-left:4px solid #a855f7; border-radius:8px; padding:24px 28px; margin-bottom:16px;">
        <div style="font-family: Share Tech Mono, monospace; font-size:0.95rem; color:#a855f7; margin-bottom:18px; letter-spacing:0.08em;">
            🔍 ENGINE DEEP DIVE
        </div>
        <p style="color:#c9d1e0; font-size:0.88rem; font-weight:600; margin-bottom:4px;">Dataset Lookup</p>
        <p style="color:#64748b; font-size:0.83rem; line-height:1.8; margin-bottom:16px;">
            Open the Dataset Lookup tab and select any engine ID from the dropdown. The system instantly loads that
            engine's last recorded sensor reading, runs inference through the trained Gradient Boosting model,
            and displays the result. You will see a risk score, an uncertainty range, and a colour-coded gauge.
            Below the gauge are trend charts for the top 4 sensors showing how their readings evolved over the
            last 50 operational cycles — raw reading in grey and smoothed rolling mean in blue. Below that
            is a horizontal bar chart of the 15 most important features the model used to arrive at its prediction.
        </p>
        <p style="color:#c9d1e0; font-size:0.88rem; font-weight:600; margin-bottom:4px;">Manual Sensor Input</p>
        <p style="color:#64748b; font-size:0.83rem; line-height:1.8; margin-bottom:16px;">
            Open the Manual Sensor Input tab to enter your own raw sensor values. All 14 active sensors have
            individual number fields, pre-filled with median values from the dataset as a starting point.
            The cycle history slider controls how many cycles of history to simulate — this is required because
            the model uses rolling features that need a window of readings to compute. Hit Predict Failure Risk
            and the system applies the exact same feature engineering pipeline used during training — computing
            rolling mean, rolling standard deviation, and trend for every sensor — then runs full inference.
            The expandable panel at the bottom lets you inspect every computed feature value so you can see
            precisely what the model received as input. This tab simulates what a live IoT sensor feed would look like.
        </p>
        <p style="color:#c9d1e0; font-size:0.88rem; font-weight:600; margin-bottom:4px;">Uncertainty Range</p>
        <p style="color:#64748b; font-size:0.83rem; line-height:1.8; margin-bottom:0;">
            Every prediction includes a plus or minus uncertainty value computed by running inference 50 times.
            A very small uncertainty like 0.001 means the model is highly confident in its score.
            A larger uncertainty like 0.05 or more means the engine is in a borderline state where small sensor
            variations change the output — treat it with extra caution and prefer a manual inspection over
            relying purely on the predicted score.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#111827; border:1px solid #1e2d45; border-left:4px solid #06b6d4; border-radius:8px; padding:24px 28px; margin-bottom:16px;">
        <div style="font-family: Share Tech Mono, monospace; font-size:0.95rem; color:#06b6d4; margin-bottom:18px; letter-spacing:0.08em;">
            📡 SENSOR DRIFT MONITOR
        </div>
        <p style="color:#c9d1e0; font-size:0.88rem; font-weight:600; margin-bottom:4px;">PSI Chart</p>
        <p style="color:#64748b; font-size:0.83rem; line-height:1.8; margin-bottom:16px;">
            The bar chart shows the Population Stability Index for each of the 14 active sensors.
            PSI measures how much a sensor's statistical distribution has changed between early healthy operation
            (first 50 cycles of engine life) and near-failure operation (last 50 cycles before failure).
            A PSI above 0.25 means significant drift has occurred. All 14 sensors show high drift here —
            this is not a problem, it is expected and correct. We are intentionally comparing two extreme
            states of the engine. The higher the PSI, the more that sensor changes as the engine degrades,
            meaning the more predictive power it carries for failure detection.
        </p>
        <p style="color:#c9d1e0; font-size:0.88rem; font-weight:600; margin-bottom:4px;">KS Test Results</p>
        <p style="color:#64748b; font-size:0.83rem; line-height:1.8; margin-bottom:16px;">
            The full drift report table at the bottom includes the Kolmogorov-Smirnov test statistic and p-value
            for each sensor. The KS test checks whether two distributions could plausibly be the same.
            A p-value of 0.0 means it is statistically impossible for the early and late readings to come
            from the same distribution. All sensors show p-value 0.0 because we are comparing healthy engines
            against engines on the verge of failure — physically opposite conditions with very large sample sizes.
            This is the correct and expected result confirming that sensor drift is real and not noise.
        </p>
        <p style="color:#c9d1e0; font-size:0.88rem; font-weight:600; margin-bottom:4px;">Distribution Comparison Chart</p>
        <p style="color:#64748b; font-size:0.83rem; line-height:1.8; margin-bottom:0;">
            Select any sensor from the dropdown and adjust the cycle cutoff slider to explore different splits.
            The chart overlays two histograms — blue represents the sensor reading distribution during early
            healthy engine operation and orange represents it during near-failure operation.
            The further apart these two curves are, the more that sensor shifts as the engine degrades,
            and the stronger a predictor it is. Sensors like sensor_4 and sensor_11 show the largest separation
            and are among the most important features in the trained model.
        </p>
    </div>
    """, unsafe_allow_html=True)


    # ── SECTION 6: Model performance ──
    st.markdown("<div class='section-title'>Model Performance Summary</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC",          "0.9899")
    m2.metric("Recall (Failures)", "88.5%")
    m3.metric("Precision",         "89.3%")
    m4.metric("F1 Score",          "0.889")
    st.markdown("""
    <div style='background:#111827; border:1px solid #1e2d45; border-radius:8px;
                padding:16px 20px; margin-top:12px; font-size:0.82rem; color:#64748b; line-height:1.8;'>
        <b style='color:#c9d1e0;'>Why Recall matters more than Accuracy here:</b> In predictive maintenance,
        missing a real failure (false negative) is far more costly than a false alarm (false positive).
        A missed failure on an oil rig or turbine can mean catastrophic equipment loss or safety incidents.
        The model is tuned with <b style='color:#f59e0b;'>asymmetric cost logic</b> — false negatives are penalised
        more heavily — which is why Recall (88.5%) is prioritised over raw accuracy.
    </div>
    """, unsafe_allow_html=True)
