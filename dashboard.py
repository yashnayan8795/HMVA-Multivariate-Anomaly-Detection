# dashboard.py
# HMVA Dashboard – preserves your UI, processes on load if needed, supports raw/scored uploads

import os
import sys
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==== Visual theme (high-contrast dark) ====
ACCENT_GRADIENT = "linear-gradient(90deg, #7C3AED, #06B6D4)"  # purple → teal
CARD_BG = "#0f172a"     # slate-900
CARD_BORDER = "#22d3ee" # cyan-400
TEXT_MUTED = "#94a3b8"  # slate-400

PLOTLY_TEMPLATE = "plotly_dark"
PAPER_BG = "rgba(0,0,0,0)"
PLOT_BG  = "rgba(0,0,0,0)"

COLOR_SCORE_LINE = "#60A5FA"   # blue-400
COLOR_TRAIN_BAND = "rgba(34,197,94,0.18)"  # green-500 @ 18%
COLOR_SEV = ["#22C55E", "#EAB308", "#F97316", "#EF4444", "#7F1D1D"]  # Normal→Severe

# -------------------------------------------------------
# Page configuration + Your original styling
# -------------------------------------------------------
st.set_page_config(layout="wide", page_title="Honeywell Anomaly Detection Dashboard", page_icon="🔍")

st.markdown(f"""
<style>
    .main-header {{
        background: {ACCENT_GRADIENT};
        padding: 1rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1.25rem;
        box-shadow: 0 6px 24px rgba(0,0,0,0.25);
    }}
    .metric-card {{
        background: {CARD_BG};
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid {CARD_BORDER};
        margin: 0.5rem 0;
    }}
    .success-box {{
        background: rgba(34,197,94,0.08);
        border: 1px solid rgba(34,197,94,0.35);
        border-radius: 8px;
        padding: 0.85rem;
        margin: 0.75rem 0;
        color: {TEXT_MUTED};
    }}
    .warning-box {{
        background: rgba(234,179,8,0.08);
        border: 1px solid rgba(234,179,8,0.35);
        border-radius: 8px;
        padding: 0.85rem;
        margin: 0.75rem 0;
        color: {TEXT_MUTED};
    }}
    .project-info {{
        background: {CARD_BG};
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #A78BFA; /* violet-400 */
        margin: 1rem 0;
        color: {TEXT_MUTED};
    }}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🔍 Honeywell MultiVariate Anomaly Detection (HMVA)</h1><p>Professional Analysis & Visualization Tool</p></div>', unsafe_allow_html=True)

# -------------------------------------------------------
# Import your pipeline to run analysis in-process
# -------------------------------------------------------
try:
    from pipeline.data import load_csv, validate_regular_intervals, impute_and_clean, split_train_analysis
    from pipeline.model import AnomalyEnsemble
    from pipeline.scoring import rank_normalize, percentile_scale, calibrate_training, smooth_series
    from pipeline.io import add_output_columns
    PIPELINE_OK = True
except Exception:
    PIPELINE_OK = False

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
SEV_BINS = [-0.01, 10, 30, 60, 90, 100]
SEV_LABS = ["Normal (0-10)", "Slight (11-30)", "Moderate (31-60)", "Significant (61-90)", "Severe (91-100)"]

def add_severity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Severity"] = pd.cut(out["Abnormality_score"], bins=SEV_BINS, labels=SEV_LABS)
    return out

def compute_features_count(df: pd.DataFrame, time_col="Time") -> int:
    top_cols = [c for c in df.columns if c.startswith("top_feature_")]
    non_core = set([time_col, "Abnormality_score"]) | set(top_cols)
    return max(df.shape[1] - len(non_core), 0)

def training_stats(df: pd.DataFrame, time_col: str, ts: str, te: str):
    msk = (df[time_col] >= pd.to_datetime(ts)) & (df[time_col] <= pd.to_datetime(te))
    tr = df.loc[msk, "Abnormality_score"]
    return (float(tr.mean()) if not tr.empty else float("nan"),
            float(tr.max()) if not tr.empty else float("nan"))

def severity_counts(df: pd.DataFrame) -> pd.Series:
    return df["Severity"].value_counts().reindex(SEV_LABS, fill_value=0)

def severity_barplot_plotly(counts: pd.Series, title: str):
    dfb = counts.rename_axis("Severity").reset_index(name="Count")
    color_map = {lab: COLOR_SEV[i] for i, lab in enumerate(SEV_LABS)}
    fig = px.bar(
        dfb, x="Severity", y="Count",
        color="Severity", color_discrete_map=color_map,
        title=title
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis_title="Severity Level",
        yaxis_title="Number of Rows",
        height=420,
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color="#E5E7EB"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.18)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.18)")
    )
    return fig

def anomaly_timeline_plotly(df: pd.DataFrame, time_col: str,
                            train_start, train_end,
                            title: str, line_color=COLOR_SCORE_LINE):
    # Ensure proper datetime and sane bounds
    x = pd.to_datetime(df[time_col], errors="coerce")
    ts = pd.to_datetime(train_start)
    te = pd.to_datetime(train_end)

    fig = go.Figure()

    # Training period band (behind traces, exact bounds)
    fig.add_vrect(
        x0=ts, x1=te,
        fillcolor=COLOR_TRAIN_BAND,
        opacity=1.0, layer="below", line_width=0,
        annotation_text="Training Period",
        annotation_position="top left",
        annotation=dict(font=dict(color="#10B981"))  # emerald label
    )

    # Anomaly score line
    fig.add_trace(go.Scatter(
        x=x, y=df["Abnormality_score"],
        mode="lines", name="Abnormality Score",
        line=dict(color=line_color, width=1.8),
        hovertemplate="<b>Time:</b> %{x}<br><b>Score:</b> %{y:.2f}<extra></extra>"
    ))

    # Severity guides (higher contrast for dark theme)
    guides = [
        (10, "Normal (≤10)",  "#22C55E"),   # green
        (30, "Slight (≤30)",  "#EAB308"),   # yellow
        (60, "Moderate (≤60)","#F97316"),   # orange
        (90, "Significant (≤90)", "#EF4444") # red
    ]
    for y, label, col in guides:
        fig.add_hline(y=y, line_dash="dash", line_color=col,
                      annotation_text=label, annotation_font_color=col)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=title,
        xaxis_title="Time", yaxis_title="Anomaly Score (0–100)",
        yaxis=dict(range=[0, 105], gridcolor="rgba(148,163,184,0.18)"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.18)"),
        height=520, showlegend=True,
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color="#E5E7EB")  # slate-200
    )
    return fig

def _choose_top(colnames, contrib_vec, min_pct=0.01):
    total = float(np.sum(contrib_vec))
    if total <= 0 or not np.isfinite(total): 
        return [""]*7
    pct = contrib_vec / (total + 1e-12)
    idx = np.where(pct >= min_pct)[0]
    entries = [(float(contrib_vec[i]), colnames[i]) for i in idx]
    entries.sort(key=lambda x: (-x[0], x[1]))
    names = [name for _, name in entries][:7]
    if len(names) < 7: names += [""] * (7 - len(names))
    return names

def run_full_pipeline_on_path(csv_path: str, time_col: str,
                              train_start: str, train_end: str, analysis_end: str,
                              smooth: int = 3) -> pd.DataFrame:
    """Run the same pipeline as main.py and return a scored DataFrame (in-memory)."""
    if not PIPELINE_OK:
        raise RuntimeError("Pipeline modules not importable. Run from project root.")
    df_num = load_csv(csv_path, time_col)
    validate_regular_intervals(df_num)
    df_num = impute_and_clean(df_num)

    original = pd.read_csv(csv_path)
    original[time_col] = pd.to_datetime(original[time_col], errors="coerce")
    original = original.set_index(original[time_col]).drop(columns=[time_col])
    original = original.reindex(df_num.index)

    train_df, analysis_df = split_train_analysis(df_num, train_start, train_end, analysis_end)

    model = AnomalyEnsemble()
    model.fit(train_df)

    ch = model.score_rows(analysis_df)
    R = (rank_normalize(ch['S1']) + rank_normalize(ch['S2']) + rank_normalize(ch['S3'])) / 3.0
    scaled = percentile_scale(R)

    idx = analysis_df.index
    train_mask = (idx >= pd.to_datetime(train_start)) & (idx <= pd.to_datetime(train_end))
    scaled = calibrate_training(scaled, train_mask)
    if smooth and int(smooth) > 1:
        scaled = smooth_series(scaled, window=int(smooth))

    # feature attribution (z + recon + IF delta)
    Z, RE = ch['Z'], ch['RECON']
    z_abs, re_abs = np.abs(Z), np.abs(RE)
    base = -model.iforest.decision_function(Z)
    deltas = np.zeros_like(Z)
    for i in range(Z.shape[1]):
        Zm = Z.copy(); Zm[:, i] = 0.0
        mod = -model.iforest.decision_function(Zm)
        deltas[:, i] = np.maximum(0.0, base - mod)

    total = z_abs + re_abs + deltas
    top_lists = [_choose_top(model.colnames, np.maximum(total[r, :], 0.0)) for r in range(total.shape[0])]

    out_df = original.loc[idx].copy()
    out_df = add_output_columns(out_df,
                                pd.Series(scaled, index=idx, name='Abnormality_score'),
                                top_lists)
    out_df.insert(0, time_col, out_df.index.astype('datetime64[ns]'))
    return out_df.reset_index(drop=True)

@st.cache_data(show_spinner="Preparing pretrained data…")
def load_or_build_pretrained(time_col: str, train_start: str, train_end: str, analysis_end: str, smooth: int):
    """
    If TEP_Output_Scored.csv exists, load it. Otherwise, if pipeline + TEP_Train_Test.csv exist,
    process on the fly and return the scored DataFrame (in memory).
    """
    if os.path.exists("TEP_Output_Scored.csv"):
        df = pd.read_csv("TEP_Output_Scored.csv", parse_dates=[time_col])
        if "Abnormality_score" not in df.columns:
            raise ValueError("TEP_Output_Scored.csv missing 'Abnormality_score'.")
        return add_severity(df)

    if PIPELINE_OK and os.path.exists("TEP_Train_Test.csv"):
        df = run_full_pipeline_on_path("TEP_Train_Test.csv", time_col, train_start, train_end, analysis_end, smooth)
        return add_severity(df)

    return None

# -------------------------------------------------------
# Tabs (names unchanged)
# -------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Original Dataset Analysis", "🔬 Custom Dataset Analysis"])

# =========================
# TAB 1 – Original Dataset
# =========================
with tab1:
    st.markdown("## 📊 Original Dataset Analysis")
    st.markdown("This tab displays the results from the original TEP dataset that was processed and scored. "
                "If a pre-scored file is not found, the app will process the raw dataset automatically.")

    st.markdown("""
    <div class="project-info">
        <h4>🎯 Project Overview</h4>
        <p>This project implements Honeywell's multivariate anomaly detection algorithm for industrial process monitoring. 
        The system analyzes time-series data to identify abnormal patterns and provides comprehensive insights into 
        contributing factors.</p>
        <h4>🔬 Methodology</h4>
        <ul>
            <li><strong>Training Period:</strong> Uses a baseline period (2004-01-01 to 2004-01-05) to establish normal behavior patterns</li>
            <li><strong>Anomaly Detection:</strong> Ensemble signals → ranking → percentile scaling (0–100)</li>
            <li><strong>Feature Analysis:</strong> Top contributing features per timestamp</li>
            <li><strong>Scoring:</strong> Abnormality scores (0–100) with severity classification</li>
        </ul>
        <h4>📁 Dataset Details</h4>
        <ul>
            <li><strong>Source:</strong> TEP (Tennessee Eastman Process)</li>
            <li><strong>Training Period:</strong> 2004-01-01 00:00 to 2004-01-05 23:59</li>
            <li><strong>Analysis Period:</strong> up to 2004-01-19 07:59</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    df_original = load_or_build_pretrained(
        time_col="Time",
        train_start="2004-01-01 00:00",
        train_end="2004-01-05 23:59",
        analysis_end="2004-01-19 07:59",
        smooth=3
    )

    if df_original is None:
        if not PIPELINE_OK:
            st.error("❌ Could not import pipeline modules. Place the app in the project root or fix PYTHONPATH.")
        elif not os.path.exists("TEP_Train_Test.csv"):
            st.error("❌ TEP_Train_Test.csv not found. Put the original raw CSV in the project folder.")
        else:
            st.error("❌ Unexpected issue preparing the pretrained dataset.")
    else:
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Rows", f"{df_original.shape[0]:,}")
        with col2: st.metric("Features", f"{compute_features_count(df_original):,}")
        with col3: st.metric("Date Range", f"{(df_original['Time'].max() - df_original['Time'].min()).days} days")
        with col4: st.metric("Max Score", f"{df_original['Abnormality_score'].max():.1f}")

        # Training period validation
        st.markdown("### 📚 Training Period Validation (Original Dataset)")
        mean_tr, max_tr = training_stats(df_original, "Time", "2004-01-01 00:00", "2004-01-05 23:59")
        colA, colB = st.columns(2)
        with colA:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Training Mean Score", f"{mean_tr:.2f}", delta="✅ Good" if mean_tr < 10 else "❌ High")
            st.markdown('</div>', unsafe_allow_html=True)
        with colB:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Training Max Score", f"{max_tr:.2f}", delta="✅ Good" if max_tr < 25 else "❌ High")
            st.markdown('</div>', unsafe_allow_html=True)
        if mean_tr < 10 and max_tr < 25:
            st.markdown('<div class="success-box">🎉 <strong>Training Validation PASSED:</strong> Mean < 10 and Max < 25 requirements met!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">⚠️ <strong>Training Validation WARNING:</strong> Scores may need calibration.</div>', unsafe_allow_html=True)

        # Timeline
        st.markdown("### 📈 Original Dataset Anomaly Timeline")
        st.plotly_chart(
            anomaly_timeline_plotly(df_original, "Time", "2004-01-01 00:00", "2004-01-05 23:59",
                                    "Original TEP Dataset: Complete Anomaly Timeline with Training Period Highlighted",
                                    line_color="blue"),
            use_container_width=True
        )

        # Severity
        st.markdown("### 🧪 Original Dataset Severity Distribution")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig_sev = severity_barplot_plotly(severity_counts(df_original), "Original Dataset: Severity Distribution")
            st.plotly_chart(fig_sev, use_container_width=True)
        with col2:
            st.markdown("### 📊 Severity Statistics")
            total_rows = len(df_original)
            for severity, count in severity_counts(df_original).items():
                pct = (count / total_rows * 100) if total_rows else 0.0
                st.metric(severity, f"{count:,}", f"{pct:.1f}%")

        # Top anomalies table
        st.markdown("### 📋 Top 50 Anomalies (Original Dataset)")
        top_feature_cols = [c for c in df_original.columns if c.startswith("top_feature_")]
        top_anoms = df_original.sort_values("Abnormality_score", ascending=False).head(50)
        st.dataframe(top_anoms[["Time", "Abnormality_score", "Severity"] + top_feature_cols], use_container_width=True)

        # Advanced Visualizations (Expandable)
        with st.expander("🔬 Advanced Analysis & Insights", expanded=False):
            st.markdown("### 📊 Score Distribution")
            bins = np.linspace(0, 100, 41)
            hist, edges = np.histogram(df_original["Abnormality_score"], bins=bins)
            fig = go.Figure(data=[go.Bar(x=edges[:-1], y=hist, width=np.diff(edges))])
            fig.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Score (0–100)", yaxis_title="Count",
                              title="Abnormality Score Histogram", paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                              font=dict(color="#E5E7EB"))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🧩 Feature Contribution Frequency Over Time (Severe/Significant)")
            df_sev = df_original[df_original["Abnormality_score"] >= 60].copy()  # significant & severe
            if not df_sev.empty:
                # melt counts per hour
                df_sev["bucket"] = df_sev["Time"].dt.floor("H")
                vals = df_sev[top_feature_cols].melt(id_vars=["bucket"], value_name="feat")["feat"]
                tbl = vals.to_frame().join(df_sev["bucket"].repeat(len(top_feature_cols))).dropna()
                freq = tbl.groupby(["bucket", "feat"]).size().reset_index(name="count")
                # take top 10 overall to keep readable
                top10 = freq.groupby("feat")["count"].sum().nlargest(10).index
                freq = freq[freq["feat"].isin(top10)]
                if not freq.empty:
                    fig = px.area(freq, x="bucket", y="count", color="feat", title="Top Contributors Over Time (≥60)")
                    fig.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Time", yaxis_title="Mentions",
                                      paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                                      font=dict(color="#E5E7EB"))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No significant anomalies found for feature contribution analysis.")
            else:
                st.info("No significant anomalies found for feature contribution analysis.")

            st.markdown("### 🗓️ Temporal Pattern Heatmap (Mean Score by Day/Hour)")
            dd = df_original.copy()
            dd["dow"] = dd["Time"].dt.day_name().str[:3]
            dd["hour"] = dd["Time"].dt.hour
            pivot = dd.pivot_table(index="dow", columns="hour", values="Abnormality_score", aggfunc="mean")
            # order days
            order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            pivot = pivot.reindex([d for d in order if d in pivot.index])
            if not pivot.empty:
                fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Turbo", origin="lower",
                                title="Mean Abnormality Score by Day of Week × Hour")
                fig.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Hour", yaxis_title="Day",
                                  paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                                  font=dict(color="#E5E7EB"))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🚨 Anomaly Incidents (Segments)")
            thr = st.slider("Incident threshold (score ≥)", 60, 95, 90, key="incident_threshold_orig")
            dff = df_original.copy()
            dff["is_high"] = dff["Abnormality_score"] >= thr
            # detect segments
            seg_id = (dff["is_high"].ne(dff["is_high"].shift(1)).cumsum()) * dff["is_high"]
            segments = dff.groupby(seg_id).agg(start=("Time","min"), end=("Time","max"),
                                               peak=("Abnormality_score","max"),
                                               rows=("Abnormality_score","size")).reset_index(drop=True)
            segments = segments.dropna()  # remove id 0 (non‑high)
            if not segments.empty:
                segments["duration_min"] = (segments["end"] - segments["start"]).dt.total_seconds()/60
                st.dataframe(segments.sort_values("peak", ascending=False), use_container_width=True)
                # timeline viz
                fig = go.Figure()
                for _, r in segments.iterrows():
                    fig.add_trace(go.Scatter(x=[r["start"], r["end"]], y=[r["peak"], r["peak"]],
                                             mode="lines", line=dict(width=6), name=f"{r['peak']:.1f}"))
                fig.update_layout(template=PLOTLY_TEMPLATE, title="Incident Timelines (peak value as y)",
                                  xaxis_title="Time", yaxis_title="Peak Score",
                                  paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, 
                                  font=dict(color="#E5E7EB"), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No incidents found above threshold {thr}.")

            st.markdown("### 🔎 Drill‑down: Feature Attribution at a Timestamp")
            # pick top‑N anomalies to choose from
            topN = df_original.sort_values("Abnormality_score", ascending=False).head(300)[["Time","Abnormality_score"]]
            choice = st.selectbox("Pick a timestamp (top anomalies)", options=topN["Time"].astype(str), key="drilldown_orig")
            row = df_original[df_original["Time"].astype(str) == choice].iloc[0]
            # convert top_feature_1..7 → ordered list
            tops = [row[c] for c in top_feature_cols if isinstance(row[c], str) and row[c].strip()][:7]
            if tops:
                bar = pd.DataFrame({"feature": tops, "rank": range(1, len(tops)+1)})
                bar["weight"] = (len(tops)+1 - bar["rank"])  # simple descending weight for visualization
                fig = px.bar(bar, x="feature", y="weight", title=f"Top Contributors @ {choice}")
                fig.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Feature", yaxis_title="Relative Weight",
                                  paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                                  font=dict(color="#E5E7EB"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No feature attribution data available for selected timestamp.")

            st.markdown("### 📈 Cumulative Severe/Significant Events")
            df_ev = df_original.copy()
            df_ev["sev_flag"] = (df_ev["Abnormality_score"] >= 60).astype(int)
            df_ev = df_ev.sort_values("Time")
            df_ev["cum_events"] = df_ev["sev_flag"].cumsum()
            fig = go.Figure(go.Scatter(x=df_ev["Time"], y=df_ev["cum_events"], mode="lines"))
            fig.update_layout(template=PLOTLY_TEMPLATE, title="Cumulative Count of Significant+Severe Points",
                              xaxis_title="Time", yaxis_title="Cumulative Events",
                              paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                              font=dict(color="#E5E7EB"))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🔁 Correlation Shift (Train vs. High‑Severity)")
            # choose subset of features (avoid overwhelming)
            numeric_cols = df_original.select_dtypes(include=[np.number]).columns.tolist()
            # remove score column from feature list
            feat_cols = [c for c in numeric_cols if c not in ["Abnormality_score"]]
            if len(feat_cols) > 1:
                # training mask (reuse your fixed window)
                tr_mask = (df_original["Time"] >= pd.to_datetime("2004-01-01 00:00")) & (df_original["Time"] <= pd.to_datetime("2004-01-05 23:59"))
                high_mask = df_original["Abnormality_score"] >= 60
                
                if tr_mask.sum() > 10 and high_mask.sum() > 10:  # need sufficient data
                    # compute correlations
                    C_train = df_original.loc[tr_mask, feat_cols].corr().fillna(0)
                    C_high  = df_original.loc[high_mask, feat_cols].corr().fillna(0)
                    C_diff  = (C_high - C_train)
                    
                    # pick top 12 by max absolute shift for readability
                    shift_score = C_diff.abs().sum().sort_values(ascending=False).head(12).index
                    C_diff_small = C_diff.loc[shift_score, shift_score]
                    fig = px.imshow(C_diff_small, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                                    title="Correlation Shift (High severity − Training)")
                    fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                                      font=dict(color="#E5E7EB"))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Insufficient data for correlation analysis (need >10 samples in each group).")
            else:
                st.info("Need at least 2 numeric features for correlation analysis.")

# =========================
# TAB 2 – Custom Dataset
# =========================
with tab2:
    st.markdown("## 🔬 Custom Dataset Analysis")
    st.markdown("Upload your own dataset (raw or scored) to analyze with the same anomaly detection pipeline.")

    # Custom sidebar (kept inside Tab 2)
    st.sidebar.header("⚙️ Custom Dataset Configuration")
    upload_option = st.sidebar.radio(
        "Upload Type",
        ["Raw CSV (will be processed)", "Scored CSV (already processed)"],
        help="Choose whether to upload raw data for processing or pre-scored data for analysis"
    )

    st.sidebar.markdown("### ⚙️ Processing Configuration")
    time_col_custom = st.sidebar.text_input("Time Column Name", value="Time", help="Name of your timestamp column")
    train_start_custom = st.sidebar.text_input("Training Start", value="2004-01-01 00:00", help="Start of normal period")
    train_end_custom   = st.sidebar.text_input("Training End", value="2004-01-05 23:59", help="End of normal period")
    analysis_end_custom = st.sidebar.text_input("Analysis End", value="2004-01-19 07:59", help="End of analysis period")
    st.session_state["train_start_custom"] = train_start_custom
    st.session_state["train_end_custom"] = train_end_custom

    st.sidebar.markdown("### 🔧 Processing Options")
    smooth_data = st.sidebar.checkbox("Apply Smoothing", value=True, help="Apply temporal smoothing to scores")

    # -------- RAW CSV path (in-process, no subprocess) --------
    if upload_option == "Raw CSV (will be processed)":
        st.sidebar.markdown("### 📁 Raw Data Upload")
        raw_file = st.sidebar.file_uploader("Upload Raw CSV", type="csv", help="Upload your raw dataset CSV file")
        if raw_file and st.sidebar.button("🚀 Process Dataset"):
            with st.spinner("Processing your dataset..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                        tmp.write(raw_file.getvalue())
                        tmp_path = tmp.name
                    scored_df = run_full_pipeline_on_path(
                        csv_path=tmp_path,
                        time_col=time_col_custom,
                        train_start=train_start_custom,
                        train_end=train_end_custom,
                        analysis_end=analysis_end_custom,
                        smooth=3 if smooth_data else 1,
                    )
                    os.unlink(tmp_path)
                    st.session_state['custom_df'] = add_severity(scored_df)
                    st.session_state['custom_filename'] = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    st.session_state['custom_time_col'] = time_col_custom
                    st.success("✅ Dataset processed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error processing dataset: {e}")

    # -------- Scored CSV path --------
    else:
        st.sidebar.markdown("### 📁 Scored Data Upload")
        scored_file = st.sidebar.file_uploader("Upload Scored CSV", type="csv",
                                               help="Must contain Abnormality_score & top_feature_1..7")
        if scored_file:
            try:
                df_custom = pd.read_csv(scored_file, parse_dates=[time_col_custom])
                st.session_state['custom_df'] = add_severity(df_custom)
                st.session_state['custom_filename'] = scored_file.name
                st.session_state['custom_time_col'] = time_col_custom
                st.success(f"✅ Successfully loaded scored dataset: {scored_file.name}")
            except Exception as e:
                st.error(f"❌ Error loading scored dataset: {e}")

    # -------- Visualization for whichever custom_df is present --------
    if 'custom_df' in st.session_state and 'custom_filename' in st.session_state:
        df_custom = st.session_state['custom_df']
        custom_filename = st.session_state['custom_filename']
        custom_time_col = st.session_state.get('custom_time_col', 'Time')

        st.markdown(f"### 📊 Analysis Results: {custom_filename}")

        if "Abnormality_score" not in df_custom.columns:
            st.error("❌ Uploaded/processed file must contain 'Abnormality_score' column.")
        else:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Rows", f"{df_custom.shape[0]:,}")
            with col2: st.metric("Features", f"{compute_features_count(df_custom, custom_time_col):,}")
            with col3: st.metric("Date Range", f"{(df_custom[custom_time_col].max() - df_custom[custom_time_col].min()).days} days")
            with col4: st.metric("Max Score", f"{df_custom['Abnormality_score'].max():.1f}")

            # Training validation with user-set dates
            st.markdown("### 📚 Training Period Validation (Custom Dataset)")
            ts_val = st.session_state.get("train_start_custom", "2004-01-01 00:00")
            te_val = st.session_state.get("train_end_custom", "2004-01-05 23:59")
            mean_c, max_c = training_stats(df_custom, custom_time_col, ts_val, te_val)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Training Mean Score", f"{mean_c:.2f}", delta="✅ Good" if mean_c < 10 else "❌ High")
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Training Max Score", f"{max_c:.2f}", delta="✅ Good" if max_c < 25 else "❌ High")
                st.markdown('</div>', unsafe_allow_html=True)
            if mean_c < 10 and max_c < 25:
                st.markdown('<div class="success-box">🎉 <strong>Training Validation PASSED:</strong> Mean < 10 and Max < 25 requirements met!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">⚠️ <strong>Training Validation WARNING:</strong> Scores may need calibration.</div>', unsafe_allow_html=True)

            # Timeline
            st.markdown("### 📈 Custom Dataset Anomaly Timeline")
            st.plotly_chart(
                anomaly_timeline_plotly(df_custom, custom_time_col, ts_val, te_val,
                                        "Custom Dataset: Complete Anomaly Timeline with Training Period Highlighted",
                                        line_color="purple"),
                use_container_width=True
            )

            # Severity
            st.markdown("### 🧪 Custom Dataset Severity Distribution")
            col1, col2 = st.columns([2, 1])
            with col1:
                fig_sev_custom = severity_barplot_plotly(severity_counts(df_custom), "Custom Dataset: Severity Distribution")
                st.plotly_chart(fig_sev_custom, use_container_width=True)
            with col2:
                st.markdown("### 📊 Severity Statistics")
                total_custom = len(df_custom)
                for severity, count in severity_counts(df_custom).items():
                    pct = (count / total_custom * 100) if total_custom else 0.0
                    st.metric(severity, f"{count:,}", f"{pct:.1f}%")

            # Top anomalies
            st.markdown("### 📋 Top 50 Anomalies (Custom Dataset)")
            top_feature_cols_custom = [c for c in df_custom.columns if c.startswith("top_feature_")]
            top_anoms_custom = df_custom.sort_values("Abnormality_score", ascending=False).head(50)
            st.dataframe(top_anoms_custom[[custom_time_col, "Abnormality_score", "Severity"] + top_feature_cols_custom],
                         use_container_width=True)

            # Downloads
            st.markdown("### 💾 Export Custom Dataset Results")
            colD1, colD2 = st.columns(2)
            with colD1:
                st.download_button(
                    "📥 Download Custom Results CSV",
                    df_custom.to_csv(index=False).encode("utf-8"),
                    file_name=f"{os.path.splitext(custom_filename)[0]}_scored.csv",
                    mime="text/csv"
                )
            with colD2:
                sev_summary = severity_counts(df_custom).rename_axis("Severity").reset_index(name="Count")
                sev_summary["Percentage"] = (sev_summary["Count"] / max(int(sev_summary["Count"].sum()), 1) * 100).round(2)
                st.download_button(
                    "📊 Download Custom Severity Summary",
                    sev_summary.to_csv(index=False).encode("utf-8"),
                    file_name=f"{os.path.splitext(custom_filename)[0]}_severity_summary.csv",
                    mime="text/csv"
                )

            # Advanced Visualizations for Custom Dataset (Expandable)
            with st.expander("🔬 Advanced Analysis & Insights (Custom Dataset)", expanded=False):
                st.markdown("### 📊 Score Distribution")
                bins = np.linspace(0, 100, 41)
                hist, edges = np.histogram(df_custom["Abnormality_score"], bins=bins)
                fig = go.Figure(data=[go.Bar(x=edges[:-1], y=hist, width=np.diff(edges))])
                fig.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Score (0–100)", yaxis_title="Count",
                                  title="Abnormality Score Histogram", paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                                  font=dict(color="#E5E7EB"))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 🧩 Feature Contribution Frequency Over Time (Severe/Significant)")
                df_sev_custom = df_custom[df_custom["Abnormality_score"] >= 60].copy()
                if not df_sev_custom.empty:
                    df_sev_custom["bucket"] = df_sev_custom[custom_time_col].dt.floor("H")
                    vals = df_sev_custom[top_feature_cols_custom].melt(id_vars=["bucket"], value_name="feat")["feat"]
                    tbl = vals.to_frame().join(df_sev_custom["bucket"].repeat(len(top_feature_cols_custom))).dropna()
                    freq = tbl.groupby(["bucket", "feat"]).size().reset_index(name="count")
                    top10 = freq.groupby("feat")["count"].sum().nlargest(10).index
                    freq = freq[freq["feat"].isin(top10)]
                    if not freq.empty:
                        fig = px.area(freq, x="bucket", y="count", color="feat", title="Top Contributors Over Time (≥60)")
                        fig.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Time", yaxis_title="Mentions",
                                          paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                                          font=dict(color="#E5E7EB"))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No significant anomalies found for feature contribution analysis.")
                else:
                    st.info("No significant anomalies found for feature contribution analysis.")

                st.markdown("### 🚨 Anomaly Incidents (Segments)")
                thr_custom = st.slider("Incident threshold (score ≥)", 60, 95, 90, key="incident_threshold_custom")
                dff_custom = df_custom.copy()
                dff_custom["is_high"] = dff_custom["Abnormality_score"] >= thr_custom
                seg_id_custom = (dff_custom["is_high"].ne(dff_custom["is_high"].shift(1)).cumsum()) * dff_custom["is_high"]
                segments_custom = dff_custom.groupby(seg_id_custom).agg(start=(custom_time_col,"min"), end=(custom_time_col,"max"),
                                                                       peak=("Abnormality_score","max"),
                                                                       rows=("Abnormality_score","size")).reset_index(drop=True)
                segments_custom = segments_custom.dropna()
                if not segments_custom.empty:
                    segments_custom["duration_min"] = (segments_custom["end"] - segments_custom["start"]).dt.total_seconds()/60
                    st.dataframe(segments_custom.sort_values("peak", ascending=False), use_container_width=True)
                    fig = go.Figure()
                    for _, r in segments_custom.iterrows():
                        fig.add_trace(go.Scatter(x=[r["start"], r["end"]], y=[r["peak"], r["peak"]],
                                                 mode="lines", line=dict(width=6), name=f"{r['peak']:.1f}"))
                    fig.update_layout(template=PLOTLY_TEMPLATE, title="Incident Timelines (peak value as y)",
                                      xaxis_title="Time", yaxis_title="Peak Score",
                                      paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, 
                                      font=dict(color="#E5E7EB"), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No incidents found above threshold {thr_custom}.")

                st.markdown("### 📈 Cumulative Severe/Significant Events")
                df_ev_custom = df_custom.copy()
                df_ev_custom["sev_flag"] = (df_ev_custom["Abnormality_score"] >= 60).astype(int)
                df_ev_custom = df_ev_custom.sort_values(custom_time_col)
                df_ev_custom["cum_events"] = df_ev_custom["sev_flag"].cumsum()
                fig = go.Figure(go.Scatter(x=df_ev_custom[custom_time_col], y=df_ev_custom["cum_events"], mode="lines"))
                fig.update_layout(template=PLOTLY_TEMPLATE, title="Cumulative Count of Significant+Severe Points",
                                  xaxis_title="Time", yaxis_title="Cumulative Events",
                                  paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                                  font=dict(color="#E5E7EB"))
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Use the sidebar to upload and analyze your custom dataset.")
        st.markdown("""
        ### 📋 **How to Use Custom Dataset Analysis:**
        **Option 1: Raw Data Processing** — Upload raw CSV, set training/analysis windows, click *Process Dataset*.
        **Option 2: Pre-scored Data** — Upload a CSV with `Abnormality_score` and `top_feature_1..7` to visualize immediately.
        """)
