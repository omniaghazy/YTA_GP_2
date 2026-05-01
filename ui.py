import os
import sys
from pathlib import Path

# Silence technical noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress absl logs
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import polars as pl
import time
import requests
from datetime import datetime
import tensorflow as tf
from xgboost import XGBClassifier
from streamlit_lottie import st_lottie

# Import classes for module shim
import src.drive_failure_system as dfs

# 1. ROBUST PATH SETTINGS
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

@st.cache_resource
def load_champion_assets():
    """Loads model, scaler, and metadata with strict path validation."""
    try:
        # Paths derived from BASE_DIR
        info_path = MODELS_DIR / "best_model_info.json"
        scaler_path = MODELS_DIR / "scaler.joblib"

        # 1. Validate info file
        if not info_path.exists():
            st.error(f"❌ Critical Error: '{info_path.name}' missing from models/ folder.")
            return None, None, None

        with open(info_path, 'r') as f:
            info = json.load(f)

        # 2. Validate model file
        champion_path = MODELS_DIR / info["champion_file"]
        if not champion_path.exists():
            st.error(f"❌ Model file not found: '{info['champion_file']}'. Check your models/ directory.")
            return None, None, None

        # 3. Validate scaler
        if not scaler_path.exists():
            st.error("❌ Scaler missing: 'models/scaler.joblib' is required for deployment.")
            return None, None, None

        # Load assets
        scaler = joblib.load(scaler_path)
        if info['champion_file'].endswith(('.h5', '.keras')):
            model = tf.keras.models.load_model(champion_path)
        elif info['champion_file'].endswith('.json'):
            model = XGBClassifier()
            model.load_model(str(champion_path))
        else:
            model = joblib.load(champion_path)
        
        # Compatibility guard (prevents runtime shape crashes).
        scaler_n = int(getattr(scaler, "n_features_in_", 0) or 0)
        model_shape = getattr(model, "input_shape", None)
        if isinstance(model_shape, list):
            model_shape = model_shape[0]
        if model_shape is not None and len(model_shape) == 2 and model_shape[1] is not None and scaler_n:
            model_n = int(model_shape[1])
            if model_n != scaler_n:
                st.error(
                    f"🚨 Artifact mismatch: model expects {model_n} features, "
                    f"but scaler expects {scaler_n}. "
                    "Upload matching artifacts from the same training run."
                )
                return info, None, None

        return info, model, scaler
    except Exception as e:
        st.error(f"🚨 Deployment Asset Error: {e}")
        return None, None, None

# Initialize Assets
info, model, scaler = load_champion_assets()

def _detect_model_mode(model_obj):
    """Return ('sequence'|'tabular', expected_feature_count, expected_window_size_or_none)."""
    # Non-keras models are treated as tabular sklearn-like estimators.
    if not hasattr(model_obj, "input_shape"):
        return "tabular", None, None

    shape = model_obj.input_shape
    # Keras may return list for multi-input models; we only support single-input here.
    if isinstance(shape, list):
        shape = shape[0]

    # Typical shapes:
    # Tabular DNN: (None, n_features)
    # Sequence model: (None, window_size, n_features)
    if len(shape) == 3:
        return "sequence", int(shape[2]), int(shape[1])
    if len(shape) == 2:
        return "tabular", int(shape[1]), None
    return "tabular", None, None

def _expected_feature_count(model_obj, scaler_obj, info_obj):
    if hasattr(scaler_obj, "n_features_in_"):
        return int(scaler_obj.n_features_in_)
    mode, expected_features, _ = _detect_model_mode(model_obj)
    if expected_features is not None:
        return int(expected_features)
    return len(info_obj.get("features", []))

def _align_feature_vector(vec_2d: np.ndarray, target_count: int) -> np.ndarray:
    """Pad/truncate feature matrix to match expected feature count."""
    cur = vec_2d.shape[1]
    if cur == target_count:
        return vec_2d
    if cur < target_count:
        pad = np.zeros((vec_2d.shape[0], target_count - cur), dtype=vec_2d.dtype)
        return np.hstack([vec_2d, pad])
    return vec_2d[:, :target_count]

def _ensure_feature_columns(df_pl: pl.DataFrame, feature_names):
    """
    Ensure all required feature columns exist.
    For *_lag1 features, derive from base SMART feature by serial_number shift.
    Remaining missing columns are filled with zeros.
    """
    if "serial_number" in df_pl.columns:
        df_pl = df_pl.sort("serial_number")
    if "date" in df_pl.columns:
        df_pl = df_pl.sort(["serial_number", "date"]) if "serial_number" in df_pl.columns else df_pl.sort("date")

    missing = [f for f in feature_names if f not in df_pl.columns]
    lag_exprs = []
    for col in missing:
        if col.endswith("_lag1"):
            base = col[:-5]
            if base in df_pl.columns:
                if "serial_number" in df_pl.columns:
                    lag_exprs.append(pl.col(base).shift(1).over("serial_number").fill_null(0).alias(col))
                else:
                    lag_exprs.append(pl.col(base).shift(1).fill_null(0).alias(col))
    if lag_exprs:
        df_pl = df_pl.with_columns(lag_exprs)

    still_missing = [f for f in feature_names if f not in df_pl.columns]
    if still_missing:
        df_pl = df_pl.with_columns([pl.lit(0.0).alias(c) for c in still_missing])
    return df_pl

# ──────────────────────────────────────────────────────────────────────────────
# 1. Page Configuration & Premium Styling
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ELITE Predictive Maintenance | YTA",
    page_icon="💎",
    layout="wide",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0b0e14; color: #e0e0e0; }
    .result-card {
        padding: 30px;
        border-radius: 20px;
        background: #1a1c23;
        border: 1px solid #30363d;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        text-align: center;
    }
    .glass-card {
        background: rgba(26, 28, 35, 0.65);
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.125);
        padding: 40px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        position: relative;
        overflow: hidden;
    }
    .pulse-high {
        animation: pulse-red 2s infinite;
    }
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 20px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    .gauge-container {
        position: relative;
        width: 200px;
        height: 200px;
        margin: 0 auto;
    }
    .gauge-bg { fill: none; stroke: #30363d; stroke-width: 12; }
    .gauge-fill {
        fill: none;
        stroke-width: 12;
        stroke-linecap: round;
        transition: stroke-dasharray 1s ease-in-out;
        transform: rotate(-90deg);
        transform-origin: 50% 50%;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        margin-left: 8px;
        color: #58a6ff;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 220px;
        background-color: #1a1c23;
        color: #fff;
        text-align: center;
        border-radius: 8px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.3s;
        border: 1px solid #30363d;
        font-size: 0.8em;
        font-weight: normal;
        line-height: 1.4;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .glow-low { box-shadow: 0 0 20px rgba(40, 167, 69, 0.3); border-color: #28a745 !important; }
    .glow-med { box-shadow: 0 0 20px rgba(255, 193, 7, 0.3); border-color: #ffc107 !important; }
    .glow-high { box-shadow: 0 0 30px rgba(220, 53, 69, 0.5); border-color: #dc3545 !important; }
    .stMetric { background-color: #1a1c23; border: 1px solid #30363d; padding: 15px; border-radius: 12px; }
    h1, h2, h3 { color: #58a6ff; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Asset Loaders
# ──────────────────────────────────────────────────────────────────────────────
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=3)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_radar = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_m6cu9atn.json")

def render_premium_risk_card(prob, sn):
    percent = prob * 100
    if percent <= 20:
        color = "#10b981"  # Emerald Green
        status = "Rock Solid"
        glow = "rgba(16, 185, 129, 0.5)"
        cls = ""
    elif percent <= 60:
        color = "#f59e0b"  # Sunray Yellow
        status = "Cautionary"
        glow = "rgba(245, 158, 11, 0.5)"
        cls = ""
    else:
        color = "#ef4444"  # Vivid Crimson
        status = "Investigate Immediately"
        glow = "rgba(239, 68, 68, 0.5)"
        cls = "pulse-high"
    
    # Clock-style dial: map 0-100% across the top semicircle (180deg -> 0deg)
    p = max(0.0, min(1.0, float(prob)))
    angle_deg = 180 - (180 * p)
    angle_rad = np.deg2rad(angle_deg)
    cx, cy, r = 110, 120, 80
    x2 = cx + (r - 16) * np.cos(angle_rad)
    y2 = cy + (r - 16) * np.sin(angle_rad)
    
    html = f"""
    <div class="glass-card {cls}" style="border-top: 4px solid {color};">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
            <span style="color: {color}; font-weight: 800; font-size: 1.2em; text-transform: uppercase; letter-spacing: 2px;">Risk Analysis</span>
            <div class="tooltip">ⓘ
                <span class="tooltiptext">This score represents the probability of failure within the next 30 days based on your drive's temporal patterns.</span>
            </div>
        </div>
        <div style="display:flex; justify-content:center; margin: 8px 0 10px 0;">
            <svg viewBox="0 0 220 190" width="220" height="190">
                <path d="M 40 120 A 70 70 0 0 1 180 120" fill="none" stroke="#30363d" stroke-width="12" stroke-linecap="round"/>
                <line x1="{cx}" y1="{cy}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="6" stroke-linecap="round"
                      style="filter: drop-shadow(0 0 6px {glow});"/>
                <circle cx="{cx}" cy="{cy}" r="8" fill="{color}"/>
                <text x="110" y="168" text-anchor="middle" style="font-size: 2.0em; font-weight: 800; fill: #ffffff; font-family: 'Inter', sans-serif;">
                    {percent:.1f}%
                </text>
                <text x="40" y="138" text-anchor="middle" style="font-size: 0.8em; fill: #8b949e;">0</text>
                <text x="180" y="138" text-anchor="middle" style="font-size: 0.8em; fill: #8b949e;">100</text>
            </svg>
        </div>
        <div style="margin-top: 20px;">
            <p style="color: #8b949e; margin: 0; font-size: 0.9em;">Drive: {sn}</p>
            <h3 style="color: {color}; margin: 5px 0 0 0; font-size: 1.4em;">Status: {status}</h3>
        </div>
    </div>
    """
    return html

# ──────────────────────────────────────────────────────────────────────────────
# 3. Sidebar - Elite Control
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("💎 ELITE HUB")
if info and model is not None and scaler is not None:
    st.sidebar.success(f"Champion: {info['champion_model_name']}")
    input_data = {}
    for feat in info['features'][:6]:
        input_data[feat] = st.sidebar.number_input(f"{feat}", value=100.0)
    model_mode, expected_features, expected_window = _detect_model_mode(model)
    if model_mode == "tabular" and info.get("window_size", 1) > 1:
        st.sidebar.warning("Model metadata says sequence, but loaded model is tabular. Using tabular mode.")
    if model_mode == "sequence" and info.get("window_size", 1) <= 1:
        st.sidebar.warning("Model metadata says tabular, but loaded model is sequence. Using sequence mode.")

    if st.sidebar.button("Run Diagnostic"):
        with st.sidebar:
            if lottie_radar: st_lottie(lottie_radar, height=100)
            else: st.spinner("Analyzing...")
            time.sleep(1)

        target_n = _expected_feature_count(model, scaler, info)
        base_vec = np.array([list(input_data.values()) + [0.0] * (len(info['features']) - 6)], dtype=np.float32)
        base_vec = _align_feature_vector(base_vec, target_n)
        if model_mode == "sequence":
            win = expected_window or info.get("window_size", 7)
            seq = np.repeat(base_vec, win, axis=0)
            seq_scaled = scaler.transform(seq).reshape(1, win, -1)
            prob = float(model.predict(seq_scaled, verbose=0).flatten()[0])
        else:
            arr_scaled = scaler.transform(base_vec)
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(arr_scaled)[0][1])
            else:
                pred = model.predict(arr_scaled)
                prob = float(np.array(pred).flatten()[0])
        st.sidebar.metric("Risk Score", f"{prob*100:.1f}%")

# 4. Main Batch Analysis Section
st.title("🛡️ Predictive Maintenance Intelligence")

if info is None or model is None or scaler is None:
    st.warning("⚠️ Application is in 'Safe Mode' because model assets are missing. See error messages above.")
else:
    uploaded_file = st.file_uploader("Drop Fleet SMART Logs (.csv)", type=["csv"])

    if uploaded_file:
        st.info("📂 File uploaded successfully. Click the button below to start AI analysis.")
        
        # Explicit Process Button
        if st.button("🚀 Process Uploaded CSV", type="primary"):
            try:
                df = pl.read_csv(uploaded_file)
                if "date" in df.columns:
                    df = df.with_columns(pl.col("date").str.to_date(strict=False))
                if "serial_number" in df.columns and "date" in df.columns:
                    df = df.sort(["serial_number", "date"])
                df = _ensure_feature_columns(df, info["features"])
                with st.status("🧠 Analyzing SMART attributes...", expanded=True) as status:
                    model_mode, _, expected_window = _detect_model_mode(model)
                    if model_mode == "sequence":
                        win = expected_window or info.get("window_size", 7)
                        X_list, sns = [], []
                        for sn, group in df.group_by("serial_number"):
                            if group.height >= win:
                                data_raw = group.tail(win).select(info["features"]).to_numpy().astype("float32")
                                data_raw = _align_feature_vector(data_raw, _expected_feature_count(model, scaler, info))
                                data = scaler.transform(data_raw)
                                X_list.append(data)
                                sns.append(sn)
                        
                        if X_list:
                            preds = model.predict(np.array(X_list), verbose=0).flatten()
                            results = pd.DataFrame({"serial_number": sns, "failure_prob": preds})
                        else:
                            results = pd.DataFrame()
                    else:
                        if "serial_number" in df.columns:
                            if "date" in df.columns:
                                latest = df.group_by("serial_number").tail(1)
                            else:
                                latest = df.group_by("serial_number").head(1)
                        else:
                            latest = df

                        X_raw = latest.select(info["features"]).to_numpy().astype("float32")
                        X_raw = _align_feature_vector(X_raw, _expected_feature_count(model, scaler, info))
                        X = scaler.transform(X_raw)
                        if hasattr(model, "predict_proba"):
                            preds = model.predict_proba(X)[:, 1]
                        else:
                            preds = np.array(model.predict(X)).flatten()
                        sn_col = latest["serial_number"] if "serial_number" in latest.columns else np.arange(len(preds))
                        results = pd.DataFrame({"serial_number": sn_col, "failure_prob": preds})
                    
                    feat_np = df.select(info["features"]).to_numpy()
                    feat_std = float(np.nanmean(np.std(feat_np, axis=0))) if feat_np.size else 0.0
                    drive_count = int(df["serial_number"].n_unique()) if "serial_number" in df.columns else len(df)
                    st.caption(
                        f"Debug | rows={len(df):,} | drives={drive_count:,} | "
                        f"mode={model_mode} | mean_feature_std={feat_std:.4f}"
                    )

                    status.update(label="Analysis Complete!", state="complete")

                if not results.empty:
                    results = (
                        results.groupby("serial_number", as_index=False)["failure_prob"]
                        .max()
                        .sort_values("failure_prob", ascending=False)
                        .reset_index(drop=True)
                    )
                    top_drive = results.sort_values("failure_prob", ascending=False).iloc[0]
                    risk_val = top_drive['failure_prob']
                    
                    c1, c2 = st.columns([1.5, 2])
                    with c1:
                        st.markdown(render_premium_risk_card(risk_val, top_drive['serial_number']), unsafe_allow_html=True)
                    with c2:
                        msg = "🚨 CRITICAL ALERT" if risk_val > 0.6 else "⚠️ Precaution Advised" if risk_val > 0.2 else "✅ System Healthy"
                        st.markdown(f"<div style='padding-top:40px;'><h1 style='font-size:3.5em; line-height:1.1;'>{msg}</h1><p style='font-size:1.2em; color:#8b949e;'>Elite AI Pipeline has identified potential anomalies in the drive's temporal behavior.</p></div>", unsafe_allow_html=True)
                        if risk_val > 0.6: st.error("⚠️ CRITICAL: Immediate replacement recommended to prevent data loss.")
                        elif risk_val < 0.2: st.balloons()

                    st.success(f"Analyzed {len(results)} unique drives.")
                    st.dataframe(results.sort_values("failure_prob", ascending=False), use_container_width=True)
                else:
                    st.warning("No valid sequences found in the CSV for the current model requirements.")

            except Exception as e:
                st.error(f"Processing Error: {e}")
    else:
        st.write("Please upload a CSV file to begin.")

st.markdown("<br><center><p style='color: grey;'>💎 YTA Elite Intelligence Hub</p></center>", unsafe_allow_html=True)
