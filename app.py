"""
=============================================================================
 Predictive Retail Analytics Platform — Streamlit Application
=============================================================================
 A premium, multi-page web application for demand forecasting and
 inventory optimization powered by SARIMA time-series modeling.

 Author : Nishanth
 Stack  : Streamlit · Plotly · Pandas · NumPy · streamlit-option-menu
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
from typing import Tuple
import os

# =============================================================================
# APP CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Predictive Retail Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# COLOR PALETTE — cohesive design tokens used across the entire app
# =============================================================================
COLORS: dict[str, str] = {
    "bg_primary":    "#0a1628",
    "bg_card":       "#111d35",
    "bg_sidebar":    "#0d1f3c",
    "accent_teal":   "#00d4aa",
    "accent_blue":   "#4e7cff",
    "accent_amber":  "#ffb347",
    "accent_rose":   "#ff6b8a",
    "accent_purple": "#a78bfa",
    "text_primary":  "#e8ecf1",
    "text_muted":    "#8892a0",
    "border":        "#1e3054",
}

CHART_COLORS: list[str] = [
    "#4e7cff", "#00d4aa", "#ffb347", "#ff6b8a",
    "#a78bfa", "#38bdf8", "#f472b6", "#34d399",
    "#fb923c", "#818cf8", "#22d3ee", "#f87171",
]

# =============================================================================
# CUSTOM CSS — premium dark-mode styling with glassmorphism + animations
# =============================================================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ---------- Hide Streamlit Chrome ---------- */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    div[data-testid="stToolbar"] {display: none;}
    div[data-testid="stDecoration"] {display: none;}

    /* ---------- Global Overrides ---------- */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #0f2744 40%, #0b1a30 70%, #0a1628 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ---------- Animated Background Grain ---------- */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(ellipse at 20% 50%, rgba(78,124,255,0.04) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(0,212,170,0.03) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 80%, rgba(167,139,250,0.03) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1f3c 0%, #091428 100%);
        border-right: 1px solid #1e3054;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span {
        color: #e8ecf1 !important;
    }

    /* ---------- KPI Metric Cards ---------- */
    .kpi-card {
        background: linear-gradient(145deg, rgba(17, 29, 53, 0.9), rgba(10, 22, 40, 0.97));
        border: 1px solid rgba(30, 48, 84, 0.6);
        border-radius: 20px;
        padding: 28px 24px 24px;
        text-align: center;
        box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.3),
            0 2px 8px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255,255,255,0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow:
            0 16px 48px rgba(0, 0, 0, 0.4),
            0 4px 12px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255,255,255,0.08);
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 20px 20px 0 0;
    }
    .kpi-card::after {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle at center, rgba(255,255,255,0.02) 0%, transparent 60%);
        opacity: 0;
        transition: opacity 0.4s ease;
        pointer-events: none;
    }
    .kpi-card:hover::after { opacity: 1; }

    .kpi-card.teal::before   { background: linear-gradient(90deg, #00d4aa, #00b894, #00d4aa); background-size: 200% 100%; animation: shimmer 3s ease infinite; }
    .kpi-card.blue::before   { background: linear-gradient(90deg, #4e7cff, #3a5fcd, #4e7cff); background-size: 200% 100%; animation: shimmer 3s ease infinite; }
    .kpi-card.amber::before  { background: linear-gradient(90deg, #ffb347, #ff9f1c, #ffb347); background-size: 200% 100%; animation: shimmer 3s ease infinite; }
    .kpi-card.rose::before   { background: linear-gradient(90deg, #ff6b8a, #ee5a6f, #ff6b8a); background-size: 200% 100%; animation: shimmer 3s ease infinite; }
    .kpi-card.purple::before { background: linear-gradient(90deg, #a78bfa, #8b5cf6, #a78bfa); background-size: 200% 100%; animation: shimmer 3s ease infinite; }

    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    .kpi-icon {
        font-size: 1.8rem;
        margin-bottom: 10px;
        display: block;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #8892a0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
    }
    .kpi-value {
        font-size: 2.4rem;
        font-weight: 900;
        color: #e8ecf1;
        line-height: 1.05;
        margin-bottom: 8px;
        letter-spacing: -1px;
    }
    .kpi-delta {
        font-size: 0.78rem;
        font-weight: 500;
        color: #00d4aa;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 4px;
    }
    .kpi-delta.negative { color: #ff6b8a; }

    /* ---------- Section Headers ---------- */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #e8ecf1;
        margin: 36px 0 18px 0;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(30, 48, 84, 0.6);
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: -0.3px;
    }

    /* ---------- Page Headers ---------- */
    .page-header {
        animation: fadeInUp 0.6s ease-out;
        margin-bottom: 12px;
    }
    .page-header h1 {
        font-size: 2.2rem;
        font-weight: 900;
        color: #e8ecf1;
        margin-bottom: 4px;
        letter-spacing: -0.8px;
        background: linear-gradient(135deg, #e8ecf1, #8892a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .page-header p {
        font-size: 0.95rem;
        color: #8892a0;
        margin-top: 0;
        font-weight: 400;
    }
    .page-header .accent-bar {
        width: 60px;
        height: 4px;
        border-radius: 2px;
        background: linear-gradient(90deg, #4e7cff, #00d4aa);
        margin-top: 8px;
    }

    /* ---------- Chart Containers ---------- */
    .chart-container {
        background: linear-gradient(145deg, rgba(17, 29, 53, 0.7), rgba(10, 22, 40, 0.85));
        border: 1px solid rgba(30, 48, 84, 0.5);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
        margin-bottom: 24px;
        transition: box-shadow 0.3s ease;
    }
    .chart-container:hover {
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* ---------- Streamlit Element Overrides ---------- */
    .stSelectbox label, .stDateInput label, .stSlider label,
    .stCheckbox label, .stToggle label, .stMultiSelect label {
        color: #e8ecf1 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #e8ecf1 !important;
    }
    .stMarkdown p {
        color: #8892a0 !important;
    }

    /* ---------- DataFrame Styling ---------- */
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    /* ---------- Sidebar Logo / Title ---------- */
    .sidebar-brand {
        text-align: center;
        padding: 20px 16px 12px;
    }
    .sidebar-brand .logo-icon {
        font-size: 2.4rem;
        display: block;
        margin-bottom: 8px;
        filter: drop-shadow(0 4px 8px rgba(0,212,170,0.3));
    }
    .sidebar-brand .brand-name {
        font-size: 1.2rem;
        font-weight: 800;
        color: #e8ecf1;
        letter-spacing: -0.3px;
        line-height: 1.2;
    }
    .sidebar-brand .brand-tag {
        font-size: 0.68rem;
        font-weight: 400;
        color: #8892a0;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* ---------- Stat Pill (mini inline metric) ---------- */
    .stat-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 24px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(17, 29, 53, 0.8);
        border: 1px solid rgba(30, 48, 84, 0.5);
        color: #e8ecf1;
    }
    .stat-pill.live {
        animation: pulse 2s ease-in-out infinite;
    }
    .stat-pill .dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    .stat-pill .dot.green { background: #00d4aa; box-shadow: 0 0 6px rgba(0,212,170,0.5); }
    .stat-pill .dot.amber { background: #ffb347; box-shadow: 0 0 6px rgba(255,179,71,0.5); }

    /* ---------- Expander Styling ---------- */
    .stExpander {
        border: 1px solid rgba(30, 48, 84, 0.5) !important;
        border-radius: 16px !important;
        background: rgba(17, 29, 53, 0.5) !important;
    }

    /* ---------- Divider ---------- */
    .styled-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #1e3054, transparent);
        margin: 24px 0;
        border: none;
    }

    /* ---------- Info Callout ---------- */
    .info-callout {
        background: linear-gradient(135deg, rgba(78,124,255,0.08), rgba(0,212,170,0.05));
        border: 1px solid rgba(78,124,255,0.2);
        border-radius: 16px;
        padding: 16px 20px;
        font-size: 0.85rem;
        color: #8892a0;
        margin: 16px 0;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }
    .info-callout .callout-icon { font-size: 1.2rem; flex-shrink: 0; }

    /* ---------- Toggle Panel ---------- */
    .toggle-row {
        display: flex;
        gap: 24px;
        padding: 12px 0;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# DATA LOADING — cached helpers with real-data + placeholder comments
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


@st.cache_data(ttl=600)
def load_forecast_results() -> pd.DataFrame:
    """
    Load historical vs. predicted sales from the SARIMA forecast output.

    PLACEHOLDER: To swap with live model predictions, replace the CSV read
    with your inference pipeline:
        import pickle
        model = pickle.load(open("models/sarima_model.pkl", "rb"))
        forecast = model.get_forecast(steps=N)
        df = forecast.summary_frame()
    """
    path = os.path.join(OUTPUTS_DIR, "forecast_results.csv")
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data(ttl=600)
def load_inventory_metrics() -> pd.DataFrame:
    """
    Load per-family inventory metrics (safety stock, reorder point, EOQ).

    PLACEHOLDER: Replace with a SQL query to your inventory database:
        import sqlalchemy
        engine = sqlalchemy.create_engine(DATABASE_URL)
        df = pd.read_sql("SELECT * FROM inventory_metrics", engine)
    """
    path = os.path.join(OUTPUTS_DIR, "inventory_metrics.csv")
    return pd.read_csv(path)


@st.cache_data(ttl=600)
def load_inventory_simulation() -> pd.DataFrame:
    """
    Load inventory simulation data (stockout/overstock costs by family).

    PLACEHOLDER: Same as above — swap CSV read with your live data source.
    """
    path = os.path.join(OUTPUTS_DIR, "inventory_simulation.csv")
    return pd.read_csv(path)


def generate_mock_forecast(
    store_id: int,
    category: str,
    start_date: datetime,
    horizon_days: int,
    include_holidays: bool,
    impute_oil: bool,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate mock SARIMA-style forecast data with seasonal patterns.

    Returns three DataFrames:
      - df_hist: 90 days of historical demand
      - df_forecast: forecasted demand with confidence intervals
      - df_actual: simulated "actual" demand (ground truth with different noise)

    The forecast and actual share the same underlying signal but have
    different noise seeds, simulating how a real model's predictions
    would differ from actual observed values.

    PLACEHOLDER: To swap with your real SARIMA model:
        import pickle
        model = pickle.load(open("models/sarima_model.pkl", "rb"))
        exog = build_exogenous_features(store_id, category, include_holidays, impute_oil)
        forecast_obj = model.get_forecast(steps=horizon_days, exog=exog)
        result = forecast_obj.summary_frame(alpha=0.05)
        # result columns: mean, mean_se, mean_ci_lower, mean_ci_upper
        # For actual demand, query from your sales database:
        # df_actual = pd.read_sql("SELECT date, sales FROM daily_sales WHERE ...", engine)
    """
    rng = np.random.default_rng(seed + store_id + hash(category) % 1000)

    hist_days = 90
    total_days = hist_days + horizon_days
    dates = pd.date_range(
        start=start_date - timedelta(days=hist_days),
        periods=total_days,
        freq="D",
    )

    base = 5000 + (hash(category) % 10) * 1200 + store_id * 50
    trend = np.linspace(0, 300, total_days)
    weekly = 800 * np.sin(2 * np.pi * np.arange(total_days) / 7)
    monthly = 400 * np.sin(2 * np.pi * np.arange(total_days) / 30.44)

    holiday_bump = np.zeros(total_days)
    if include_holidays:
        holiday_indices = rng.choice(
            total_days, size=max(1, total_days // 20), replace=False
        )
        holiday_bump[holiday_indices] = rng.uniform(
            1500, 4000, size=len(holiday_indices)
        )

    oil_effect = np.zeros(total_days)
    if impute_oil:
        oil_effect = -200 * np.sin(2 * np.pi * np.arange(total_days) / 60)

    # Shared signal (trend + seasonality + external factors)
    signal = base + trend + weekly + monthly + holiday_bump + oil_effect

    # Forecast uses one noise seed (model's prediction)
    forecast_noise = rng.normal(0, 350, total_days)
    forecast_values = np.maximum(signal + forecast_noise, 100)

    # Actual demand uses a different noise seed (real-world variation)
    rng_actual = np.random.default_rng(seed + store_id + hash(category) % 1000 + 9999)
    actual_noise = rng_actual.normal(0, 400, total_days)
    actual_values = np.maximum(signal + actual_noise, 100)

    historical = actual_values[:hist_days]
    forecast = forecast_values[hist_days:]
    actual_future = actual_values[hist_days:]

    ci_expansion = np.linspace(0.5, 2.5, horizon_days)
    std_base = float(np.std(historical)) * 0.6
    lower_bound = forecast - std_base * ci_expansion * 1.96
    upper_bound = forecast + std_base * ci_expansion * 1.96

    df_hist = pd.DataFrame({
        "date": dates[:hist_days],
        "value": historical,
        "type": "Historical",
    })
    df_forecast = pd.DataFrame({
        "date": dates[hist_days:],
        "value": forecast,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "type": "Forecast",
    })
    df_actual = pd.DataFrame({
        "date": dates[hist_days:],
        "value": actual_future,
        "type": "Actual",
    })

    return df_hist, df_forecast, df_actual


def generate_reorder_table(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Reorder Recommendation Engine table from inventory metrics.

    Adds simulated current stock levels and computes an action recommendation.

    PLACEHOLDER: Replace current_stock with live warehouse data:
        stock_df = pd.read_sql("SELECT sku, on_hand_qty FROM warehouse", engine)
        df = inventory_df.merge(stock_df, on="sku")
    """
    rng = np.random.default_rng(2024)
    df = inventory_df.copy()

    df["sku"] = [f"SKU-{1000 + i:04d}" for i in range(len(df))]
    df["current_stock"] = (
        df["safety_stock"] * rng.uniform(0.15, 1.8, len(df))
    ).astype(int)
    df["forecast_30d_demand"] = (df["avg_daily_demand"] * 30).astype(int)

    def get_action(row: pd.Series) -> str:
        ratio = (
            row["current_stock"] / row["reorder_point"]
            if row["reorder_point"] > 0
            else 1.0
        )
        if ratio < 0.4:
            return "🔴 URGENT REORDER"
        elif ratio < 0.85:
            return "🟡 REORDER"
        else:
            return "🟢 HOLD"

    df["action"] = df.apply(get_action, axis=1)

    display_cols = [
        "sku", "family", "current_stock", "forecast_30d_demand",
        "safety_stock", "reorder_point", "eoq", "action",
    ]
    return df[display_cols]


# =============================================================================
# PLOTLY CHART DEFAULTS — consistent dark theme across all charts
# =============================================================================
PLOTLY_LAYOUT: dict = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["text_primary"], size=13),
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,0,0,0)",
        font=dict(size=12),
    ),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.05)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.05)",
    ),
)


def apply_chart_layout(
    fig: go.Figure, title: str = "", height: int = 480
) -> go.Figure:
    """Apply the standard dark layout to any Plotly figure."""
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=title,
            font=dict(size=17, color=COLORS["text_primary"], family="Inter"),
            x=0.01,
        ),
        height=height,
    )
    return fig


# =============================================================================
# HELPER — render a KPI card with icon
# =============================================================================
def kpi_card(
    label: str,
    value: str,
    delta: str = "",
    accent: str = "teal",
    icon: str = "",
) -> str:
    """Return HTML for a single KPI card with optional icon."""
    delta_class = (
        "negative"
        if delta.startswith("-") or "risk" in label.lower()
        else ""
    )
    delta_html = (
        f'<div class="kpi-delta {delta_class}">▸ {delta}</div>'
        if delta
        else ""
    )
    icon_html = (
        f'<span class="kpi-icon">{icon}</span>'
        if icon
        else ""
    )
    return f"""
    <div class="kpi-card {accent}">
        {icon_html}
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


def render_page_header(title: str, subtitle: str) -> None:
    """Render a styled page header with gradient text and accent bar."""
    st.markdown(
        f"""
        <div class="page-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <div class="accent-bar"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_divider() -> None:
    """Render a styled gradient divider."""
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)


def spacer(px: int = 24) -> None:
    """Render vertical spacing."""
    st.markdown(
        f"<div style='height: {px}px;'></div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <span class="logo-icon">📊</span>
            <div class="brand-name">Retail Analytics</div>
            <div class="brand-tag">Predictive Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    selected_page = option_menu(
        menu_title=None,
        options=[
            "Executive Dashboard",
            "Demand Forecasting",
            "Inventory Optimization",
        ],
        icons=["speedometer2", "graph-up-arrow", "box-seam"],
        default_index=0,
        styles={
            "container": {
                "padding": "0 !important",
                "background-color": "transparent",
            },
            "icon": {
                "color": COLORS["accent_teal"],
                "font-size": "17px",
            },
            "nav-link": {
                "font-size": "13.5px",
                "font-weight": "500",
                "color": COLORS["text_muted"],
                "text-align": "left",
                "padding": "12px 16px",
                "border-radius": "12px",
                "margin": "4px 0",
                "--hover-color": COLORS["bg_card"],
            },
            "nav-link-selected": {
                "background": (
                    f"linear-gradient(135deg, "
                    f"{COLORS['accent_blue']}22, "
                    f"{COLORS['accent_teal']}18)"
                ),
                "color": COLORS["text_primary"],
                "font-weight": "600",
                "border": f"1px solid {COLORS['border']}",
            },
        },
    )

    st.markdown("---")

    # Sidebar status pills
    st.markdown(
        f"""
        <div style="padding: 8px 12px;">
            <div class="stat-pill live" style="margin-bottom: 8px; width: 100%; justify-content: center;">
                <span class="dot green"></span>
                SARIMA v1.0 · MAPE 7.9%
            </div>
            <div class="stat-pill" style="width: 100%; justify-content: center;">
                <span class="dot amber"></span>
                Last sync: Aug 2026
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE 1 — EXECUTIVE DASHBOARD
# =============================================================================
if selected_page == "Executive Dashboard":

    render_page_header(
        "Executive Dashboard",
        "Real-time overview of demand forecasting performance and inventory health",
    )

    # --- Load data ---
    inv_metrics = load_inventory_metrics()
    inv_sim = load_inventory_simulation()
    forecast_df = load_forecast_results()

    total_safety_stock = float(inv_metrics["safety_stock"].sum())
    total_risk = float(inv_sim["total_inventory_cost"].sum())
    avg_daily_demand = float(inv_metrics["avg_daily_demand"].sum())
    total_families = len(inv_metrics)

    # --- KPI Row ---
    kpi_cols = st.columns(4)
    kpis = [
        ("Overall MAPE", "7.9%", "Model Accuracy", "teal", "🎯"),
        (
            "Total Safety Stock",
            f"{total_safety_stock / 1e6:.2f}M",
            "Units recommended",
            "blue",
            "📦",
        ),
        (
            "Inventory Risk",
            f"${total_risk / 1e9:.2f}B",
            "Identified exposure",
            "rose",
            "⚠️",
        ),
        (
            "Avg Daily Demand",
            f"{avg_daily_demand / 1e3:.0f}K",
            f"Across {total_families} families",
            "amber",
            "📈",
        ),
    ]
    for col, (label, value, delta, accent, icon) in zip(kpi_cols, kpis):
        col.markdown(
            kpi_card(label, value, delta, accent, icon),
            unsafe_allow_html=True,
        )

    spacer(28)

    # --- Charts Row ---
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        st.markdown(
            '<div class="section-header">📈 Historical vs. Predicted Sales</div>',
            unsafe_allow_html=True,
        )

        fig_sales = go.Figure()

        # Confidence interval band
        fig_sales.add_trace(go.Scatter(
            x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
            y=pd.concat([
                forecast_df["upper_bound"],
                forecast_df["lower_bound"][::-1],
            ]),
            fill="toself",
            fillcolor="rgba(78, 124, 255, 0.08)",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=True,
            name="95% Confidence Interval",
        ))

        # Actual sales
        fig_sales.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["actual_sales"],
            mode="lines",
            name="Actual Sales",
            line=dict(color=COLORS["accent_teal"], width=2.5),
            hovertemplate="Date: %{x}<br>Actual: %{y:,.0f}<extra></extra>",
        ))

        # Forecasted sales
        fig_sales.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["forecasted_sales"],
            mode="lines",
            name="SARIMA Forecast",
            line=dict(
                color=COLORS["accent_blue"], width=2.5, dash="dot"
            ),
            hovertemplate=(
                "Date: %{x}<br>Forecast: %{y:,.0f}<extra></extra>"
            ),
        ))

        fig_sales = apply_chart_layout(fig_sales, "", height=440)
        fig_sales.update_layout(
            xaxis_title="Date",
            yaxis_title="Total Sales (Units)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_sales, config={"displayModeBar": False})

    with chart_col2:
        st.markdown(
            '<div class="section-header">🍩 Inventory Risk by Category</div>',
            unsafe_allow_html=True,
        )

        sim_sorted = inv_sim.sort_values(
            "total_inventory_cost", ascending=False
        )
        top_n = 8
        top_cats = sim_sorted.head(top_n).copy()
        others_cost = float(
            sim_sorted.iloc[top_n:]["total_inventory_cost"].sum()
        )
        if others_cost > 0:
            others_row = pd.DataFrame(
                [{"family": "Others", "total_inventory_cost": others_cost}]
            )
            top_cats = pd.concat([top_cats, others_row], ignore_index=True)

        fig_donut = go.Figure(data=[go.Pie(
            labels=top_cats["family"],
            values=top_cats["total_inventory_cost"],
            hole=0.6,
            marker=dict(
                colors=CHART_COLORS[: len(top_cats)],
                line=dict(color=COLORS["bg_primary"], width=2),
            ),
            textinfo="label+percent",
            textposition="outside",
            textfont=dict(size=10.5),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Risk: $%{value:,.0f}<br>"
                "Share: %{percent}<extra></extra>"
            ),
        )])

        fig_donut = apply_chart_layout(fig_donut, "", height=440)
        fig_donut.update_layout(
            showlegend=False,
            annotations=[dict(
                text=(
                    f"<b>${total_risk/1e9:.1f}B</b><br>"
                    f"<span style='font-size:10px;"
                    f"color:{COLORS['text_muted']}'>Total Risk</span>"
                ),
                x=0.5,
                y=0.5,
                font=dict(size=20, color=COLORS["text_primary"]),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_donut, config={"displayModeBar": False})

    render_divider()

    # --- Forecast Error Distribution ---
    st.markdown(
        '<div class="section-header">📊 Forecast Error Distribution</div>',
        unsafe_allow_html=True,
    )

    # Info callout
    st.markdown(
        """
        <div class="info-callout">
            <span class="callout-icon">💡</span>
            <span>Bars are color-coded by severity:
            <b style="color:#00d4aa">green</b> (&lt;5%),
            <b style="color:#ffb347">amber</b> (5–10%),
            <b style="color:#ff6b8a">rose</b> (&gt;10%).
            The dashed line marks the overall MAPE threshold.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_error = go.Figure()
    error_colors = [
        COLORS["accent_rose"] if abs(x) > 10
        else (COLORS["accent_amber"] if abs(x) > 5 else COLORS["accent_teal"])
        for x in forecast_df["error_pct"]
    ]
    fig_error.add_trace(go.Bar(
        x=forecast_df["date"],
        y=forecast_df["error_pct"],
        marker=dict(color=error_colors, line=dict(width=0)),
        hovertemplate="Date: %{x}<br>Error: %{y:.1f}%<extra></extra>",
    ))
    fig_error = apply_chart_layout(fig_error, "", height=300)
    fig_error.update_layout(
        xaxis_title="Date",
        yaxis_title="Error (%)",
        showlegend=False,
    )
    fig_error.add_hline(
        y=7.9,
        line_dash="dash",
        line_color=COLORS["accent_amber"],
        annotation_text="MAPE: 7.9%",
        annotation_font_color=COLORS["accent_amber"],
        annotation_font_size=12,
    )
    st.plotly_chart(fig_error, config={"displayModeBar": False})


# =============================================================================
# PAGE 2 — DEMAND FORECASTING (SARIMA INTERFACE)
# =============================================================================
elif selected_page == "Demand Forecasting":

    render_page_header(
        "Demand Forecasting Engine",
        "Configure SARIMA model parameters to generate demand forecasts "
        "with confidence intervals",
    )

    # --- Input Panel ---
    st.markdown(
        '<div class="section-header">⚙️ Forecast Configuration</div>',
        unsafe_allow_html=True,
    )

    inv_metrics = load_inventory_metrics()
    categories = inv_metrics["family"].tolist()

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        store_id = st.selectbox(
            "Store ID",
            options=list(range(1, 55)),
            index=0,
            help="Select a Favorita store (1–54)",
        )

    with col_b:
        category = st.selectbox(
            "Product Category",
            options=categories,
            index=0,
            help="Select a product family for forecasting",
        )

    with col_c:
        forecast_start = st.date_input(
            "Forecast Start Date",
            value=datetime(2017, 8, 16),
            help="Select the date to begin the forecast from",
        )

    with col_d:
        horizon = st.slider(
            "Forecast Horizon (Days)",
            min_value=7,
            max_value=90,
            value=30,
            step=1,
            help="Number of days to forecast ahead",
        )

    # Toggle options
    tog_col1, tog_col2, _, _ = st.columns(4)
    with tog_col1:
        include_holidays = st.toggle(
            "Include Holiday / Promotion Flags", value=True
        )
    with tog_col2:
        impute_oil = st.toggle("Impute Weekend Oil Prices", value=False)

    spacer(16)

    # --- Generate Forecast ---
    df_hist, df_fc, df_actual = generate_mock_forecast(
        store_id=store_id,
        category=category,
        start_date=datetime.combine(forecast_start, datetime.min.time()),
        horizon_days=horizon,
        include_holidays=include_holidays,
        impute_oil=impute_oil,
    )

    # --- KPI Summary for this forecast ---
    avg_forecast = float(df_fc["value"].mean())
    avg_actual = float(df_actual["value"].mean())
    peak_demand = float(df_actual["value"].max())
    avg_historical = float(df_hist["value"].mean())
    change_pct = ((avg_forecast - avg_historical) / avg_historical) * 100
    # Compute forecast MAPE vs simulated actuals
    fc_mape = float(
        np.mean(np.abs(df_fc["value"].values - df_actual["value"].values)
                / np.maximum(df_actual["value"].values, 1)) * 100
    )

    kpi_fc_cols = st.columns(5)
    fc_kpis = [
        (
            "Forecast MAPE",
            f"{fc_mape:.1f}%",
            f"across {horizon} days",
            "teal" if fc_mape < 10 else "amber",
            "🎯",
        ),
        (
            "Avg Forecasted Demand",
            f"{avg_forecast:,.0f}",
            "units/day",
            "blue",
            "📊",
        ),
        (
            "Avg Actual Demand",
            f"{avg_actual:,.0f}",
            "units/day",
            "purple",
            "📈",
        ),
        (
            "Peak Actual Demand",
            f"{peak_demand:,.0f}",
            f"within {horizon}d window",
            "amber",
            "🔺",
        ),
        (
            "Demand Shift",
            f"{change_pct:+.1f}%",
            "vs. historical avg",
            "rose" if change_pct < 0 else "teal",
            "🔄",
        ),
    ]
    for col, (lbl, val, dlt, acc, ico) in zip(kpi_fc_cols, fc_kpis):
        col.markdown(
            kpi_card(lbl, val, dlt, acc, ico),
            unsafe_allow_html=True,
        )

    spacer(20)

    # --- Forecast Chart ---
    st.markdown(
        '<div class="section-header">'
        "📈 SARIMA Forecast with Confidence Intervals"
        "</div>",
        unsafe_allow_html=True,
    )

    fig_fc = go.Figure()

    # Confidence interval band
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([df_fc["date"], df_fc["date"][::-1]]),
        y=pd.concat([df_fc["upper_bound"], df_fc["lower_bound"][::-1]]),
        fill="toself",
        fillcolor="rgba(0, 212, 170, 0.1)",
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=True,
        name="95% Confidence Interval",
    ))

    # Historical line
    fig_fc.add_trace(go.Scatter(
        x=df_hist["date"],
        y=df_hist["value"],
        mode="lines",
        name="Historical Demand",
        line=dict(color=COLORS["accent_blue"], width=2),
        hovertemplate=(
            "Date: %{x|%b %d, %Y}<br>"
            "Demand: %{y:,.0f}<extra></extra>"
        ),
    ))

    # Forecast line
    fig_fc.add_trace(go.Scatter(
        x=df_fc["date"],
        y=df_fc["value"],
        mode="lines",
        name="SARIMA Forecast",
        line=dict(color=COLORS["accent_teal"], width=2.5),
        hovertemplate=(
            "Date: %{x|%b %d, %Y}<br>"
            "Forecast: %{y:,.0f}<extra></extra>"
        ),
    ))

    # Actual demand line (simulated ground truth)
    fig_fc.add_trace(go.Scatter(
        x=df_actual["date"],
        y=df_actual["value"],
        mode="lines",
        name="Actual Demand",
        line=dict(color=COLORS["accent_rose"], width=2, dash="dot"),
        hovertemplate=(
            "Date: %{x|%b %d, %Y}<br>"
            "Actual: %{y:,.0f}<extra></extra>"
        ),
    ))

    # Vertical line separating historical and forecast
    fig_fc.add_vline(
        x=df_fc["date"].iloc[0].timestamp() * 1000,
        line_dash="dash",
        line_color=COLORS["accent_amber"],
        annotation_text="Forecast Start",
        annotation_font_color=COLORS["accent_amber"],
        annotation_font_size=12,
        annotation_position="top",
    )

    fig_fc = apply_chart_layout(
        fig_fc,
        f"Store #{store_id} · {category} · {horizon}-Day Forecast",
        height=500,
    )
    fig_fc.update_layout(
        xaxis_title="Date",
        yaxis_title="Demand (Units)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_fc, config={"displayModeBar": False})

    # --- Forecast Data Table (collapsible) ---
    with st.expander("📋 View Raw Forecast Data", expanded=False):
        display_fc = df_fc[["date", "value", "lower_bound", "upper_bound"]].copy()
        display_fc["actual_demand"] = df_actual["value"].values
        display_fc["error_pct"] = (
            np.abs(display_fc["value"] - display_fc["actual_demand"])
            / np.maximum(display_fc["actual_demand"], 1) * 100
        ).round(1)
        display_fc.columns = [
            "Date",
            "Forecasted Demand",
            "Lower Bound (95%)",
            "Upper Bound (95%)",
            "Actual Demand",
            "Error %",
        ]
        display_fc["Date"] = display_fc["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            display_fc,
            hide_index=True,
            column_config={
                "Forecasted Demand": st.column_config.NumberColumn(
                    format="%,.0f"
                ),
                "Lower Bound (95%)": st.column_config.NumberColumn(
                    format="%,.0f"
                ),
                "Upper Bound (95%)": st.column_config.NumberColumn(
                    format="%,.0f"
                ),
                "Actual Demand": st.column_config.NumberColumn(
                    format="%,.0f"
                ),
                "Error %": st.column_config.NumberColumn(
                    format="%.1f%%"
                ),
            },
        )


# =============================================================================
# PAGE 3 — INVENTORY OPTIMIZATION
# =============================================================================
elif selected_page == "Inventory Optimization":

    render_page_header(
        "Inventory Optimization Engine",
        "AI-driven reorder recommendations based on forecasted demand "
        "and safety stock analysis",
    )

    # Load and build reorder table
    inv_metrics = load_inventory_metrics()
    reorder_df = generate_reorder_table(inv_metrics)

    # --- Summary KPIs ---
    urgent_count = int((reorder_df["action"] == "🔴 URGENT REORDER").sum())
    reorder_count = int((reorder_df["action"] == "🟡 REORDER").sum())
    hold_count = int((reorder_df["action"] == "🟢 HOLD").sum())
    total_reorder_value = int(
        reorder_df.loc[
            reorder_df["action"].isin(["🔴 URGENT REORDER", "🟡 REORDER"]),
            "eoq",
        ].sum()
    )

    kpi_inv_cols = st.columns(4)
    inv_kpis = [
        (
            "Urgent Reorders",
            str(urgent_count),
            "Critically low SKUs",
            "rose",
            "🚨",
        ),
        (
            "Standard Reorders",
            str(reorder_count),
            "Below reorder point",
            "amber",
            "🔔",
        ),
        (
            "Stock Healthy",
            str(hold_count),
            "No action needed",
            "teal",
            "✅",
        ),
        (
            "Total Reorder Volume",
            f"{total_reorder_value:,}",
            "EOQ units to order",
            "blue",
            "📋",
        ),
    ]
    for col, (label, value, delta, accent, icon) in zip(
        kpi_inv_cols, inv_kpis
    ):
        col.markdown(
            kpi_card(label, value, delta, accent, icon),
            unsafe_allow_html=True,
        )

    spacer(28)

    # --- Reorder Recommendation Table ---
    st.markdown(
        '<div class="section-header">📦 Reorder Recommendation Engine</div>',
        unsafe_allow_html=True,
    )

    # Info callout
    st.markdown(
        """
        <div class="info-callout">
            <span class="callout-icon">📌</span>
            <span>Recommendations are based on current stock levels relative
            to the computed reorder point.
            <b style="color:#ff6b8a">URGENT</b> = stock below 40% of
            reorder point,
            <b style="color:#ffb347">REORDER</b> = below 85%,
            <b style="color:#00d4aa">HOLD</b> = above threshold.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Filters
    filter_col1, filter_col2, _ = st.columns([1, 1, 2])
    with filter_col1:
        action_filter = st.multiselect(
            "Filter by Action",
            options=["🔴 URGENT REORDER", "🟡 REORDER", "🟢 HOLD"],
            default=["🔴 URGENT REORDER", "🟡 REORDER", "🟢 HOLD"],
        )
    with filter_col2:
        sort_by = st.selectbox(
            "Sort by",
            options=[
                "action",
                "current_stock",
                "forecast_30d_demand",
                "safety_stock",
            ],
            index=0,
        )

    filtered_df = reorder_df[
        reorder_df["action"].isin(action_filter)
    ].sort_values(
        sort_by,
        ascending=(sort_by != "forecast_30d_demand"),
    )

    st.dataframe(
        filtered_df,
        hide_index=True,
        height=500,
        column_config={
            "sku": st.column_config.TextColumn("SKU", width="small"),
            "family": st.column_config.TextColumn(
                "Product Family", width="medium"
            ),
            "current_stock": st.column_config.NumberColumn(
                "Current Stock",
                format="%,d",
                help="Simulated current on-hand inventory",
            ),
            "forecast_30d_demand": st.column_config.NumberColumn(
                "30-Day Demand",
                format="%,d",
                help="Forecasted demand over next 30 days",
            ),
            "safety_stock": st.column_config.NumberColumn(
                "Safety Stock",
                format="%,d",
                help="Recommended buffer stock",
            ),
            "reorder_point": st.column_config.NumberColumn(
                "Reorder Point",
                format="%,d",
                help="Threshold to trigger reorder",
            ),
            "eoq": st.column_config.NumberColumn(
                "EOQ",
                format="%,d",
                help="Economic Order Quantity",
            ),
            "action": st.column_config.TextColumn(
                "Action", width="medium"
            ),
        },
    )

    spacer(28)
    render_divider()

    # --- Visual Breakdown ---
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown(
            '<div class="section-header">'
            "📊 Stock vs. Reorder Point"
            "</div>",
            unsafe_allow_html=True,
        )

        fig_bar = go.Figure()

        fig_bar.add_trace(go.Bar(
            x=reorder_df["family"],
            y=reorder_df["current_stock"],
            name="Current Stock",
            marker_color=COLORS["accent_blue"],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Current Stock: %{y:,.0f}<extra></extra>"
            ),
        ))
        fig_bar.add_trace(go.Bar(
            x=reorder_df["family"],
            y=reorder_df["reorder_point"],
            name="Reorder Point",
            marker_color=COLORS["accent_rose"],
            opacity=0.7,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Reorder Point: %{y:,.0f}<extra></extra>"
            ),
        ))

        fig_bar = apply_chart_layout(fig_bar, "", height=420)
        fig_bar.update_layout(
            barmode="group",
            xaxis_title="Product Family",
            yaxis_title="Units",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_bar, config={"displayModeBar": False})

    with viz_col2:
        st.markdown(
            '<div class="section-header">🎯 Action Distribution</div>',
            unsafe_allow_html=True,
        )

        action_counts = reorder_df["action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Count"]

        action_color_map = {
            "🔴 URGENT REORDER": COLORS["accent_rose"],
            "🟡 REORDER": COLORS["accent_amber"],
            "🟢 HOLD": COLORS["accent_teal"],
        }

        fig_action = go.Figure(data=[go.Pie(
            labels=action_counts["Action"],
            values=action_counts["Count"],
            hole=0.55,
            marker=dict(
                colors=[
                    action_color_map.get(a, "#888")
                    for a in action_counts["Action"]
                ],
                line=dict(color=COLORS["bg_primary"], width=3),
            ),
            textinfo="label+value",
            textposition="outside",
            textfont=dict(size=11.5),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Count: %{value}<br>"
                "Share: %{percent}<extra></extra>"
            ),
        )])

        fig_action = apply_chart_layout(fig_action, "", height=420)
        fig_action.update_layout(
            showlegend=False,
            annotations=[dict(
                text=(
                    f"<b>{len(reorder_df)}</b><br>"
                    f"<span style='font-size:10px;"
                    f"color:{COLORS['text_muted']}'>Total SKUs</span>"
                ),
                x=0.5,
                y=0.5,
                font=dict(size=18, color=COLORS["text_primary"]),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_action, config={"displayModeBar": False})
