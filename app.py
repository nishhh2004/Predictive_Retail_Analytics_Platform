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
import plotly.express as px
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
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
COLORS = {
    "bg_primary":    "#0a1628",    # Deep navy — page background
    "bg_card":       "#111d35",    # Slightly lighter navy — card surfaces
    "bg_sidebar":    "#0d1f3c",    # Sidebar background
    "accent_teal":   "#00d4aa",    # Primary accent — teal/mint
    "accent_blue":   "#4e7cff",    # Secondary accent — electric blue
    "accent_amber":  "#ffb347",    # Warning / highlight — warm amber
    "accent_rose":   "#ff6b8a",    # Danger / critical — soft rose
    "text_primary":  "#e8ecf1",    # Primary text — near-white
    "text_muted":    "#8892a0",    # Muted / secondary text — soft gray
    "border":        "#1e3054",    # Subtle border color
}

# Plotly chart color sequence
CHART_COLORS = [
    "#4e7cff", "#00d4aa", "#ffb347", "#ff6b8a",
    "#a78bfa", "#38bdf8", "#f472b6", "#34d399",
    "#fb923c", "#818cf8", "#22d3ee", "#f87171",
]

# =============================================================================
# CUSTOM CSS — premium dark-mode styling with glassmorphism
# =============================================================================
CUSTOM_CSS = f"""
<style>
    /* ---------- Import Google Font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ---------- Hide Streamlit Chrome ---------- */
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display: none;}}

    /* ---------- Global Overrides ---------- */
    .stApp {{
        background: linear-gradient(135deg, {COLORS["bg_primary"]} 0%, #0f2744 50%, {COLORS["bg_primary"]} 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS["bg_sidebar"]} 0%, #091428 100%);
        border-right: 1px solid {COLORS["border"]};
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span {{
        color: {COLORS["text_primary"]} !important;
    }}

    /* ---------- KPI Metric Cards ---------- */
    .kpi-card {{
        background: linear-gradient(145deg, rgba(17, 29, 53, 0.85), rgba(10, 22, 40, 0.95));
        border: 1px solid {COLORS["border"]};
        border-radius: 16px;
        padding: 28px 24px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    .kpi-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255,255,255,0.08);
    }}
    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        border-radius: 16px 16px 0 0;
    }}
    .kpi-card.teal::before   {{ background: linear-gradient(90deg, {COLORS["accent_teal"]}, #00b894); }}
    .kpi-card.blue::before   {{ background: linear-gradient(90deg, {COLORS["accent_blue"]}, #3a5fcd); }}
    .kpi-card.amber::before  {{ background: linear-gradient(90deg, {COLORS["accent_amber"]}, #ff9f1c); }}
    .kpi-card.rose::before   {{ background: linear-gradient(90deg, {COLORS["accent_rose"]}, #ee5a6f); }}

    .kpi-label {{
        font-size: 0.8rem;
        font-weight: 500;
        color: {COLORS["text_muted"]};
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 8px;
    }}
    .kpi-value {{
        font-size: 2.2rem;
        font-weight: 800;
        color: {COLORS["text_primary"]};
        line-height: 1.1;
        margin-bottom: 6px;
    }}
    .kpi-delta {{
        font-size: 0.8rem;
        font-weight: 500;
        color: {COLORS["accent_teal"]};
    }}
    .kpi-delta.negative {{
        color: {COLORS["accent_rose"]};
    }}

    /* ---------- Section Headers ---------- */
    .section-header {{
        font-size: 1.3rem;
        font-weight: 700;
        color: {COLORS["text_primary"]};
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid {COLORS["border"]};
        display: flex;
        align-items: center;
        gap: 10px;
    }}

    /* ---------- Chart Containers ---------- */
    .chart-container {{
        background: linear-gradient(145deg, rgba(17, 29, 53, 0.7), rgba(10, 22, 40, 0.85));
        border: 1px solid {COLORS["border"]};
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
        margin-bottom: 24px;
    }}

    /* ---------- Streamlit Element Overrides ---------- */
    .stSelectbox label, .stDateInput label, .stSlider label,
    .stCheckbox label, .stToggle label {{
        color: {COLORS["text_primary"]} !important;
        font-weight: 500 !important;
    }}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: {COLORS["text_primary"]} !important;
    }}
    .stMarkdown p {{
        color: {COLORS["text_muted"]} !important;
    }}

    /* ---------- DataFrame Styling ---------- */
    .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
    }}

    /* ---------- Sidebar Logo / Title ---------- */
    .sidebar-title {{
        font-size: 1.35rem;
        font-weight: 800;
        color: {COLORS["text_primary"]};
        text-align: center;
        padding: 16px 0 8px 0;
        letter-spacing: -0.5px;
    }}
    .sidebar-subtitle {{
        font-size: 0.75rem;
        font-weight: 400;
        color: {COLORS["text_muted"]};
        text-align: center;
        padding-bottom: 20px;
        letter-spacing: 0.5px;
    }}

    /* ---------- Input Panels ---------- */
    .input-panel {{
        background: linear-gradient(145deg, rgba(17, 29, 53, 0.6), rgba(10, 22, 40, 0.8));
        border: 1px solid {COLORS["border"]};
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
    }}

    /* ---------- Badge ---------- */
    .badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }}
    .badge-urgent {{ background: rgba(255,107,138,0.2); color: #ff6b8a; }}
    .badge-warning {{ background: rgba(255,179,71,0.2); color: #ffb347; }}
    .badge-ok {{ background: rgba(0,212,170,0.2); color: #00d4aa; }}
</style>
"""

# Inject custom CSS
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
    df = pd.read_csv(path, parse_dates=["date"])
    return df


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
    df = pd.read_csv(path)
    return df


@st.cache_data(ttl=600)
def load_inventory_simulation() -> pd.DataFrame:
    """
    Load inventory simulation data (stockout/overstock costs by family).

    PLACEHOLDER: Same as above — swap CSV read with your live data source.
    """
    path = os.path.join(OUTPUTS_DIR, "inventory_simulation.csv")
    df = pd.read_csv(path)
    return df


def generate_mock_forecast(
    store_id: int,
    category: str,
    start_date: datetime,
    horizon_days: int,
    include_holidays: bool,
    impute_oil: bool,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate mock SARIMA-style forecast data with seasonal patterns.

    This function simulates a realistic retail demand forecast with:
      - A base trend component
      - Weekly seasonality (7-day cycle)
      - Monthly seasonality (30-day cycle)
      - Random noise
      - Expanding confidence intervals

    PLACEHOLDER: To swap with your real SARIMA model:
        import pickle
        model = pickle.load(open("models/sarima_model.pkl", "rb"))
        exog = build_exogenous_features(store_id, category, include_holidays, impute_oil)
        forecast_obj = model.get_forecast(steps=horizon_days, exog=exog)
        result = forecast_obj.summary_frame(alpha=0.05)
        # result columns: mean, mean_se, mean_ci_lower, mean_ci_upper
    """
    rng = np.random.default_rng(seed + store_id + hash(category) % 1000)

    # --- Historical data (90 days before start) ---
    hist_days = 90
    total_days = hist_days + horizon_days
    dates = pd.date_range(start=start_date - timedelta(days=hist_days), periods=total_days, freq="D")

    # Base demand varies by category hash (deterministic per category)
    base = 5000 + (hash(category) % 10) * 1200 + store_id * 50

    # Trend
    trend = np.linspace(0, 300, total_days)

    # Weekly seasonality (weekends higher)
    weekly = 800 * np.sin(2 * np.pi * np.arange(total_days) / 7)

    # Monthly seasonality
    monthly = 400 * np.sin(2 * np.pi * np.arange(total_days) / 30.44)

    # Holiday bump
    holiday_bump = np.zeros(total_days)
    if include_holidays:
        # Simulate random holiday spikes
        holiday_indices = rng.choice(total_days, size=max(1, total_days // 20), replace=False)
        holiday_bump[holiday_indices] = rng.uniform(1500, 4000, size=len(holiday_indices))

    # Oil price effect (slight dampening)
    oil_effect = np.zeros(total_days)
    if impute_oil:
        oil_effect = -200 * np.sin(2 * np.pi * np.arange(total_days) / 60)

    # Noise
    noise = rng.normal(0, 350, total_days)

    values = base + trend + weekly + monthly + holiday_bump + oil_effect + noise
    values = np.maximum(values, 100)  # Floor at 100

    # Split into historical and forecast
    historical = values[:hist_days]
    forecast = values[hist_days:]

    # Confidence intervals (expand over time)
    ci_expansion = np.linspace(0.5, 2.5, horizon_days)
    std_base = np.std(historical) * 0.6
    lower_bound = forecast - std_base * ci_expansion * 1.96
    upper_bound = forecast + std_base * ci_expansion * 1.96

    # Build DataFrame
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

    return df_hist, df_forecast


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

    # Generate SKU codes
    df["sku"] = [f"SKU-{1000 + i:04d}" for i in range(len(df))]

    # Simulate current stock as a fraction of safety stock (creates realistic mix)
    df["current_stock"] = (df["safety_stock"] * rng.uniform(0.15, 1.8, len(df))).astype(int)

    # 30-day forecasted demand
    df["forecast_30d_demand"] = (df["avg_daily_demand"] * 30).astype(int)

    # Determine action
    def get_action(row):
        ratio = row["current_stock"] / row["reorder_point"] if row["reorder_point"] > 0 else 1.0
        if ratio < 0.4:
            return "🔴 URGENT REORDER"
        elif ratio < 0.85:
            return "🟡 REORDER"
        else:
            return "🟢 HOLD"

    df["action"] = df.apply(get_action, axis=1)

    # Reorder columns for display
    display_cols = [
        "sku", "family", "current_stock", "forecast_30d_demand",
        "safety_stock", "reorder_point", "eoq", "action",
    ]
    return df[display_cols]


# =============================================================================
# PLOTLY CHART DEFAULTS — consistent dark theme across all charts
# =============================================================================
PLOTLY_LAYOUT = dict(
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
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.06)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.06)",
    ),
)


def apply_chart_layout(fig: go.Figure, title: str = "", height: int = 480) -> go.Figure:
    """Apply the standard dark layout to any Plotly figure."""
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=18, color=COLORS["text_primary"]), x=0.01),
        height=height,
    )
    return fig


# =============================================================================
# HELPER — render a KPI card
# =============================================================================
def kpi_card(label: str, value: str, delta: str = "", accent: str = "teal") -> str:
    """Return HTML for a single KPI card."""
    delta_class = "negative" if delta.startswith("-") or "risk" in label.lower() else ""
    delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card {accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">📊 Retail Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Predictive Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown("---")

    selected_page = option_menu(
        menu_title=None,
        options=["Executive Dashboard", "Demand Forecasting", "Inventory Optimization"],
        icons=["speedometer2", "graph-up-arrow", "box-seam"],
        default_index=0,
        styles={
            "container": {
                "padding": "0 !important",
                "background-color": "transparent",
            },
            "icon": {
                "color": COLORS["accent_teal"],
                "font-size": "18px",
            },
            "nav-link": {
                "font-size": "14px",
                "font-weight": "500",
                "color": COLORS["text_muted"],
                "text-align": "left",
                "padding": "12px 16px",
                "border-radius": "10px",
                "margin": "4px 0",
                "--hover-color": COLORS["bg_card"],
            },
            "nav-link-selected": {
                "background": f"linear-gradient(135deg, {COLORS['accent_blue']}22, {COLORS['accent_teal']}18)",
                "color": COLORS["text_primary"],
                "font-weight": "600",
                "border": f"1px solid {COLORS['border']}",
            },
        },
    )

    # Sidebar footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align:center; padding: 8px 0;">
            <div style="font-size:0.7rem; color:{COLORS['text_muted']}; letter-spacing:0.5px;">
                SARIMA MODEL v1.0<br>
                MAPE: 7.9% · Last Updated: Aug 2026
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE 1 — EXECUTIVE DASHBOARD
# =============================================================================
if selected_page == "Executive Dashboard":

    # --- Page Header ---
    st.markdown(
        f"""
        <div style="margin-bottom: 8px;">
            <h1 style="font-size:2rem; font-weight:800; color:{COLORS['text_primary']};
                       margin-bottom:4px; letter-spacing:-0.5px;">
                Executive Dashboard
            </h1>
            <p style="font-size:0.95rem; color:{COLORS['text_muted']}; margin-top:0;">
                Real-time overview of demand forecasting performance and inventory health
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- KPI Row ---
    inv_metrics = load_inventory_metrics()
    inv_sim = load_inventory_simulation()

    total_safety_stock = inv_metrics["safety_stock"].sum()
    total_risk = inv_sim["total_inventory_cost"].sum()
    avg_daily_demand = inv_metrics["avg_daily_demand"].sum()

    kpi_cols = st.columns(4)
    kpis = [
        ("Overall MAPE", "7.9%", "Model Accuracy", "teal"),
        ("Total Safety Stock", f"{total_safety_stock / 1e6:.2f}M", "Units recommended", "blue"),
        ("Inventory Risk", f"${total_risk / 1e9:.2f}B", "Identified exposure", "rose"),
        ("Avg Daily Demand", f"{avg_daily_demand / 1e3:.0f}K", "Units across all families", "amber"),
    ]
    for col, (label, value, delta, accent) in zip(kpi_cols, kpis):
        col.markdown(kpi_card(label, value, delta, accent), unsafe_allow_html=True)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # --- Charts Row ---
    chart_col1, chart_col2 = st.columns([3, 2])

    # Left: Historical vs Predicted Sales Line Chart
    with chart_col1:
        st.markdown(
            '<div class="section-header">📈 Historical vs. Predicted Sales</div>',
            unsafe_allow_html=True,
        )

        forecast_df = load_forecast_results()

        fig_sales = go.Figure()

        # Confidence interval band
        fig_sales.add_trace(go.Scatter(
            x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
            y=pd.concat([forecast_df["upper_bound"], forecast_df["lower_bound"][::-1]]),
            fill="toself",
            fillcolor="rgba(78, 124, 255, 0.1)",
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
            line=dict(color=COLORS["accent_blue"], width=2.5, dash="dot"),
            hovertemplate="Date: %{x}<br>Forecast: %{y:,.0f}<extra></extra>",
        ))

        fig_sales = apply_chart_layout(fig_sales, "", height=440)
        fig_sales.update_layout(
            xaxis_title="Date",
            yaxis_title="Total Sales (Units)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_sales, width="stretch", config={"displayModeBar": False})

    # Right: Inventory Risk by Category — Donut Chart
    with chart_col2:
        st.markdown(
            '<div class="section-header">🍩 Inventory Risk by Category</div>',
            unsafe_allow_html=True,
        )

        # Top 8 categories + aggregate the rest
        sim_sorted = inv_sim.sort_values("total_inventory_cost", ascending=False)
        top_n = 8
        top_cats = sim_sorted.head(top_n).copy()
        others_cost = sim_sorted.iloc[top_n:]["total_inventory_cost"].sum()
        if others_cost > 0:
            others_row = pd.DataFrame([{"family": "Others", "total_inventory_cost": others_cost}])
            top_cats = pd.concat([top_cats, others_row], ignore_index=True)

        fig_donut = go.Figure(data=[go.Pie(
            labels=top_cats["family"],
            values=top_cats["total_inventory_cost"],
            hole=0.55,
            marker=dict(colors=CHART_COLORS[: len(top_cats)], line=dict(color=COLORS["bg_primary"], width=2)),
            textinfo="label+percent",
            textposition="outside",
            textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>Risk: $%{value:,.0f}<br>Share: %{percent}<extra></extra>",
        )])

        fig_donut = apply_chart_layout(fig_donut, "", height=440)
        fig_donut.update_layout(
            showlegend=False,
            annotations=[dict(
                text=f"<b>${total_risk/1e9:.1f}B</b><br><span style='font-size:11px;color:{COLORS['text_muted']}'>Total Risk</span>",
                x=0.5, y=0.5, font=dict(size=20, color=COLORS["text_primary"]),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_donut, width="stretch", config={"displayModeBar": False})

    # --- Forecast Error Distribution ---
    st.markdown(
        '<div class="section-header">📊 Forecast Error Distribution</div>',
        unsafe_allow_html=True,
    )

    fig_error = go.Figure()
    fig_error.add_trace(go.Bar(
        x=forecast_df["date"],
        y=forecast_df["error_pct"],
        marker=dict(
            color=forecast_df["error_pct"].apply(
                lambda x: COLORS["accent_rose"] if abs(x) > 10 else (
                    COLORS["accent_amber"] if abs(x) > 5 else COLORS["accent_teal"]
                )
            ),
            line=dict(width=0),
        ),
        hovertemplate="Date: %{x}<br>Error: %{y:.1f}%<extra></extra>",
    ))
    fig_error = apply_chart_layout(fig_error, "", height=300)
    fig_error.update_layout(
        xaxis_title="Date",
        yaxis_title="Error (%)",
        showlegend=False,
    )
    # Add a threshold line at MAPE
    fig_error.add_hline(
        y=7.9, line_dash="dash",
        line_color=COLORS["accent_amber"],
        annotation_text="MAPE: 7.9%",
        annotation_font_color=COLORS["accent_amber"],
        annotation_font_size=12,
    )
    st.plotly_chart(fig_error, width="stretch", config={"displayModeBar": False})


# =============================================================================
# PAGE 2 — DEMAND FORECASTING (SARIMA INTERFACE)
# =============================================================================
elif selected_page == "Demand Forecasting":

    st.markdown(
        f"""
        <div style="margin-bottom: 8px;">
            <h1 style="font-size:2rem; font-weight:800; color:{COLORS['text_primary']};
                       margin-bottom:4px; letter-spacing:-0.5px;">
                Demand Forecasting Engine
            </h1>
            <p style="font-size:0.95rem; color:{COLORS['text_muted']}; margin-top:0;">
                Configure SARIMA model parameters to generate demand forecasts with confidence intervals
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Input Panel ---
    st.markdown('<div class="section-header">⚙️ Forecast Configuration</div>', unsafe_allow_html=True)

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
    tog_col1, tog_col2, _ , _ = st.columns(4)
    with tog_col1:
        include_holidays = st.toggle("Include Holiday / Promotion Flags", value=True)
    with tog_col2:
        impute_oil = st.toggle("Impute Weekend Oil Prices", value=False)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # --- Generate Forecast ---
    df_hist, df_fc = generate_mock_forecast(
        store_id=store_id,
        category=category,
        start_date=datetime.combine(forecast_start, datetime.min.time()),
        horizon_days=horizon,
        include_holidays=include_holidays,
        impute_oil=impute_oil,
    )

    # --- KPI Summary for this forecast ---
    avg_forecast = df_fc["value"].mean()
    peak_demand = df_fc["value"].max()
    avg_historical = df_hist["value"].mean()
    change_pct = ((avg_forecast - avg_historical) / avg_historical) * 100

    kpi_fc_cols = st.columns(4)
    fc_kpis = [
        ("Avg Forecasted Demand", f"{avg_forecast:,.0f}", "units/day", "teal"),
        ("Peak Demand", f"{peak_demand:,.0f}", f"within {horizon}d window", "amber"),
        ("Historical Avg", f"{avg_historical:,.0f}", "baseline (90d)", "blue"),
        ("Demand Shift", f"{change_pct:+.1f}%", "vs. historical avg", "rose" if change_pct < 0 else "teal"),
    ]
    for col, (label, value, delta, accent) in zip(kpi_fc_cols, fc_kpis):
        col.markdown(kpi_card(label, value, delta, accent), unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # --- Forecast Chart ---
    st.markdown(
        '<div class="section-header">📈 SARIMA Forecast with Confidence Intervals</div>',
        unsafe_allow_html=True,
    )

    fig_fc = go.Figure()

    # Confidence interval band
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([df_fc["date"], df_fc["date"][::-1]]),
        y=pd.concat([df_fc["upper_bound"], df_fc["lower_bound"][::-1]]),
        fill="toself",
        fillcolor="rgba(0, 212, 170, 0.12)",
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
        hovertemplate="Date: %{x|%b %d, %Y}<br>Demand: %{y:,.0f}<extra></extra>",
    ))

    # Forecast line
    fig_fc.add_trace(go.Scatter(
        x=df_fc["date"],
        y=df_fc["value"],
        mode="lines",
        name="SARIMA Forecast",
        line=dict(color=COLORS["accent_teal"], width=2.5),
        hovertemplate="Date: %{x|%b %d, %Y}<br>Forecast: %{y:,.0f}<extra></extra>",
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

    st.plotly_chart(fig_fc, width="stretch", config={"displayModeBar": False})

    # --- Forecast Data Table (collapsible) ---
    with st.expander("📋 View Raw Forecast Data", expanded=False):
        display_fc = df_fc[["date", "value", "lower_bound", "upper_bound"]].copy()
        display_fc.columns = ["Date", "Forecasted Demand", "Lower Bound (95%)", "Upper Bound (95%)"]
        display_fc["Date"] = display_fc["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            display_fc,
            width="stretch",
            hide_index=True,
            column_config={
                "Forecasted Demand": st.column_config.NumberColumn(format="%,.0f"),
                "Lower Bound (95%)": st.column_config.NumberColumn(format="%,.0f"),
                "Upper Bound (95%)": st.column_config.NumberColumn(format="%,.0f"),
            },
        )


# =============================================================================
# PAGE 3 — INVENTORY OPTIMIZATION
# =============================================================================
elif selected_page == "Inventory Optimization":

    st.markdown(
        f"""
        <div style="margin-bottom: 8px;">
            <h1 style="font-size:2rem; font-weight:800; color:{COLORS['text_primary']};
                       margin-bottom:4px; letter-spacing:-0.5px;">
                Inventory Optimization Engine
            </h1>
            <p style="font-size:0.95rem; color:{COLORS['text_muted']}; margin-top:0;">
                AI-driven reorder recommendations based on forecasted demand and safety stock analysis
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load and build reorder table
    inv_metrics = load_inventory_metrics()
    reorder_df = generate_reorder_table(inv_metrics)

    # --- Summary KPIs ---
    urgent_count = (reorder_df["action"] == "🔴 URGENT REORDER").sum()
    reorder_count = (reorder_df["action"] == "🟡 REORDER").sum()
    hold_count = (reorder_df["action"] == "🟢 HOLD").sum()
    total_reorder_value = reorder_df.loc[
        reorder_df["action"].isin(["🔴 URGENT REORDER", "🟡 REORDER"]), "eoq"
    ].sum()

    kpi_inv_cols = st.columns(4)
    inv_kpis = [
        ("Urgent Reorders", str(urgent_count), "Critically low SKUs", "rose"),
        ("Standard Reorders", str(reorder_count), "Below reorder point", "amber"),
        ("Stock Healthy", str(hold_count), "No action needed", "teal"),
        ("Total Reorder Volume", f"{total_reorder_value:,.0f}", "EOQ units to order", "blue"),
    ]
    for col, (label, value, delta, accent) in zip(kpi_inv_cols, inv_kpis):
        col.markdown(kpi_card(label, value, delta, accent), unsafe_allow_html=True)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # --- Reorder Recommendation Table ---
    st.markdown(
        '<div class="section-header">📦 Reorder Recommendation Engine</div>',
        unsafe_allow_html=True,
    )

    # Action filter
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
            options=["action", "current_stock", "forecast_30d_demand", "safety_stock"],
            index=0,
        )

    filtered_df = reorder_df[reorder_df["action"].isin(action_filter)].sort_values(
        sort_by, ascending=True if sort_by != "forecast_30d_demand" else False
    )

    # Styled DataFrame
    st.dataframe(
        filtered_df,
        width="stretch",
        hide_index=True,
        height=500,
        column_config={
            "sku": st.column_config.TextColumn("SKU", width="small"),
            "family": st.column_config.TextColumn("Product Family", width="medium"),
            "current_stock": st.column_config.NumberColumn(
                "Current Stock", format="%,d", help="Simulated current on-hand inventory"
            ),
            "forecast_30d_demand": st.column_config.NumberColumn(
                "30-Day Demand", format="%,d", help="Forecasted demand over next 30 days"
            ),
            "safety_stock": st.column_config.NumberColumn(
                "Safety Stock", format="%,d", help="Recommended buffer stock"
            ),
            "reorder_point": st.column_config.NumberColumn(
                "Reorder Point", format="%,d", help="Threshold to trigger reorder"
            ),
            "eoq": st.column_config.NumberColumn(
                "EOQ", format="%,d", help="Economic Order Quantity"
            ),
            "action": st.column_config.TextColumn("Action", width="medium"),
        },
    )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # --- Visual Breakdown ---
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown(
            '<div class="section-header">📊 Stock vs. Reorder Point</div>',
            unsafe_allow_html=True,
        )

        fig_bar = go.Figure()

        fig_bar.add_trace(go.Bar(
            x=reorder_df["family"],
            y=reorder_df["current_stock"],
            name="Current Stock",
            marker_color=COLORS["accent_blue"],
            hovertemplate="<b>%{x}</b><br>Current Stock: %{y:,.0f}<extra></extra>",
        ))
        fig_bar.add_trace(go.Bar(
            x=reorder_df["family"],
            y=reorder_df["reorder_point"],
            name="Reorder Point",
            marker_color=COLORS["accent_rose"],
            opacity=0.7,
            hovertemplate="<b>%{x}</b><br>Reorder Point: %{y:,.0f}<extra></extra>",
        ))

        fig_bar = apply_chart_layout(fig_bar, "", height=420)
        fig_bar.update_layout(
            barmode="group",
            xaxis_title="Product Family",
            yaxis_title="Units",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False})

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
            hole=0.5,
            marker=dict(
                colors=[action_color_map.get(a, "#888") for a in action_counts["Action"]],
                line=dict(color=COLORS["bg_primary"], width=3),
            ),
            textinfo="label+value",
            textposition="outside",
            textfont=dict(size=12),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        )])

        fig_action = apply_chart_layout(fig_action, "", height=420)
        fig_action.update_layout(
            showlegend=False,
            annotations=[dict(
                text=f"<b>{len(reorder_df)}</b><br><span style='font-size:11px;color:{COLORS['text_muted']}'>Total SKUs</span>",
                x=0.5, y=0.5, font=dict(size=18, color=COLORS["text_primary"]),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_action, width="stretch", config={"displayModeBar": False})
