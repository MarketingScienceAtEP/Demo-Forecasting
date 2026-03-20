import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ----------------------------
# App configuration
# ----------------------------
st.set_page_config(
    page_title="Forecasting Solutions Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Data generation
# ----------------------------
@st.cache_data
def generate_dummy_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", "2028-12-01", freq="MS")
    hist_end = pd.Timestamp("2025-12-01")

    df = pd.DataFrame({"date": dates})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["is_forecast"] = df["date"] > hist_end

    # Core drivers
    t = np.arange(len(df))
    season = 1 + 0.12 * np.sin(2 * np.pi * t / 12 - 0.8) + 0.03 * np.cos(4 * np.pi * t / 12)

    # Historical category volumes (Chocolate)
    hist_mask = ~df["is_forecast"]
    n_hist = hist_mask.sum()
    n_fore = (~hist_mask).shape[0] if False else df["is_forecast"].sum()

    base_volume = 920 + 2.2 * np.arange(len(df))
    volume_noise = rng.normal(0, 18, len(df))
    category_volume = base_volume * season + volume_noise
    category_volume = np.maximum(category_volume, 650)

    # Price per unit index leads to value
    price_index = 4.8 + 0.015 * np.arange(len(df))
    price_noise = rng.normal(0, 0.03, len(df))
    category_value = category_volume * (price_index + price_noise)

    # Overwrite forecast period with more structured projections
    forecast_idx = np.where(df["is_forecast"])[0]
    hist_idx = np.where(hist_mask)[0]

    # Use last historical point as anchor
    last_hist_vol = float(category_volume[hist_idx[-1]])
    last_hist_val = float(category_value[hist_idx[-1]])

    # Forecast assumptions by month
    m = np.arange(1, len(forecast_idx) + 1)
    forecast_season = season[forecast_idx]

    # Volume forecast: moderate growth + seasonality
    vol_growth = 1 + 0.0028 * m
    forecast_volume = last_hist_vol * vol_growth * (forecast_season / forecast_season[0]) + rng.normal(0, 10, len(m))
    forecast_volume = np.maximum(forecast_volume, 700)

    # Decomposition drivers for value growth (cumulative effect vs last actual)
    infl_monthly = np.linspace(0.0025, 0.0045, len(m))   # stronger price effect
    pop_monthly = np.linspace(0.0002, 0.00055, len(m))   # small steady support
    other_monthly = 0.0012 + 0.002 * np.sin(2 * np.pi * m / 12 + 0.5)  # mix effect / innovation / promo

    infl_cum = np.cumsum(infl_monthly)
    pop_cum = np.cumsum(pop_monthly)
    other_cum = np.cumsum(other_monthly)

    infl_abs = last_hist_val * infl_cum
    pop_abs = last_hist_val * pop_cum
    other_abs = last_hist_val * other_cum

    # Add small seasonality and link to volume growth so the forecast doesn't look too flat
    seasonal_value_multiplier = 1 + 0.04 * np.sin(2 * np.pi * m / 12 - 0.5)
    volume_link = (forecast_volume / forecast_volume[0]) - 1

    forecast_value = (last_hist_val + infl_abs + pop_abs + other_abs) * seasonal_value_multiplier * (1 + 0.35 * volume_link)

    category_volume[forecast_idx] = forecast_volume
    category_value[forecast_idx] = forecast_value

    df["category_value"] = category_value.round(0)
    df["category_volume"] = category_volume.round(0)

    # Components for forecast decomposition page (absolute contribution above last historical actual)
    df["inflation_effect"] = np.nan
    df["population_effect"] = np.nan
    df["other_effect"] = np.nan
    df.loc[df["is_forecast"], "inflation_effect"] = infl_abs
    df.loc[df["is_forecast"], "population_effect"] = pop_abs
    df.loc[df["is_forecast"], "other_effect"] = other_abs
    df["base_last_actual"] = np.where(df["is_forecast"], last_hist_val, np.nan)

    # Brand XYZ share of category value
    brand_share = 0.285 + 0.01 * np.sin(2 * np.pi * t / 18) + 0.004 * np.cos(2 * np.pi * t / 8)
    brand_share = np.clip(brand_share, 0.24, 0.33)
    brand_total = df["category_value"].values * brand_share
    df["brand_value_total"] = brand_total.round(0)

    # Hierarchical split: SKUs (must sum to total)
    sku_mix = pd.DataFrame({
        "SKU A": 0.42 + 0.03 * np.sin(2 * np.pi * t / 20),
        "SKU B": 0.33 + 0.02 * np.cos(2 * np.pi * t / 14 + 0.4),
        "SKU C": 0.25 + 0.02 * np.sin(2 * np.pi * t / 11 + 1.1),
    })
    sku_mix = sku_mix.div(sku_mix.sum(axis=1), axis=0)
    for col in sku_mix.columns:
        df[col] = (df["brand_value_total"] * sku_mix[col]).round(0)

    # Fix rounding so components add up exactly to total
    df["SKU C"] += df["brand_value_total"] - df[["SKU A", "SKU B", "SKU C"]].sum(axis=1)

    # Hierarchical split: retailer types (must sum to total)
    retail_mix = pd.DataFrame({
        "Discounters": 0.36 + 0.02 * np.sin(2 * np.pi * t / 17),
        "Supermarkets": 0.47 + 0.02 * np.cos(2 * np.pi * t / 13 + 0.2),
        "Others": 0.17 + 0.015 * np.sin(2 * np.pi * t / 9 + 0.7),
    })
    retail_mix = retail_mix.div(retail_mix.sum(axis=1), axis=0)
    for col in retail_mix.columns:
        df[col] = (df["brand_value_total"] * retail_mix[col]).round(0)
    df["Others"] += df["brand_value_total"] - df[["Discounters", "Supermarkets", "Others"]].sum(axis=1)

    # Scenario forecasts for category value
    hist_value = df.loc[hist_mask, "category_value"].copy()
    future_value_base = df.loc[df["is_forecast"], "category_value"].copy().to_numpy()
    scen_dates = df.loc[df["is_forecast"], "date"]
    mm = np.arange(1, len(scen_dates) + 1)

    # Scenario 1: higher inflation, stronger nominal value growth
    scen1 = future_value_base * (1 + 0.03 * (mm / len(mm)) + 0.01 * np.sin(2 * np.pi * mm / 12))
    # Scenario 2: base case
    scen2 = future_value_base.copy()
    # Scenario 3: lower inflation + stronger private label pressure => softer branded/category value
    scen3 = future_value_base * (1 - 0.035 * (mm / len(mm)) - 0.008 * np.sin(2 * np.pi * mm / 12 + 0.4))

    scenarios = pd.DataFrame({
        "date": scen_dates.values,
        "Scenario 1 Forecast": np.round(scen1, 0),
        "Scenario 2 Forecast": np.round(scen2, 0),
        "Scenario 3 Forecast": np.round(scen3, 0),
    })

    return df, scenarios, hist_end


# ----------------------------
# Helper functions
# ----------------------------
def fmt_axis(fig, y_title):
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Month", showgrid=False),
        yaxis=dict(title=y_title, separatethousands=True),
        template="plotly_white",
        height=420,
    )
    return fig


def add_actual_forecast_traces(fig, x_actual, y_actual, x_fc, y_fc, actual_name="Available data", forecast_name="Forecast"):
    fig.add_trace(
        go.Scatter(
            x=x_actual,
            y=y_actual,
            mode="lines",
            name=actual_name,
            line=dict(color="#1f77b4", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_fc,
            y=y_fc,
            mode="lines",
            name=forecast_name,
            line=dict(color="#ff7f0e", width=3, dash="dash"),
        )
    )
    return fig


def page_main(df, hist_end):
    st.title("Forecasting Solutions — Demo Dashboard")
    st.caption("Category: ABC | Frequency: Monthly | Historical data: 2020–2025 | Forecast: 2026–2028")

    hist = df[df["date"] <= hist_end]
    fc = df[df["date"] > hist_end]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest actual month", hist["date"].max().strftime("%b %Y"))
    c2.metric("Latest actual value", f"€{hist['category_value'].iloc[-1]:,.0f}")
    c3.metric("Latest actual volume", f"{hist['category_volume'].iloc[-1]:,.0f}")
    c4.metric("Forecast horizon", f"{len(fc)} months")

    fig1 = go.Figure()
    add_actual_forecast_traces(fig1, hist["date"], hist["category_value"], fc["date"], fc["category_value"])
    fig1.add_vline(x=hist_end, line_width=1, line_dash="dot", line_color="gray")
    fig1.add_annotation(x=hist_end, y=float(df["category_value"].max()) * 1.03, text="Forecast starts", showarrow=False, font=dict(color="gray"))
    fmt_axis(fig1, "Category value (€)")
    fig1.update_layout(title="1. TOTAL VALUE")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    add_actual_forecast_traces(fig2, hist["date"], hist["category_volume"], fc["date"], fc["category_volume"])
    fig2.add_vline(x=hist_end, line_width=1, line_dash="dot", line_color="gray")
    fmt_axis(fig2, "Category volume (units)")
    fig2.update_layout(title="2. TOTAL VOLUME")
    st.plotly_chart(fig2, use_container_width=True)



def page_decomposition(df, hist_end):
    st.title("Forecast Decomposition")
    st.caption("Available data and forecast for ABC category value, with driver decomposition for forecast months.")

    hist = df[df["date"] <= hist_end]
    fc = df[df["date"] > hist_end].copy()

    fig1 = go.Figure()
    add_actual_forecast_traces(fig1, hist["date"], hist["category_value"], fc["date"], fc["category_value"])
    fig1.add_vline(x=hist_end, line_width=1, line_dash="dot", line_color="gray")
    fmt_axis(fig1, "Category value (€)")
    fig1.update_layout(title="3A. TOTAL CATEGORY VALUE")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=fc["date"], y=fc["inflation_effect"], name="Effect of inflation", marker_color="#d62728"))
    fig2.add_trace(go.Bar(x=fc["date"], y=fc["population_effect"], name="Effect of population", marker_color="#2ca02c"))
    fig2.add_trace(go.Bar(x=fc["date"], y=fc["other_effect"], name="Effect of other sources", marker_color="#9467bd"))
    fig2.add_trace(
        go.Scatter(
            x=fc["date"],
            y=fc["category_value"],
            name="Total forecast value",
            mode="lines",
            line=dict(color="#111111", width=3),
            yaxis="y2",
        )
    )
    fig2.update_layout(
        barmode="stack",
        margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Forecast month", showgrid=False),
        yaxis=dict(title="Cumulative contribution vs Dec 2025 (€)", separatethousands=True),
        yaxis2=dict(title="Forecast value (€)", overlaying="y", side="right", separatethousands=True),
        height=460,
        title="3B. Forecast decomposition: inflation, population, and other sources",
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("What the decomposition means"):
        st.markdown(
            """
            - **Effect of inflation**: nominal value lift associated with price changes.
            - **Effect of population**: small structural growth from more households / shoppers.
            - **Effect of other sources**: assortment, innovation, promo intensity, mix, and unexplained residuals.
            - The stacked bars show **cumulative contribution versus the last actual month (Dec 2025)**.
            - The black line shows the **final forecasted category value** for each month.
            """
        )



def page_hierarchy(df, hist_end):
    st.title("Hierarchical Forecasts")
    st.caption("Top panel = total brand value for XYZ. Bottom panel = decomposition that adds up exactly to the total.")

    split_choice = st.selectbox(
        "Choose bottom-panel decomposition",
        ["SKUs", "Retailer types"],
        index=0,
    )

    hist = df[df["date"] <= hist_end]
    fc = df[df["date"] > hist_end]

    if split_choice == "SKUs":
        components = ["SKU A", "SKU B", "SKU C"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    else:
        components = ["Discounters", "Supermarkets", "Others"]
        colors = ["#d62728", "#17becf", "#9467bd"]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Total brand value (XYZ)",
            f"Decomposition by {split_choice.lower()}",
        ),
    )

    # Top panel
    fig.add_trace(
        go.Scatter(
            x=hist["date"], y=hist["brand_value_total"], mode="lines", name="Available data",
            line=dict(color="#1f77b4", width=3)
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=fc["date"], y=fc["brand_value_total"], mode="lines", name="Forecast",
            line=dict(color="#ff7f0e", width=3, dash="dash")
        ), row=1, col=1
    )

    # Bottom panel (stacked area)
    for comp, color in zip(components, colors):
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[comp],
                mode="lines",
                stackgroup="one",
                name=comp,
                line=dict(width=1.5, color=color),
            ),
            row=2,
            col=1,
        )

    fig.add_vline(x=hist_end, line_width=1, line_dash="dot", line_color="gray", row=1, col=1)
    fig.add_vline(x=hist_end, line_width=1, line_dash="dot", line_color="gray", row=2, col=1)
    fig.update_layout(
        margin=dict(l=20, r=20, t=80, b=20),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=720,
        title="4. Hierarchical forecasts for Brand XYZ",
    )
    fig.update_yaxes(title_text="Brand value (€)", row=1, col=1, separatethousands=True)
    fig.update_yaxes(title_text="Component value (€)", row=2, col=1, separatethousands=True)
    fig.update_xaxes(title_text="Month", row=2, col=1, showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    # Validation message
    check = (df[components].sum(axis=1) - df["brand_value_total"]).abs().max()
    st.success(f"Validation check: bottom-panel components add up to the total brand value forecast (max rounding difference = {check:,.0f}).")



def page_scenarios(df, scenarios, hist_end):
    st.title("Scenario Forecasts")
    st.caption("Compare alternative forward-looking environments for Category ABC value.")

    hist = df[df["date"] <= hist_end]
    scenario = st.selectbox(
        "Select scenario",
        ["Scenario 1 Forecast", "Scenario 2 Forecast", "Scenario 3 Forecast"],
        index=1,
    )

    descriptions = {
        "Scenario 1 Forecast": "High inflationary environment",
        "Scenario 2 Forecast": "Base Case",
        "Scenario 3 Forecast": "Low inflationary environment"
    }
    st.info(descriptions[scenario])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist["date"], y=hist["category_value"], mode="lines", name="Available data",
            line=dict(color="#1f77b4", width=3)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=scenarios["date"], y=scenarios[scenario], mode="lines", name=scenario,
            line=dict(color="#ff7f0e", width=3, dash="dash")
        )
    )
    fig.add_vline(x=hist_end, line_width=1, line_dash="dot", line_color="gray")
    fig.add_annotation(x=hist_end, y=float(max(hist["category_value"].max(), scenarios[scenario].max())) * 1.03,
                       text="Scenario forecast starts", showarrow=False, font=dict(color="gray"))
    fmt_axis(fig, "Category value (€)")
    fig.update_layout(title="5. Scenario forecasts for Category ABC value")
    st.plotly_chart(fig, use_container_width=True)

    # Quick scenario summary table
    scenario_summary = pd.DataFrame({
        "Scenario": ["Scenario 1 Forecast", "Scenario 2 Forecast", "Scenario 3 Forecast"],
        "Dec 2028 forecast (€)": [
            int(scenarios["Scenario 1 Forecast"].iloc[-1]),
            int(scenarios["Scenario 2 Forecast"].iloc[-1]),
            int(scenarios["Scenario 3 Forecast"].iloc[-1]),
        ],
        "Description": [
            descriptions["Scenario 1 Forecast"],
            descriptions["Scenario 2 Forecast"],
            descriptions["Scenario 3 Forecast"],
        ],
    })
    st.dataframe(scenario_summary, use_container_width=True, hide_index=True)


# ----------------------------
# Main app
# ----------------------------
df, scenarios, hist_end = generate_dummy_data()

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to page",
        [
            "Basic Forecasts",
            "Forecast Decomposition",
            "Hierarchical Forecast Decomposition",
            "Scenario-Based Forecasts",
        ],
    )

    st.divider()
    st.subheader("Demo scope")
    st.markdown(
        """
        - **Sample**: 2020–2025
        - **Forecast**: 2026–2028
        - **Frequency**: Monthly
        - **Category**: Category ABC
        - **Brand**: Brand XYZ
        """
    )

    st.divider()
    st.caption("All numbers are synthetic and for proposal/demo purposes only.")
    # End of sidebar

# Render the selected page
if page == "Basic Forecasts":
    page_main(df, hist_end)
elif page == "Forecast Decomposition":
    page_decomposition(df, hist_end)
elif page == "Hierarchical Forecast Decomposition":
    page_hierarchy(df, hist_end)
elif page == "Scenario-Based Forecasts":
    page_scenarios(df, scenarios, hist_end)

