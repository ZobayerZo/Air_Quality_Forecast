import calendar
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dashboard"
MODELS = ROOT / "models"
ASSETS = ROOT / "assets"
HORIZONS = [24, 48, 72]
WHO = 5.0  # WHO annual PM2.5 guideline, µg/m³

st.set_page_config(page_title="Air Quality Forecast", page_icon="🌫️", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown(
    """
<style>
@keyframes fadeUp {from {opacity:0; transform:translateY(14px);} to {opacity:1; transform:none;}}
@keyframes pulse {0% {box-shadow:0 0 0 0 rgba(255,255,255,.45);}
                  70% {box-shadow:0 0 0 20px rgba(255,255,255,0);}
                  100% {box-shadow:0 0 0 0 rgba(255,255,255,0);}}
.hero {border-radius:18px; padding:22px 28px; margin-bottom:14px; animation:fadeUp .7s ease both;}
.hero .big {font-size:3.4rem; font-weight:800; line-height:1.1;}
.pulse {animation:fadeUp .7s ease both, pulse 2.2s infinite;}
.advice {border-left:6px solid; border-radius:8px; padding:12px 16px;
         background:rgba(128,128,128,.08); animation:fadeUp .8s ease both;}
.small {opacity:.75; font-size:.85rem;}
</style>
""",
    unsafe_allow_html=True,
)

# US EPA PM2.5 categories: (upper bound, label, colour, text colour)
BANDS = [
    (9.0, "Good", "#2e9e44", "#ffffff"),
    (35.4, "Moderate", "#f2c200", "#1a1a1a"),
    (55.4, "Unhealthy for sensitive groups", "#f28c00", "#ffffff"),
    (125.4, "Unhealthy", "#e03131", "#ffffff"),
    (225.4, "Very unhealthy", "#8f3fb0", "#ffffff"),
    (float("inf"), "Hazardous", "#7a1f2b", "#ffffff"),
]
BAND_COLORS = {label: color for _, label, color, _ in BANDS}
ADVICE = {
    "Good": "Air quality is good. Enjoy outdoor activities.",
    "Moderate": "Acceptable for most people. Unusually sensitive people should "
                "consider shorter outdoor exertion.",
    "Unhealthy for sensitive groups": "Children, older adults and people with asthma or heart "
                "conditions should reduce prolonged outdoor exertion. A well-fitted mask helps.",
    "Unhealthy": "Everyone should reduce prolonged outdoor exertion. Wear an N95/FFP2 mask "
                 "outdoors, keep windows closed, and use an air purifier if you have one.",
    "Very unhealthy": "Avoid outdoor exertion. Stay indoors with windows closed; wear an "
                      "N95/FFP2 mask if you must go out.",
    "Hazardous": "Health emergency conditions. Stay indoors, use an air purifier, and avoid "
                 "all outdoor activity.",
}


def band(pm):
    for upper, label, color, text in BANDS:
        if pm <= upper:
            return label, color, text
    return BANDS[-1][1:]


def aqi_label(pm):
    return band(pm)[0]


# ------------------------------------------------------------------ data
@st.cache_data
def load_data():
    d = dict(
        recent=pd.read_parquet(DATA / "recent.parquet"),
        forecast=pd.read_csv(DATA / "latest_forecast.csv",
                             parse_dates=["base_time", "forecast_time"]),
        intervals=pd.read_csv(DATA / "intervals.csv"),
        backtest=pd.read_parquet(DATA / "backtest_recent.parquet"),
        shap=pd.read_csv(DATA / "shap_importance.csv"),
        metrics=pd.read_csv(MODELS / "metrics.csv"),
        monthly=pd.read_csv(DATA / "city_monthly_summary.csv"),
        profile=pd.read_parquet(DATA / "hourly_profile.parquet"),
        latest_feats=None,
    )
    if (DATA / "latest_features.parquet").exists():
        d["latest_feats"] = pd.read_parquet(DATA / "latest_features.parquet")
    with open(MODELS / "meta.json") as f:
        d["features"] = json.load(f)["features"]
    return d


@st.cache_resource
def load_boosters():
    try:
        import lightgbm as lgb
        return {h: lgb.Booster(model_file=str(MODELS / f"lgbm_{h}h.txt")) for h in HORIZONS}
    except Exception:
        return None


d = load_data()
cities = sorted(d["recent"]["city"].unique())
coords = d["recent"].drop_duplicates("city").set_index("city")[["latitude", "longitude"]]
base_time = d["forecast"]["base_time"].max()
age_h = (pd.Timestamp.now("UTC").tz_localize(None) - base_time).total_seconds() / 3600

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🌫️ Air Quality Forecast")
default_city = st.query_params.get("city", "Dhaka")
if default_city not in cities:
    default_city = "Dhaka" if "Dhaka" in cities else cities[0]
city = st.sidebar.selectbox("Choose a city", cities, index=cities.index(default_city))
st.query_params["city"] = city  # makes the URL shareable: ...?city=Dhaka

if age_h <= 36:
    st.sidebar.success(f"Live: forecast updated {age_h:.0f} h ago")
else:
    st.sidebar.warning(f"Forecast is {age_h / 24:.1f} days old")
st.sidebar.caption(
    "24/48/72-hour PM2.5 forecasts from LightGBM models trained on 4 years of hourly "
    "data for 30 cities (1M+ records)."
)
with st.sidebar.expander("About the data"):
    st.write(
        "Air quality and weather data: [Open-Meteo](https://open-meteo.com/) "
        "(CAMS atmospheric model and ERA5 reanalysis). Values are **modeled**, not "
        "ground-station measurements, so they can differ from local monitors."
    )

# ------------------------------------------------------------------ hero
rc = d["recent"][d["recent"]["city"] == city].sort_values("time")
fc = (d["forecast"][d["forecast"]["city"] == city]
      .merge(d["intervals"], on=["city", "horizon_h"]).sort_values("horizon_h"))
fc["lo"] = (fc["pm2_5_pred"] + fc["q10"]).clip(lower=0)
fc["hi"] = fc["pm2_5_pred"] + fc["q90"]
current = float(rc["pm2_5"].iloc[-1])
label, color, txt = band(current)
pulse = " pulse" if current > 55.4 else ""
st.markdown(
    f"""<div class="hero{pulse}" style="background:linear-gradient(135deg,{color},{color}cc);color:{txt}">
<div style="opacity:.9">{city} · latest modeled PM2.5</div>
<div class="big">{current:.0f} <span style="font-size:1.2rem;font-weight:500">µg/m³</span></div>
<div style="font-size:1.15rem">{label} · {current / WHO:.1f}× the WHO annual guideline (5 µg/m³)</div>
</div>""",
    unsafe_allow_html=True,
)

tab_fc, tab_map, tab_time, tab_pat, tab_wi, tab_model, tab_shap = st.tabs(
    ["Forecast", "Live map", "Time machine", "Patterns & compare",
     "What-if lab", "Model performance", "Why? (SHAP)"]
)

# ------------------------------------------------------------------ Forecast
with tab_fc:
    options = ["Now"] + [f"+{h}h" for h in HORIZONS]
    sel = st.select_slider("Look ahead", options=options, value="+24h")
    values = {"Now": current}
    for _, r in fc.iterrows():
        values[f"+{int(r.horizon_h)}h"] = float(r.pm2_5_pred)
    val = values.get(sel, current)

    left, right = st.columns([1, 1.15])
    with left:
        steps = [0, 9, 35.4, 55.4, 125.4, 225.4, 300]
        fig = go.Figure(go.Indicator(
            mode="gauge+number" if sel == "Now" else "gauge+number+delta",
            value=val,
            number={"suffix": " µg/m³", "font": {"size": 38}},
            delta={"reference": current, "increasing": {"color": "#e03131"},
                   "decreasing": {"color": "#2e9e44"}},
            title={"text": f"{city}: {sel}"},
            gauge={"axis": {"range": [0, 300]},
                   "bar": {"color": "rgba(25,25,25,0.85)", "thickness": 0.25},
                   "steps": [{"range": [steps[i], steps[i + 1]], "color": BANDS[i][2]}
                             for i in range(6)]},
        ))
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=10),
                          transition={"duration": 700, "easing": "cubic-in-out"})
        st.plotly_chart(fig, width="stretch")
    with right:
        lab, col, _ = band(val)
        st.markdown(
            f"""<div class="advice" style="border-color:{col}">
<b>{lab}</b> ({sel})<br>{ADVICE[lab]}<br>
<span class="small">General information, not medical advice.</span></div>""",
            unsafe_allow_html=True,
        )
        st.write("")
        cols = st.columns(3)
        for c, (_, r) in zip(cols, fc.iterrows()):
            c.metric(f"+{int(r.horizon_h)}h", f"{r.pm2_5_pred:.0f}",
                     delta=f"{r.pm2_5_pred - current:+.0f}", delta_color="inverse")
            c.caption(f"80% range {r.lo:.0f}–{r.hi:.0f}")

    xs = [base_time] + list(fc["forecast_time"])
    ymax = max(rc["pm2_5"].tail(168).max(), fc["hi"].max()) * 1.15
    fig = go.Figure()
    lower = 0
    for upper, lab, colr, _ in BANDS[:-1]:
        if lower >= ymax:
            break
        fig.add_hrect(y0=lower, y1=min(upper, ymax), fillcolor=colr, opacity=0.08,
                      line_width=0)
        lower = upper
    fig.add_trace(go.Scatter(x=rc["time"].tail(168), y=rc["pm2_5"].tail(168),
                             name="Past 7 days", line=dict(color="#2a7de1")))
    fig.add_trace(go.Scatter(
        x=xs + xs[::-1],
        y=[current] + list(fc["hi"]) + list(fc["lo"])[::-1] + [current],
        fill="toself", fillcolor="rgba(224,86,42,0.18)", line=dict(width=0),
        name="80% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[current] + list(fc["pm2_5_pred"]), name="Forecast",
                             mode="lines+markers", line=dict(color="#e0562a", dash="dash")))
    fig.update_layout(yaxis_title="PM2.5 (µg/m³)", height=420, hovermode="x unified",
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Forecast issued from data ending {base_time:%Y-%m-%d %H:%M} UTC. "
               "Background colours are the US EPA categories. Drag the slider under the "
               "chart to zoom.")
    export = fc[["city", "base_time", "horizon_h", "forecast_time", "pm2_5_pred", "lo", "hi"]]
    st.download_button("⬇ Download this forecast (CSV)", export.to_csv(index=False),
                       file_name=f"{city}_pm25_forecast.csv", mime="text/csv")

# ------------------------------------------------------------------ Live map
with tab_map:
    latest = (d["recent"].sort_values("time").groupby("city").tail(1)
              [["city", "latitude", "longitude", "pm2_5"]]
              .rename(columns={"pm2_5": "Latest"}))
    piv = d["forecast"].pivot(index="city", columns="horizon_h", values="pm2_5_pred")
    piv.columns = [f"+{h}h" for h in piv.columns]
    m = latest.merge(piv.reset_index(), on="city")

    c1, c2, c3 = st.columns([2.2, 1.4, 1])
    choice = c1.radio("Show", ["Latest", "+24h", "+48h", "+72h"], horizontal=True)
    style = c2.selectbox("Map style", ["open-street-map", "carto-positron", "carto-darkmatter"])
    focus = c3.checkbox(f"Zoom to {city}")
    m["PM2.5"] = m[choice].round(1)
    m["Category"] = m["PM2.5"].map(aqi_label)
    m["size"] = m["PM2.5"].clip(lower=8)
    kw = dict(zoom=1)
    if focus:
        kw = dict(zoom=5, center=dict(lat=float(coords.loc[city, "latitude"]),
                                      lon=float(coords.loc[city, "longitude"])))
    fig = px.scatter_map(m, lat="latitude", lon="longitude", color="Category", size="size",
                         size_max=24, hover_name="city",
                         hover_data={"PM2.5": True, "size": False, "latitude": False,
                                     "longitude": False},
                         color_discrete_map=BAND_COLORS, map_style=style, height=520, **kw)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, width="stretch")

    st.subheader("City ranking")
    table = m[["city", "Latest", "+24h", "+48h", "+72h"]].sort_values("Latest", ascending=False)
    table = table.rename(columns={"city": "City"})
    cfg = {c: st.column_config.ProgressColumn(c, min_value=0, max_value=250, format="%.0f")
           for c in ["Latest", "+24h", "+48h", "+72h"]}
    st.dataframe(table, hide_index=True, width="stretch", column_config=cfg)

# ------------------------------------------------------------------ Time machine
with tab_time:
    st.write("Press **▶ Play** to watch four years of pollution unfold month by month.")
    view = st.radio("View", ["Animated map", "Pollution race"], horizontal=True)
    mon = d["monthly"].sort_values(["year_month", "city"]).copy()
    if view == "Animated map":
        mm = mon.merge(coords.reset_index(), on="city")
        mm["size"] = mm["avg_pm2_5"].clip(lower=5)
        fig = px.scatter_map(mm, lat="latitude", lon="longitude", color="avg_pm2_5",
                             size="size", size_max=26, animation_frame="year_month",
                             hover_name="city", color_continuous_scale="YlOrRd",
                             range_color=[0, 120], zoom=1, map_style="carto-positron",
                             height=600, labels={"avg_pm2_5": "PM2.5 (µg/m³)"})
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    else:
        order = (mon.groupby("city")["avg_pm2_5"].mean().sort_values().index.tolist())
        fig = px.bar(mon, x="avg_pm2_5", y="city", orientation="h", color="avg_pm2_5",
                     animation_frame="year_month", category_orders={"city": order},
                     range_x=[0, max(150, mon["avg_pm2_5"].max() * 1.05)],
                     color_continuous_scale="YlOrRd", range_color=[0, 120],
                     labels={"avg_pm2_5": "Monthly mean PM2.5 (µg/m³)", "city": ""},
                     height=max(450, 24 * len(order)))
        fig.update_layout(coloraxis_showscale=False)
    try:  # speed up the animation
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 450
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 250
    except Exception:
        pass
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ Patterns & compare
with tab_pat:
    st.subheader(f"When is the air worst in {city}?")
    lon = float(coords.loc[city, "longitude"])
    off = int(round(lon / 15))  # rough UTC offset from longitude
    pr = d["profile"][d["profile"]["city"] == city].copy()
    pr["lhour"] = (pr["hour"] + off) % 24
    hm = pr.pivot_table(index="lhour", columns="month", values="avg_pm2_5").sort_index()
    by_hour = pr.groupby("lhour")["avg_pm2_5"].mean()
    a, b = st.columns([1.5, 1])
    fig = go.Figure(go.Heatmap(z=hm.values, x=[calendar.month_abbr[int(x)] for x in hm.columns],
                               y=hm.index, colorscale="YlOrRd",
                               colorbar=dict(title="µg/m³")))
    fig.update_layout(height=380, yaxis_title="Hour of day (approx. local)",
                      margin=dict(l=10, r=10, t=30, b=10))
    fig.update_yaxes(dtick=3)
    a.plotly_chart(fig, width="stretch")
    fig = px.line(x=by_hour.index, y=by_hour.values, labels={"x": "Hour (approx. local)",
                                                              "y": "Mean PM2.5 (µg/m³)"})
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
    b.plotly_chart(fig, width="stretch")
    st.info(f"Cleanest hour: about **{int(by_hour.idxmin()):02d}:00**  ·  "
            f"most polluted hour: about **{int(by_hour.idxmax()):02d}:00** "
            "(local time estimated from longitude, so it can be off by up to an hour).")

    mo = d["monthly"][d["monthly"]["city"] == city].copy()
    mo["Year"] = mo["year_month"].str[:4]
    mo["Month"] = mo["year_month"].str[5:7].astype(int)
    fig = px.line(mo, x="Month", y="avg_pm2_5", color="Year", markers=True,
                  labels={"avg_pm2_5": "Monthly mean PM2.5 (µg/m³)"})
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)),
                     ticktext=[calendar.month_abbr[i] for i in range(1, 13)])
    fig.add_hline(y=WHO, line_dash="dot", annotation_text="WHO guideline (5)")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Compare cities")
    picked = st.multiselect("Cities", cities,
                            default=[c for c in ["Dhaka", "Delhi", "London"] if c in cities])
    if picked:
        sub = d["monthly"][d["monthly"]["city"].isin(picked)]
        fig = px.line(sub, x="year_month", y="avg_pm2_5", color="city",
                      labels={"avg_pm2_5": "Monthly mean PM2.5 (µg/m³)", "year_month": ""})
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")
        avg = (sub.groupby("city")["avg_pm2_5"].mean().round(1).sort_values(ascending=False)
               .reset_index())
        fig = px.bar(avg, x="city", y="avg_pm2_5", text="avg_pm2_5",
                     labels={"avg_pm2_5": "Average PM2.5 (µg/m³)", "city": ""})
        fig.add_hline(y=WHO, line_dash="dot", annotation_text="WHO guideline (5)")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ What-if lab
with tab_wi:
    st.subheader(f"What-if lab: {city}")
    st.caption("Change the weather and see how the trained model's forecast responds.")
    boosters = load_boosters()
    lf = d["latest_feats"]
    if boosters is None or lf is None or lf[lf["city"] == city].empty:
        st.info("The what-if lab needs the model files and `latest_features.parquet`.")
    else:
        feats = d["features"]
        row = lf[lf["city"] == city].iloc[[0]]
        r = row.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        wind = c1.slider("Wind speed (km/h)", 0.0, 60.0,
                         float(np.clip(r["wind_speed_10m"], 0, 60)), 0.5, key=f"w_{city}")
        hum = c2.slider("Humidity (%)", 5.0, 100.0,
                        float(np.clip(r["relative_humidity_2m"], 5, 100)), 1.0, key=f"h_{city}")
        temp = c3.slider("Temperature (°C)", -10.0, 45.0,
                         float(np.clip(r["temperature_2m"], -10, 45)), 0.5, key=f"t_{city}")
        rain = c4.slider("Rain, last 24h (mm)", 0.0, 100.0,
                         float(np.clip(r["precip_sum_24h"], 0, 100)), 0.5, key=f"r_{city}")
        X0 = row[feats].copy()
        X1 = X0.copy()
        for col in ("wind_speed_10m", "wind_roll_mean_24h"):
            X1[col] = wind
        for col in ("relative_humidity_2m", "humidity_roll_mean_24h"):
            X1[col] = hum
        for col in ("temperature_2m", "temp_roll_mean_24h"):
            X1[col] = temp
        X1["precip_sum_24h"] = rain
        base = [float(max(boosters[h].predict(X0)[0], 0)) for h in HORIZONS]
        scen = [float(max(boosters[h].predict(X1)[0], 0)) for h in HORIZONS]
        cols = st.columns(3)
        for c, h, b0, s1 in zip(cols, HORIZONS, base, scen):
            c.metric(f"+{h}h scenario", f"{s1:.1f} µg/m³", delta=f"{s1 - b0:+.1f} vs current weather",
                     delta_color="inverse")
        res = pd.DataFrame({"Horizon": [f"+{h}h" for h in HORIZONS] * 2,
                            "PM2.5": base + scen,
                            "Case": ["Current weather"] * 3 + ["Your scenario"] * 3})
        fig = px.bar(res, x="Horizon", y="PM2.5", color="Case", barmode="group",
                     labels={"PM2.5": "Predicted PM2.5 (µg/m³)"},
                     color_discrete_sequence=["#2a7de1", "#e0562a"])
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("This probes the machine-learning model, not atmospheric physics. Effects "
                   "are modest because recent PM2.5 history (about 70% of model influence) "
                   "stays fixed; only the weather inputs change.")

# ------------------------------------------------------------------ Model performance
with tab_model:
    st.subheader("Test results (Nov 2025 – Oct 2026, never used for training)")
    c1, c2 = st.columns([1, 1.2])
    c1.dataframe(d["metrics"], hide_index=True, width="stretch")
    fig = px.bar(d["metrics"], x="horizon_h", y="MAE", color="model", barmode="group",
                 labels={"horizon_h": "Horizon (hours)", "MAE": "MAE (µg/m³)"})
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    c2.plotly_chart(fig, width="stretch")

    h = st.radio("Backtest horizon", HORIZONS, horizontal=True, format_func=lambda x: f"+{x}h")
    bt = d["backtest"][d["backtest"]["city"] == city].dropna(subset=[f"actual_{h}h"])
    when = bt["time"] + pd.Timedelta(hours=int(h))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=when, y=bt[f"actual_{h}h"], name="Actual",
                             line=dict(color="#2a7de1")))
    fig.add_trace(go.Scatter(x=when, y=bt[f"pred_{h}h"], name=f"Predicted ({h}h ahead)",
                             line=dict(color="#e0562a")))
    fig.update_layout(yaxis_title="PM2.5 (µg/m³)", height=380, hovermode="x unified",
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
    st.caption("Last ~120 days of the test period for the selected city.")

# ------------------------------------------------------------------ SHAP
with tab_shap:
    s = d["shap"][d["shap"]["city"] == city].nlargest(15, "mean_abs_shap").iloc[::-1]
    fig = px.bar(s, x="mean_abs_shap", y="feature", color="group", orientation="h",
                 labels={"mean_abs_shap": "Mean |SHAP| (µg/m³)", "feature": ""},
                 title=f"Top drivers of the +24h forecast: {city}")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    p1, p2 = ASSETS / "shap_dhaka_beeswarm.png", ASSETS / "shap_driver_groups.png"
    if p1.exists():
        c1.image(str(p1), caption="Dhaka: effect of each feature")
    if p2.exists():
        c2.image(str(p2), caption="Share of influence by feature group")
    st.caption("SHAP describes what the model relies on, not causal effects.")