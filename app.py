"""
Aftershock Risk Forecaster
----------------------------
Real earthquake aftershock forecasting using the Reasenberg-Jones modified
Omori-Utsu model — the same rate model behind the USGS's own public
aftershock forecasts. Pick a real recent significant earthquake (or enter
a hypothetical one), and the app forecasts the probability of further
aftershocks at various magnitude thresholds over the next day/week/month,
with proper uncertainty via Monte Carlo, and — if real aftershock data is
available — a Bayesian update that adapts the generic model to that
specific sequence's own observed behavior.

Why this problem: aftershock risk communication is a genuine, high-stakes
real-world forecasting problem — after a major earthquake, emergency
managers and the public need calibrated answers to "how likely is another
big one," not vague reassurance or vague alarm. This is exactly the
question the USGS's Operational Aftershock Forecasting system exists to
answer, using this same statistical machinery.

Run with:
    streamlit run app.py
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from aftershock_model import (
    GENERIC_A, GENERIC_A_SIGMA, GENERIC_B, GENERIC_C, GENERIC_P,
    completeness_magnitude, expected_count, forecast, posterior_over_a, rate,
)
from usgs_client import USGSClientError, fetch_aftershock_sequence, fetch_significant_earthquakes

st.set_page_config(page_title="Aftershock Risk Forecaster", layout="wide", page_icon="\U0001F30B")


@st.cache_data(ttl=1800, show_spinner=False)
def cached_significant(period: str):
    return fetch_significant_earthquakes(period)


@st.cache_data(ttl=900, show_spinner=False)
def cached_sequence(mainshock_time_iso, lat, lon, radius_km, days_after, min_mag):
    mainshock_time = pd.Timestamp(mainshock_time_iso).to_pydatetime()
    return fetch_aftershock_sequence(mainshock_time, lat, lon, radius_km=radius_km,
                                      days_after=days_after, min_magnitude=min_mag)


st.title("\U0001F30B Aftershock Risk Forecaster")
st.caption(
    "The Reasenberg-Jones modified Omori-Utsu model — the statistical engine behind the "
    "USGS's own public aftershock forecasts — applied to a real or hypothetical mainshock."
)

st.sidebar.title("Mainshock")
source = st.sidebar.radio("Choose a mainshock", ["Pick a recent significant earthquake", "Enter one manually"])

mainshock = None  # dict with mag, lat, lon, time, place

if source == "Pick a recent significant earthquake":
    period = st.sidebar.radio("Look back over", ["month", "week"], horizontal=True)
    try:
        with st.spinner("Fetching significant earthquakes from USGS..."):
            sig_df = cached_significant(period)
    except USGSClientError as e:
        st.error(str(e))
        st.stop()
    if sig_df.empty:
        st.warning(f"No significant earthquakes reported in the last {period}. Try the other period, or enter one manually.")
        st.stop()
    sig_df = sig_df.sort_values("time", ascending=False)
    labels = [f"M{row.magnitude:.1f} — {row.place} ({row.time.strftime('%Y-%m-%d')})" for row in sig_df.itertuples()]
    choice = st.sidebar.selectbox("Event", labels)
    row = sig_df.iloc[labels.index(choice)]
    mainshock = {
        "magnitude": float(row["magnitude"]), "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]), "time": row["time"], "place": row["place"],
    }
else:
    c1, c2 = st.sidebar.columns(2)
    mag = c1.number_input("Magnitude", 3.0, 9.5, 6.5, 0.1)
    lat = c2.number_input("Latitude", -90.0, 90.0, 35.0, 0.1)
    lon = st.sidebar.number_input("Longitude", -180.0, 180.0, 139.0, 0.1)
    date = st.sidebar.date_input("Date", datetime.now(timezone.utc).date())
    mainshock = {"magnitude": mag, "latitude": lat, "longitude": lon,
                 "time": pd.Timestamp(date, tz="UTC"), "place": "Hypothetical event"}

st.sidebar.divider()
radius_km = st.sidebar.slider("Aftershock search radius (km)", 20, 500, 150)
days_after = st.sidebar.slider("Sequence window to fetch (days since mainshock)", 1, 90, 30)
min_mag_catalog = st.sidebar.slider("Minimum catalog magnitude", 1.0, 5.0, 2.5, 0.1)

now = pd.Timestamp.now(tz="UTC")
mainshock_time = pd.Timestamp(mainshock["time"])
if mainshock_time.tzinfo is None:
    mainshock_time = mainshock_time.tz_localize("UTC")
t_now_days = max((now - mainshock_time).total_seconds() / 86400, 0.01)

st.header(f"M{mainshock['magnitude']:.1f} — {mainshock['place']}")
st.caption(f"Occurred {mainshock_time.strftime('%Y-%m-%d %H:%M UTC')} — {t_now_days:.1f} days ago")

sequence_df = pd.DataFrame()
if source == "Pick a recent significant earthquake" or st.sidebar.checkbox("Also fetch real aftershock data for this hypothetical location", value=False):
    try:
        with st.spinner("Fetching aftershock sequence from the USGS catalog..."):
            sequence_df = cached_sequence(mainshock_time.isoformat(), mainshock["latitude"], mainshock["longitude"],
                                           radius_km, min(days_after, int(t_now_days) + 1), min_mag_catalog)
    except USGSClientError as e:
        st.warning(f"Could not fetch real aftershock data: {e}")

mc = completeness_magnitude(sequence_df["magnitude"], floor=min_mag_catalog) if not sequence_df.empty else min_mag_catalog
complete_df = sequence_df[sequence_df["magnitude"] >= mc - 1e-9] if not sequence_df.empty else sequence_df

tab_seq, tab_forecast, tab_about = st.tabs(["\U0001F5FA️ Sequence", "\U0001F52E Forecast", "ℹ️ Methodology"])

with tab_seq:
    if sequence_df.empty:
        st.info(
            "No observed aftershock data available (either a hypothetical mainshock, or none "
            "found in the search radius/window). The Forecast tab will use the generic model only."
        )
    else:
        sequence_df["days_since_mainshock"] = (sequence_df["time"] - mainshock_time).dt.total_seconds() / 86400
        c1, c2, c3 = st.columns(3)
        c1.metric("Aftershocks found", len(sequence_df))
        c2.metric("Largest aftershock", f"M{sequence_df['magnitude'].max():.1f}")
        c3.metric("Search radius / window", f"{radius_km} km / {days_after}d")

        fig_map = px.scatter_geo(
            sequence_df, lat="latitude", lon="longitude", size="magnitude", color="days_since_mainshock",
            hover_name="place", projection="natural earth", title="Aftershock locations",
            color_continuous_scale="Viridis",
        )
        fig_map.add_scattergeo(lat=[mainshock["latitude"]], lon=[mainshock["longitude"]], mode="markers",
                                marker=dict(size=18, color="red", symbol="star"), name="Mainshock")
        fig_map.update_geos(
            fitbounds="locations", showland=True, landcolor="#2b3137", showocean=True, oceancolor="#0d1b2a",
            showcoastlines=True, coastlinecolor="#8b949e", showcountries=True, countrycolor="#57606a",
            bgcolor="rgba(0,0,0,0)", lataxis_showgrid=True, lonaxis_showgrid=True,
            lataxis_gridcolor="#30363d", lonaxis_gridcolor="#30363d",
        )
        fig_map.update_layout(height=450, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_map, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            fig_mt = px.scatter(sequence_df, x="days_since_mainshock", y="magnitude",
                                 title="Magnitude vs. time since mainshock", labels={"days_since_mainshock": "Days since mainshock"})
            st.plotly_chart(fig_mt, width="stretch")
        with c2:
            cum = sequence_df.copy()
            cum["cumulative"] = range(1, len(cum) + 1)
            fig_cum = px.line(cum, x="days_since_mainshock", y="cumulative",
                               title="Cumulative aftershock count", labels={"days_since_mainshock": "Days since mainshock"})
            st.plotly_chart(fig_cum, width="stretch")

        st.subheader("Observed decay rate vs. the generic Omori-Utsu curve")
        st.caption(
            "Bars: actual observed aftershock rate, binned by day. Line: the generic-California "
            "model's predicted rate, both at the catalog's estimated completeness magnitude. Real sequences are noisy "
            "and often don't match the generic curve exactly — that gap is exactly why the Forecast "
            "tab updates the model with this sequence's own data."
        )
        decay_df = sequence_df[sequence_df["magnitude"] >= mc - 1e-9]
        bins = np.arange(0, max(decay_df["days_since_mainshock"].max(), 1) + 1, 1)
        counts, edges = np.histogram(decay_df["days_since_mainshock"], bins=bins)
        bin_centers = (edges[:-1] + edges[1:]) / 2
        predicted_rate = rate(bin_centers, mainshock["magnitude"], mc)
        fig_decay = go.Figure()
        fig_decay.add_trace(go.Bar(x=bin_centers, y=counts, name="Observed (per day)"))
        fig_decay.add_trace(go.Scatter(x=bin_centers, y=predicted_rate, name="Generic model prediction", line=dict(color="crimson")))
        fig_decay.update_layout(title=f"Aftershock rate, M ≥ {mc:.1f} (estimated completeness)", xaxis_title="Days since mainshock", yaxis_title="Events/day", height=400)
        st.plotly_chart(fig_decay, width="stretch")

with tab_forecast:
    st.subheader("Forecast: probability of further aftershocks")

    use_bayes = False
    a_grid = a_post = None
    if not sequence_df.empty and len(sequence_df) >= 3:
        use_bayes = st.checkbox(
            "Update the generic model with this sequence's own observed data (Bayesian update)",
            value=True,
        )
        if use_bayes:
            obs_count = len(complete_df)
            # The observation window is the period actually searched, not the time of the
            # last event found (that would shrink the window and overstate productivity).
            obs_t1, obs_t2 = 0.0, float(min(days_after, t_now_days))
            a_grid, a_post = posterior_over_a(obs_count, obs_t1, obs_t2, mainshock["magnitude"], mc)
            _trapz = getattr(np, "trapezoid", None) or np.trapz
            post_mean = _trapz(a_grid * a_post, a_grid)
            st.caption(
                f"Generic prior productivity a = {GENERIC_A:.2f} → updated posterior mean a = {post_mean:.2f} "
                f"based on {obs_count} observed aftershocks of M ≥ {mc:.1f}, the catalog's estimated "
                f"completeness magnitude. {'This sequence is *more* productive than the generic model expects.' if post_mean > GENERIC_A else 'This sequence is *less* productive than the generic model expects.'}"
            )
        if mc > min_mag_catalog + 0.3:
            st.info(
                f"**Catalog completeness:** the smallest aftershock found is M{sequence_df['magnitude'].min():.1f}, "
                f"well above the M{min_mag_catalog:.1f} search threshold, so the catalog is missing small events "
                f"here (common outside the US, Japan and Europe). The update counts only the "
                f"{len(complete_df)} events of M ≥ {mc:.1f} and compares them with what the model expects at "
                f"M ≥ {mc:.1f}. Counting from M{min_mag_catalog:.1f} would make the sequence look far less "
                f"productive than it really is."
            )
    else:
        st.caption("No sufficient real aftershock data to update the model — showing the generic-California prior forecast only.")

    windows = {
        "next 1 day": (t_now_days, t_now_days + 1),
        "next 7 days": (t_now_days, t_now_days + 7),
        "next 30 days": (t_now_days, t_now_days + 30),
    }
    base_thresholds = sorted({round(mainshock["magnitude"] - d, 1) for d in [1, 2, 3]} | {round(mainshock["magnitude"], 1)})
    base_thresholds = [m for m in base_thresholds if m >= 2.5]

    rows = forecast(
        mainshock["magnitude"], t_now_days, windows, base_thresholds,
        a_grid=a_grid if use_bayes else None, a_posterior=a_post if use_bayes else None,
    )
    fdf = pd.DataFrame([r.__dict__ for r in rows])

    pivot = fdf.pivot(index="mag_threshold", columns="window_label", values="prob_mean")
    pivot = pivot[[w for w in windows.keys() if w in pivot.columns]] * 100
    pivot.index = [f"M ≥ {m:.1f}" for m in pivot.index]
    st.markdown("**Probability of ≥1 aftershock at or above M, within each window** (mean over parameter uncertainty)")
    st.dataframe(
        pivot,
        width="stretch",
        column_config={
            w: st.column_config.ProgressColumn(w, format="%.1f%%", min_value=0, max_value=100)
            for w in pivot.columns
        },
    )

    focus_window = st.selectbox("Show detail + uncertainty for window:", list(windows.keys()), index=1)
    detail = fdf[fdf["window_label"] == focus_window].sort_values("mag_threshold")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=detail["mag_threshold"], y=detail["prob_mean"],
        error_y=dict(type="data", symmetric=False,
                      array=detail["prob_high"] - detail["prob_mean"],
                      arrayminus=detail["prob_mean"] - detail["prob_low"]),
        marker_color="indianred",
    ))
    fig_bar.update_layout(title=f"P(≥1 aftershock ≥ M) — {focus_window}, with 90% interval",
                           xaxis_title="Magnitude threshold", yaxis_title="Probability", yaxis_tickformat=".0%", height=400)
    st.plotly_chart(fig_bar, width="stretch")

    biggest_risk = detail[detail["mag_threshold"] >= mainshock["magnitude"] - 1].sort_values("mag_threshold", ascending=False)
    if not biggest_risk.empty:
        top = biggest_risk.iloc[0]
        st.warning(
            f"Over the **{focus_window}**, the model estimates a **{top['prob_mean']:.1%}** chance "
            f"(90% interval: {top['prob_low']:.1%}-{top['prob_high']:.1%}) of at least one M≥{top['mag_threshold']} aftershock."
        )

with tab_about:
    st.markdown(
        f"""
### The real-world problem

After a major earthquake, the public and emergency managers immediately
ask "is a bigger one coming?" Vague answers ("aftershocks are possible")
aren't useful for decisions like whether to re-enter damaged buildings or
how long to keep emergency shelters open. The USGS's actual answer to this
is a calibrated statistical forecast, published on real earthquake event
pages — this app reproduces the same statistical engine.

### The model

**Modified Omori-Utsu law** (Reasenberg & Jones, 1989, *Science*):
aftershock rate at time *t* days after a mainshock of magnitude M_main,
for aftershocks ≥ magnitude M:

    λ(t) = 10^(a + b·(M_main − M)) / (t + c)^p

Integrating λ(t) over a forecast window gives the expected count N, and
assuming a Poisson process, P(≥1 event) = 1 − e⁻ᴺ.

**Generic California parameters** (Reasenberg & Jones, 1989): a = {GENERIC_A},
b = {GENERIC_B}, c = {GENERIC_C}, p = {GENERIC_P}.

**Bayesian update**: productivity varies a lot from one sequence to the
next. The prior on `a` has σ = {GENERIC_A_SIGMA}, a round value for the
between-sequence spread reported by Page et al. (2016), so with no data the
forecast intervals are wide on purpose. When real aftershock data is available for the
selected sequence, this app computes a posterior distribution over `a` by
combining that generic prior with a Poisson likelihood from the *observed*
aftershock count — the same idea the USGS's own Operational Aftershock
Forecasting system uses to adapt the generic model to a specific,
unfolding sequence rather than relying on the generic parameters alone.

### Data sources

- Earthquake catalog & significant-earthquake list: [USGS Earthquake Catalog API](https://earthquake.usgs.gov/fdsnws/event/1/)
  and [GeoJSON feeds](https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php) — free, no API key.
- Model & parameters: Reasenberg, P.A. & Jones, L.M. (1989). "Earthquake
  Hazard After a Mainshock in California." *Science*, 243, 1173-1176;
  Page, M.T. et al. (2016). "Three ingredients for improved global aftershock
  forecasts." *Seismological Research Letters*, 87(6);
  Hardebeck, J.L. et al. (2019). "Updated California aftershock parameters."
  *Seismological Research Letters*, doi:10.1785/0220180240; [USGS Operational Aftershock Forecasting background](https://earthquake.usgs.gov/data/oaf/background.php).

### Honest limitations (read before presenting this)

1. **The generic parameters are California-calibrated.** Aftershock
   productivity genuinely varies by tectonic region — applying California's
   generic model to, say, a subduction-zone earthquake in Japan or
   Indonesia is a real simplification. The real USGS system uses different
   generic parameter sets for different tectonic regions (Page et al.,
   2016); this prototype uses one generic set everywhere and *relies on
   the Bayesian update* to correct toward the real sequence's behavior
   once data exists — which is a partial fix, not a complete one.
2. **Catalog completeness varies by region.** Areas with sparser seismic
   monitoring record fewer small aftershocks than actually occurred. The
   app estimates the completeness magnitude Mc (maximum curvature + 0.2
   with 20 or more events, otherwise the smallest magnitude found) and
   updates the model only with events of M ≥ Mc. With very few events this
   estimate is rough, and right after a large mainshock the catalog also
   misses events that overlap in time.
3. **This models statistical clustering, not physical rupture mechanics.**
   It says nothing about which specific fault will slip next.
4. **This is a methodology demonstration**, not the official USGS forecast
   — for real decisions, always defer to the official aftershock forecast
   on the USGS event page itself.
        """
    )

st.divider()
st.caption(
    "Prototype build. Real next steps: region-specific generic parameter sets (Page et al. 2016), "
    "full ETAS spatial modeling instead of a radius cutoff, and proper catalog-completeness (Mc) estimation."
)
