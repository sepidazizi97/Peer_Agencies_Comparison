"""
BFT Fixed-Route Peer Dashboard

Install dependencies:
    pip install streamlit pandas plotly openpyxl

Run:
    streamlit run bft_peer_dashboard.py

Upload Peer Comparison(2).xlsx in the dashboard, or place it beside this file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BFT_NAME = "Ben Franklin Transit"
FIXED_ROUTE_BUS_MODES = ["MB", "CB", "RB"]
METRIC_SHEETS = ["UPT", "VRM", "VRH", "VOMS"]

METRIC_LABELS = {
    "UPT": "Unlinked Passenger Trips",
    "VRM": "Vehicle Revenue Miles",
    "VRH": "Vehicle Revenue Hours",
    "VOMS": "Average Monthly VOMS",
    "UPT_per_VRH": "Boardings per Revenue Hour",
    "UPT_per_VRM": "Boardings per Revenue Mile",
    "Speed": "Revenue Miles per Revenue Hour",
    "UPT_per_VOMS": "Boardings per Average VOMS",
}

SHORT_NAMES = {
    "Whatcom Transportation Authority": "Whatcom",
    "Corpus Christi Regional Transportation Authority": "Corpus Christi",
    "Santa Cruz Metropolitan Transit District": "Santa Cruz METRO",
    "Mass Transportation Authority": "Flint MTA",
    "Marin County Transit District": "Marin Transit",
    "City of Tallahassee": "StarMetro",
    "Des Moines Area Regional Transit Authority": "Des Moines DART",
    "Lexington Transit Authority": "Lextran",
    "Kitsap County Public Transportation Benefit Area Authority": "Kitsap Transit",
    "Southeastern Regional Transit Authority": "Southeastern RTA",
    "Lane Transit District": "Lane Transit",
    "Ben Franklin Transit": "BFT",
}


st.set_page_config(
    page_title="BFT Fixed-Route Peer Dashboard",
    page_icon="🚌",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
      [data-testid="stMetric"] {
        background: #F4F8FB;
        border: 1px solid #D7E3EC;
        border-radius: 10px;
        padding: 12px 14px;
      }
      [data-testid="stMetricLabel"] {color: #003B71;}
      h1, h2, h3 {color: #003B71;}
      .small-note {color: #52697A; font-size: 0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_agency(series: pd.Series) -> pd.Series:
    """Normalize agency labels without changing their identity."""
    return series.astype("string").str.strip()


def parse_month_column(column) -> pd.Timestamp | None:
    """Return a month-start timestamp for Excel headers such as 1/2023."""
    if isinstance(column, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(column).to_period("M").to_timestamp()
    parsed = pd.to_datetime(str(column).strip(), errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_period("M").to_timestamp()


@st.cache_data(show_spinner=False)
def read_source(file_source) -> dict[str, pd.DataFrame]:
    """Read the four required NTD metric sheets."""
    book = pd.ExcelFile(file_source)
    missing = [name for name in METRIC_SHEETS if name not in book.sheet_names]
    if missing:
        raise ValueError(f"Missing required sheet(s): {', '.join(missing)}")
    return {name: pd.read_excel(book, sheet_name=name) for name in METRIC_SHEETS}


def metric_long(raw: pd.DataFrame, metric: str, modes: list[str]) -> pd.DataFrame:
    """Convert one wide sheet to clean agency-month observations."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required = ["Agency", "Mode/Type of Service Status", "Mode", "TOS"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{metric} sheet is missing: {', '.join(missing)}")

    df["Agency"] = clean_agency(df["Agency"])
    df["Mode"] = df["Mode"].astype("string").str.strip().str.upper()
    df["TOS"] = df["TOS"].astype("string").str.strip().str.upper()
    status = df["Mode/Type of Service Status"].astype("string").str.strip()

    # Active fixed-route bus only. This excludes ferry, demand response,
    # vanpool, and inactive legacy records.
    df = df.loc[status.eq("Active") & df["Mode"].isin(modes)].copy()

    date_map = {c: parse_month_column(c) for c in df.columns}
    date_columns = [c for c, parsed in date_map.items() if parsed is not None]
    date_columns = [c for c in date_columns if date_map[c] >= pd.Timestamp("2023-01-01")]
    if not date_columns:
        raise ValueError(f"No month columns beginning in 2023 were found in {metric}.")

    long = df.melt(
        id_vars=["Agency", "Mode", "TOS"],
        value_vars=date_columns,
        var_name="SourceMonth",
        value_name=metric,
    )
    long["Date"] = long["SourceMonth"].map(date_map)
    long[metric] = pd.to_numeric(long[metric], errors="coerce")
    long = long.drop(columns="SourceMonth")

    # DO and PT are summed here. min_count=1 preserves a missing month when
    # every applicable mode/TOS record is missing.
    agency_month = (
        long.groupby(["Agency", "Date"], as_index=False)[metric]
        .sum(min_count=1)
        .sort_values(["Agency", "Date"])
    )
    return agency_month


def service_mix(raw: pd.DataFrame, modes: list[str]) -> pd.DataFrame:
    """List the active mode/TOS combinations included for each agency."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df["Agency"] = clean_agency(df["Agency"])
    df["Mode"] = df["Mode"].astype("string").str.strip().str.upper()
    df["TOS"] = df["TOS"].astype("string").str.strip().str.upper()
    status = df["Mode/Type of Service Status"].astype("string").str.strip()
    df = df.loc[status.eq("Active") & df["Mode"].isin(modes)].copy()
    df["Mode/TOS"] = df["Mode"] + "/" + df["TOS"]
    return (
        df.groupby("Agency")["Mode/TOS"]
        .agg(lambda s: ", ".join(sorted(set(s.dropna()))))
        .rename("Included Service")
        .reset_index()
    )


@st.cache_data(show_spinner=False)
def build_panel(sheets: dict[str, pd.DataFrame], mode_scope: str):
    modes = ["MB"] if mode_scope == "Motor Bus only (MB)" else FIXED_ROUTE_BUS_MODES
    metric_frames = [metric_long(sheets[m], m, modes) for m in METRIC_SHEETS]

    panel = metric_frames[0]
    for frame in metric_frames[1:]:
        panel = panel.merge(frame, on=["Agency", "Date"], how="outer")

    panel = panel.sort_values(["Agency", "Date"]).reset_index(drop=True)
    panel["Short Agency"] = panel["Agency"].map(SHORT_NAMES).fillna(panel["Agency"])
    panel["UPT_per_VRH"] = panel["UPT"] / panel["VRH"].replace(0, np.nan)
    panel["UPT_per_VRM"] = panel["UPT"] / panel["VRM"].replace(0, np.nan)
    panel["Speed"] = panel["VRM"] / panel["VRH"].replace(0, np.nan)
    panel["UPT_per_VOMS"] = panel["UPT"] / panel["VOMS"].replace(0, np.nan)

    mix = service_mix(sheets["UPT"], modes)
    return panel, mix


def period_summary(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    selected = panel.loc[panel["Date"].between(start, end)].copy()
    expected_months = len(pd.date_range(start, end, freq="MS"))

    summary = (
        selected.groupby(["Agency", "Short Agency"], as_index=False)
        .agg(
            UPT=("UPT", lambda x: x.sum(min_count=1)),
            VRM=("VRM", lambda x: x.sum(min_count=1)),
            VRH=("VRH", lambda x: x.sum(min_count=1)),
            VOMS=("VOMS", "mean"),
            UPT_Months=("UPT", "count"),
            VRM_Months=("VRM", "count"),
            VRH_Months=("VRH", "count"),
            VOMS_Months=("VOMS", "count"),
        )
    )
    summary["UPT_per_VRH"] = summary["UPT"] / summary["VRH"].replace(0, np.nan)
    summary["UPT_per_VRM"] = summary["UPT"] / summary["VRM"].replace(0, np.nan)
    summary["Speed"] = summary["VRM"] / summary["VRH"].replace(0, np.nan)
    summary["UPT_per_VOMS"] = summary["UPT"] / summary["VOMS"].replace(0, np.nan)
    summary["Complete Months"] = summary[
        ["UPT_Months", "VRM_Months", "VRH_Months", "VOMS_Months"]
    ].min(axis=1)
    summary["Expected Months"] = expected_months
    summary["Complete"] = summary["Complete Months"].eq(expected_months)
    return summary


def metric_format(metric: str, value: float) -> str:
    if pd.isna(value):
        return "—"
    if metric in {"UPT", "VRM", "VRH"}:
        return f"{value:,.0f}"
    if metric == "VOMS":
        return f"{value:,.1f}"
    return f"{value:,.2f}"


def main():
    st.title("BFT Fixed-Route Peer Dashboard")
    st.markdown(
        "<div class='small-note'>2023–present monthly NTD comparison • "
        "DO and PT are combined • ferry, demand response, vanpool and inactive records are excluded</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.sidebar.file_uploader("Upload peer workbook", type=["xlsx"])
    local_file = Path(__file__).with_name("Peer Comparison(2).xlsx")
    file_source = uploaded if uploaded is not None else (local_file if local_file.exists() else None)
    if file_source is None:
        st.info("Upload Peer Comparison(2).xlsx using the sidebar to begin.")
        st.stop()

    try:
        sheets = read_source(file_source)
    except Exception as exc:
        st.error(f"The workbook could not be read: {exc}")
        st.stop()

    control_1, control_2, control_3 = st.columns([1.15, 1.25, 1.6])
    with control_1:
        mode_scope = st.selectbox(
            "Fixed-route definition",
            ["All fixed-route bus (MB + CB + RB)", "Motor Bus only (MB)"],
        )
    with control_2:
        period = st.selectbox(
            "Comparison period",
            ["2026 Q1", "2025", "2024", "2023", "Custom"],
        )
    with control_3:
        metric = st.selectbox(
            "Primary comparison metric",
            list(METRIC_LABELS),
            format_func=lambda x: METRIC_LABELS[x],
            index=4,
        )

    panel, mix = build_panel(sheets, mode_scope)

    period_dates = {
        "2026 Q1": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-01")),
        "2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-01")),
        "2024": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-01")),
        "2023": (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-01")),
    }
    if period == "Custom":
        min_date = panel["Date"].min().date()
        max_date = panel["Date"].max().date()
        date_values = st.date_input(
            "Custom month range",
            value=(pd.Timestamp("2025-01-01").date(), max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if len(date_values) != 2:
            st.warning("Select both a start and end date.")
            st.stop()
        start = pd.Timestamp(date_values[0]).to_period("M").to_timestamp()
        end = pd.Timestamp(date_values[1]).to_period("M").to_timestamp()
    else:
        start, end = period_dates[period]

    summary = period_summary(panel, start, end).merge(mix, on="Agency", how="left")
    summary = summary.loc[summary[metric].notna()].copy()
    if summary.empty:
        st.warning("No comparable values are available for this selection.")
        st.stop()

    bft = summary.loc[summary["Agency"].eq(BFT_NAME)]
    peers = summary.loc[~summary["Agency"].eq(BFT_NAME)]
    if bft.empty:
        st.warning("Ben Franklin Transit is not present for this selection.")
        st.stop()

    bft_value = bft.iloc[0][metric]
    peer_median = peers[metric].median()
    difference = (bft_value / peer_median - 1) if pd.notna(peer_median) and peer_median != 0 else np.nan
    ascending = metric in {"Speed"}  # descriptive, not inherently better/worse
    ranked = summary.sort_values(metric, ascending=False).reset_index(drop=True)
    bft_rank = int(ranked.index[ranked["Agency"].eq(BFT_NAME)][0]) + 1
    complete_peers = int(summary["Complete"].sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"BFT: {METRIC_LABELS[metric]}", metric_format(metric, bft_value))
    k2.metric("Peer median", metric_format(metric, peer_median))
    k3.metric("BFT vs. peer median", "—" if pd.isna(difference) else f"{difference:+.1%}")
    k4.metric("BFT rank", f"{bft_rank} of {len(summary)}", f"{complete_peers} complete")

    incomplete = summary.loc[~summary["Complete"], ["Short Agency", "Complete Months", "Expected Months"]]
    if not incomplete.empty:
        names = ", ".join(incomplete["Short Agency"].tolist())
        st.warning(
            f"Incomplete all-metric coverage for the selected period: {names}. "
            "UPT and VRM extend through June 2026, but BFT VRH and VOMS currently stop in March 2026."
        )

    left, right = st.columns([1.02, 1])

    with left:
        plot_data = summary.sort_values(metric, ascending=True).copy()
        plot_data["Color"] = np.where(plot_data["Agency"].eq(BFT_NAME), "BFT", "Peers")
        bar = px.bar(
            plot_data,
            x=metric,
            y="Short Agency",
            orientation="h",
            color="Color",
            color_discrete_map={"BFT": "#FFC72C", "Peers": "#005F9E"},
            labels={metric: METRIC_LABELS[metric], "Short Agency": ""},
            title=f"{METRIC_LABELS[metric]} — {start:%b %Y} to {end:%b %Y}",
        )
        bar.add_vline(x=peer_median, line_dash="dash", line_color="#E3A008")
        bar.update_layout(showlegend=False, height=500, margin=dict(l=5, r=15, t=55, b=10))
        st.plotly_chart(bar, use_container_width=True)

    with right:
        trend = panel.loc[panel["Date"].between(start, end) & panel[metric].notna()].copy()
        fig = go.Figure()
        for agency, group in trend.loc[~trend["Agency"].eq(BFT_NAME)].groupby("Short Agency"):
            fig.add_trace(
                go.Scatter(
                    x=group["Date"], y=group[metric], mode="lines",
                    name=agency, line=dict(color="#A8BBC9", width=1),
                    opacity=0.65, hovertemplate=f"{agency}<br>%{{x|%b %Y}}<br>%{{y:,.2f}}<extra></extra>",
                )
            )
        bft_trend = trend.loc[trend["Agency"].eq(BFT_NAME)]
        fig.add_trace(
            go.Scatter(
                x=bft_trend["Date"], y=bft_trend[metric], mode="lines+markers",
                name="BFT", line=dict(color="#003B71", width=4),
                marker=dict(color="#FFC72C", size=8, line=dict(color="#003B71", width=1)),
                hovertemplate="BFT<br>%{x|%b %Y}<br>%{y:,.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"Monthly Trend: {METRIC_LABELS[metric]}",
            height=500, margin=dict(l=5, r=15, t=55, b=10),
            yaxis_title=METRIC_LABELS[metric], xaxis_title="", showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    scatter_data = summary.dropna(subset=["VRH", "UPT"]).copy()
    scatter = px.scatter(
        scatter_data,
        x="VRH",
        y="UPT",
        size="VOMS",
        color=np.where(scatter_data["Agency"].eq(BFT_NAME), "BFT", "Peers"),
        color_discrete_map={"BFT": "#FFC72C", "Peers": "#2563EB"},
        hover_name="Short Agency",
        hover_data={"VRH": ":,.0f", "UPT": ":,.0f", "VOMS": ":,.1f"},
        labels={"VRH": "Vehicle Revenue Hours", "UPT": "Unlinked Passenger Trips", "color": ""},
        title="Service Supplied vs. Ridership (bubble size = average monthly VOMS)",
    )
    scatter.update_layout(height=460, showlegend=False, margin=dict(l=5, r=15, t=55, b=10))
    st.plotly_chart(scatter, use_container_width=True)

    st.subheader("Agency Comparison Table")
    display = summary[
        [
            "Short Agency", "Included Service", "UPT", "VRM", "VRH", "VOMS",
            "UPT_per_VRH", "UPT_per_VRM", "Speed", "UPT_per_VOMS",
            "Complete Months", "Expected Months",
        ]
    ].copy()
    display.columns = [
        "Agency", "Included Mode/TOS", "UPT", "VRM", "VRH", "Avg. VOMS",
        "UPT/VRH", "UPT/VRM", "VRM/VRH", "UPT/Avg. VOMS",
        "Complete Months", "Expected Months",
    ]
    display = display.sort_values("UPT/VRH", ascending=False)
    st.dataframe(
        display.style.format(
            {
                "UPT": "{:,.0f}", "VRM": "{:,.0f}", "VRH": "{:,.0f}",
                "Avg. VOMS": "{:,.1f}", "UPT/VRH": "{:,.2f}",
                "UPT/VRM": "{:,.2f}", "VRM/VRH": "{:,.2f}",
                "UPT/Avg. VOMS": "{:,.0f}",
            },
            na_rep="—",
        ),
        use_container_width=True,
        hide_index=True,
        height=455,
    )

    st.caption(
        "Method: active MB, CB and RB records are included by default; DO and PT are summed before "
        "ratios are calculated. VOMS is averaged across months. For an apples-to-apples BFT check, "
        "switch the fixed-route definition to Motor Bus only (MB)."
    )


if __name__ == "__main__":
    main()
