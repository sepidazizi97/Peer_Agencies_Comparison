"""BFT fixed-route peer dashboard.

The workbook loads from the project's GitHub repository. Set PEER_DATA_URL in
the hosting service only if you want to override the default address.
"""

from io import BytesIO
import os
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BFT_NAME = "Ben Franklin Transit"
FIXED_ROUTE_MODES = ["MB", "CB", "RB"]
METRIC_SHEETS = ["UPT", "VRM", "VRH", "VOMS"]
GITHUB_EXCEL_URL = os.getenv(
    "PEER_DATA_URL",
    "https://raw.githubusercontent.com/sepidazizi97/Peer_Agencies_Comparison/main/Peer%20Comparison.xlsx",
)
METRIC_LABELS = {
    "UPT": "Unlinked Passenger Trips",
    "VRM": "Vehicle Revenue Miles",
    "VRH": "Vehicle Revenue Hours",
    "VOMS": "Average Monthly VOMS",
    "UPT_per_VRH": "Boardings per Revenue Hour",
    "UPT_per_VRM": "Boardings per Revenue Mile",
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
    BFT_NAME: "BFT",
}

st.set_page_config(page_title="BFT Fixed-Route Peers", page_icon="🚌", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top:1.25rem;padding-bottom:2rem}
    [data-testid="stMetric"] {background:#F4F8FB;border:1px solid #D7E3EC;
      border-radius:10px;padding:12px 14px}
    [data-testid="stMetricLabel"],h1,h2,h3 {color:#003B71}
    .note {color:#52697A;font-size:.92rem;margin-bottom:1rem}
    .overview-hero {background:linear-gradient(120deg,#003B71 0%,#005F9E 70%,#026873 100%);
      border-radius:16px;padding:1.6rem 1.8rem;margin-bottom:1.25rem;color:white;
      box-shadow:0 8px 22px rgba(0,59,113,.16)}
    .overview-hero h1 {color:white;margin:0 0 .35rem 0;font-size:2.25rem}
    .overview-hero p {margin:0;color:#E8F3FA;font-size:1rem}
    .eyebrow {color:#FFC72C;font-size:.78rem;font-weight:700;letter-spacing:.12em;
      text-transform:uppercase;margin-bottom:.4rem}
    .section-label {font-size:1.05rem;font-weight:700;color:#003B71;margin-top:1.2rem;
      margin-bottom:.25rem}
    </style>""",
    unsafe_allow_html=True,
)


def clean_agency(values):
    return values.astype("string").str.strip()


def month_header(value):
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).to_period("M").to_timestamp()
    parsed = pd.to_datetime(str(value).strip(), errors="coerce")
    return None if pd.isna(parsed) else parsed.to_period("M").to_timestamp()


@st.cache_data(ttl=3600, show_spinner=False)
def read_source(url):
    request = Request(url, headers={"User-Agent": "BFT-peer-dashboard/1.0"})
    with urlopen(request, timeout=45) as response:
        book = pd.ExcelFile(BytesIO(response.read()))
    required = ["Master", *METRIC_SHEETS]
    missing = [name for name in required if name not in book.sheet_names]
    if missing:
        raise ValueError(f"Missing sheet(s): {', '.join(missing)}")
    return {name: pd.read_excel(book, sheet_name=name) for name in required}


def fixed_route_rows(raw):
    df = raw.copy()
    df.columns = [str(column).strip() for column in df.columns]
    df["Agency"] = clean_agency(df["Agency"])
    df["Mode"] = df["Mode"].astype("string").str.strip().str.upper()
    df["TOS"] = df["TOS"].astype("string").str.strip().str.upper()
    status = df["Mode/Type of Service Status"].astype("string").str.strip()
    return df.loc[status.eq("Active") & df["Mode"].isin(FIXED_ROUTE_MODES)].copy()


def metric_long(raw, metric):
    df = fixed_route_rows(raw)
    date_map = {column: month_header(column) for column in df.columns}
    columns = [
        column for column, date in date_map.items()
        if date is not None and date >= pd.Timestamp("2023-01-01")
    ]
    if not columns:
        raise ValueError(f"No 2023-or-later month columns found in {metric}.")
    long = df.melt(
        id_vars=["Agency", "Mode", "TOS"], value_vars=columns,
        var_name="SourceMonth", value_name=metric,
    )
    long["Date"] = long["SourceMonth"].map(date_map)
    long[metric] = pd.to_numeric(long[metric], errors="coerce")
    # DO and PT are additive components and are summed before ratios.
    return (
        long.groupby(["Agency", "Date"], as_index=False)[metric]
        .sum(min_count=1).sort_values(["Agency", "Date"])
    )


@st.cache_data(show_spinner=False)
def build_panel(sheets):
    frames = [metric_long(sheets[name], name) for name in METRIC_SHEETS]
    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on=["Agency", "Date"], how="outer")
    panel = panel.sort_values(["Agency", "Date"]).reset_index(drop=True)
    panel["Short Agency"] = panel["Agency"].map(SHORT_NAMES).fillna(panel["Agency"])
    panel["UPT_per_VRH"] = panel["UPT"] / panel["VRH"].replace(0, np.nan)
    panel["UPT_per_VRM"] = panel["UPT"] / panel["VRM"].replace(0, np.nan)
    panel["UPT_per_VOMS"] = panel["UPT"] / panel["VOMS"].replace(0, np.nan)

    mix = fixed_route_rows(sheets["UPT"])
    mix["Mode/TOS"] = mix["Mode"] + "/" + mix["TOS"]
    mix = (
        mix.groupby("Agency")["Mode/TOS"]
        .agg(lambda values: ", ".join(sorted(set(values.dropna()))))
        .rename("Included Service").reset_index()
    )
    return panel, mix


@st.cache_data(show_spinner=False)
def build_profiles(raw):
    df = fixed_route_rows(raw)
    numeric = [
        "UZA SQ Miles", "UZA Population", "Service Area Population",
        "Service Area SQ Miles", "Last Closed Report Year", "Passenger Miles FY",
        "Unlinked Passenger Trips FY", "Fares FY", "Operating Expenses FY",
    ]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["Mode/TOS"] = df["Mode"] + "/" + df["TOS"]
    static = [
        "Organization Type", "HQ City", "HQ State", "UZA Name",
        "UZA SQ Miles", "UZA Population", "Service Area Population",
        "Service Area SQ Miles", "Last Closed Report Year",
    ]
    profiles = df.groupby("Agency", as_index=False)[static].first()
    mix = (
        df.groupby("Agency")["Mode/TOS"]
        .agg(lambda values: ", ".join(sorted(set(values.dropna()))))
        .rename("Fixed-Route Service").reset_index()
    )
    totals = df.groupby("Agency", as_index=False).agg(
        FY_UPT=("Unlinked Passenger Trips FY", lambda x: x.sum(min_count=1)),
        FY_Passenger_Miles=("Passenger Miles FY", lambda x: x.sum(min_count=1)),
        FY_Fares=("Fares FY", lambda x: x.sum(min_count=1)),
        FY_Operating_Expenses=("Operating Expenses FY", lambda x: x.sum(min_count=1)),
    )
    profiles = profiles.merge(mix, on="Agency").merge(totals, on="Agency")
    profiles["Short Agency"] = profiles["Agency"].map(SHORT_NAMES).fillna(profiles["Agency"])
    profiles["Service Area Density"] = (
        profiles["Service Area Population"] / profiles["Service Area SQ Miles"].replace(0, np.nan)
    )
    profiles["FY Cost per Trip"] = (
        profiles["FY_Operating_Expenses"] / profiles["FY_UPT"].replace(0, np.nan)
    )
    return profiles.sort_values("Agency").reset_index(drop=True)


def period_summary(panel, start, end):
    selected = panel.loc[panel["Date"].between(start, end)]
    expected = len(pd.date_range(start, end, freq="MS"))
    result = selected.groupby(["Agency", "Short Agency"], as_index=False).agg(
        UPT=("UPT", lambda x: x.sum(min_count=1)),
        VRM=("VRM", lambda x: x.sum(min_count=1)),
        VRH=("VRH", lambda x: x.sum(min_count=1)),
        VOMS=("VOMS", "mean"),
        UPT_Months=("UPT", "count"), VRM_Months=("VRM", "count"),
        VRH_Months=("VRH", "count"), VOMS_Months=("VOMS", "count"),
    )
    result["UPT_per_VRH"] = result["UPT"] / result["VRH"].replace(0, np.nan)
    result["UPT_per_VRM"] = result["UPT"] / result["VRM"].replace(0, np.nan)
    result["UPT_per_VOMS"] = result["UPT"] / result["VOMS"].replace(0, np.nan)
    result["Complete Months"] = result[
        ["UPT_Months", "VRM_Months", "VRH_Months", "VOMS_Months"]
    ].min(axis=1)
    result["Expected Months"] = expected
    result["Complete"] = result["Complete Months"].eq(expected)
    return result


def metric_format(metric, value):
    if pd.isna(value):
        return "—"
    if metric in {"UPT", "VRM", "VRH"}:
        return f"{value:,.0f}"
    return f"{value:,.1f}" if metric == "VOMS" else f"{value:,.2f}"


def overview_page(profiles):
    st.markdown(
        "<div class='overview-hero'><div class='eyebrow'>BFT Peer Benchmarking</div>"
        "<h1>Agency Overview</h1><p>Explore the communities, service areas, and "
        "operating scale of Ben Franklin Transit and its fixed-route peers.</p></div>",
        unsafe_allow_html=True,
    )
    options = profiles.assign(IsPeer=profiles["Agency"].ne(BFT_NAME)).sort_values(
        ["IsPeer", "Agency"]
    )["Agency"].tolist()
    selected = st.selectbox(
        "Select an agency", options, format_func=lambda x: SHORT_NAMES.get(x, x)
    )
    row = profiles.loc[profiles["Agency"].eq(selected)].iloc[0]
    st.markdown(
        "<div class='section-label'>Community &amp; Service Area</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric("Service population", f"{row['Service Area Population']:,.0f}")
    cols[1].metric("Service area (sq. mi.)", f"{row['Service Area SQ Miles']:,.0f}")
    cols[2].metric("Population density", f"{row['Service Area Density']:,.0f}/sq. mi.")
    cols[3].metric("UZA population", f"{row['UZA Population']:,.0f}")

    st.markdown(
        "<div class='section-label'>Latest Closed-Year Operating Context</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric("Fixed-route trips", f"{row['FY_UPT']:,.0f}")
    cols[1].metric("Passenger miles", f"{row['FY_Passenger_Miles']:,.0f}")
    cols[2].metric("Operating expense", f"${row['FY_Operating_Expenses']:,.0f}")
    cols[3].metric("Cost per trip", f"${row['FY Cost per Trip']:,.2f}")

    st.markdown(
        "<div class='section-label'>Peer Agency Reference</div>",
        unsafe_allow_html=True,
    )
    table = profiles[[
        "Short Agency", "HQ State", "UZA Name", "Service Area Population",
        "Service Area SQ Miles", "Service Area Density", "UZA Population",
        "Fixed-Route Service", "Last Closed Report Year",
    ]].copy()
    table.columns = [
        "Agency", "State", "Urbanized Area", "Service Population",
        "Service Area Sq. Mi.", "People/Sq. Mi.", "UZA Population",
        "Included Mode/TOS", "Report Year",
    ]
    st.dataframe(
        table.style.format({
            "Service Population": "{:,.0f}", "Service Area Sq. Mi.": "{:,.0f}",
            "People/Sq. Mi.": "{:,.0f}", "UZA Population": "{:,.0f}",
            "Report Year": "{:.0f}",
        }, na_rep="—"),
        use_container_width=True, hide_index=True, height=455,
    )


def performance_page(panel, mix):
    st.title("BFT Fixed-Route Performance Comparison")
    st.markdown(
        "<div class='note'>2023–present monthly NTD comparison • active MB, CB and RB "
        "• DO and PT combined • ferry and non-fixed-route modes excluded</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1, 1.6])
    with c1:
        period = st.selectbox("Comparison period", ["2026 Q1", "2025", "2024", "2023"])
    with c2:
        metric = st.selectbox(
            "Primary metric", list(METRIC_LABELS),
            format_func=lambda x: METRIC_LABELS[x], index=4,
        )
    periods = {
        "2026 Q1": ("2026-01-01", "2026-03-01"),
        "2025": ("2025-01-01", "2025-12-01"),
        "2024": ("2024-01-01", "2024-12-01"),
        "2023": ("2023-01-01", "2023-12-01"),
    }
    start, end = map(pd.Timestamp, periods[period])

    summary = period_summary(panel, start, end).merge(mix, on="Agency", how="left")
    summary = summary.loc[summary[metric].notna()].copy()
    bft = summary.loc[summary["Agency"].eq(BFT_NAME)]
    peers = summary.loc[~summary["Agency"].eq(BFT_NAME)]
    if summary.empty or bft.empty:
        st.warning("No comparable BFT values are available for this selection.")
        st.stop()
    bft_value = bft.iloc[0][metric]
    median = peers[metric].median()
    difference = bft_value / median - 1 if pd.notna(median) and median else np.nan
    ranked = summary.sort_values(metric, ascending=False).reset_index(drop=True)
    rank = int(ranked.index[ranked["Agency"].eq(BFT_NAME)][0]) + 1

    cols = st.columns(4)
    cols[0].metric(f"BFT: {METRIC_LABELS[metric]}", metric_format(metric, bft_value))
    cols[1].metric("Peer median", metric_format(metric, median))
    cols[2].metric("BFT vs. peer median", "—" if pd.isna(difference) else f"{difference:+.1%}")
    cols[3].metric("BFT rank", f"{rank} of {len(summary)}", f"{int(summary['Complete'].sum())} complete")
    incomplete = summary.loc[~summary["Complete"], "Short Agency"].tolist()
    if incomplete:
        st.warning(
            "Incomplete all-metric coverage: " + ", ".join(incomplete)
            + ". Use 2026 Q1 for the latest common BFT comparison."
        )

    left, right = st.columns([1.02, 1])
    with left:
        plot = summary.sort_values(metric).copy()
        plot["Series"] = np.where(plot["Agency"].eq(BFT_NAME), "BFT", "Peers")
        bar = px.bar(
            plot, x=metric, y="Short Agency", orientation="h", color="Series",
            color_discrete_map={"BFT": "#FFC72C", "Peers": "#005F9E"},
            labels={metric: METRIC_LABELS[metric], "Short Agency": ""},
            title=f"{METRIC_LABELS[metric]} — {start:%b %Y} to {end:%b %Y}",
        )
        bar.add_vline(x=median, line_dash="dash", line_color="#E3A008")
        bar.update_layout(showlegend=False, height=500, margin=dict(l=5, r=15, t=55, b=10))
        st.plotly_chart(bar, use_container_width=True)
    with right:
        trend = panel.loc[panel["Date"].between(start, end) & panel[metric].notna()]
        figure = go.Figure()
        for agency, group in trend.loc[~trend["Agency"].eq(BFT_NAME)].groupby("Short Agency"):
            figure.add_trace(go.Scatter(
                x=group["Date"], y=group[metric], mode="lines", name=agency,
                line=dict(color="#A8BBC9", width=1), opacity=.65,
                hovertemplate=f"{agency}<br>%{{x|%b %Y}}<br>%{{y:,.2f}}<extra></extra>",
            ))
        group = trend.loc[trend["Agency"].eq(BFT_NAME)]
        figure.add_trace(go.Scatter(
            x=group["Date"], y=group[metric], mode="lines+markers", name="BFT",
            line=dict(color="#003B71", width=4),
            marker=dict(color="#FFC72C", size=8, line=dict(color="#003B71", width=1)),
            hovertemplate="BFT<br>%{x|%b %Y}<br>%{y:,.2f}<extra></extra>",
        ))
        figure.update_layout(
            title=f"Monthly Trend: {METRIC_LABELS[metric]}", height=500,
            margin=dict(l=5, r=15, t=55, b=10),
            yaxis_title=METRIC_LABELS[metric], xaxis_title="", showlegend=False,
        )
        st.plotly_chart(figure, use_container_width=True)

    scatter_data = summary.dropna(subset=["VRH", "UPT"])
    scatter = px.scatter(
        scatter_data, x="VRH", y="UPT", size="VOMS",
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
    display = summary[[
        "Short Agency", "Included Service", "UPT", "VRM", "VRH", "VOMS",
        "UPT_per_VRH", "UPT_per_VRM", "UPT_per_VOMS",
        "Complete Months", "Expected Months",
    ]].copy()
    display.columns = [
        "Agency", "Included Mode/TOS", "UPT", "VRM", "VRH", "Avg. VOMS",
        "UPT/VRH", "UPT/VRM", "UPT/Avg. VOMS", "Complete Months", "Expected Months",
    ]
    display = display.sort_values("UPT/VRH", ascending=False)
    st.dataframe(
        display.style.format({
            "UPT": "{:,.0f}", "VRM": "{:,.0f}", "VRH": "{:,.0f}",
            "Avg. VOMS": "{:,.1f}", "UPT/VRH": "{:,.2f}",
            "UPT/VRM": "{:,.2f}", "UPT/Avg. VOMS": "{:,.0f}",
        }, na_rep="—"),
        use_container_width=True, hide_index=True, height=455,
    )


def main():
    st.sidebar.title("BFT Peer Comparison")
    page = st.sidebar.radio("Page", ["Agency Overview", "Performance Dashboard"])
    st.sidebar.caption("Data refreshes from GitHub every hour.")
    try:
        sheets = read_source(GITHUB_EXCEL_URL)
        panel, mix = build_panel(sheets)
        profiles = build_profiles(sheets["Master"])
    except Exception as exc:
        st.error(f"The GitHub workbook could not be loaded: {exc}")
        st.stop()
    overview_page(profiles) if page == "Agency Overview" else performance_page(panel, mix)


if __name__ == "__main__":
    main()
