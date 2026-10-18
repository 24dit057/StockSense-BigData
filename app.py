
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "analysis_output"

DAILY_FILE = DATA_DIR / "daily_analysis.csv"
SUMMARY_FILE = DATA_DIR / "stock_summary.csv"

st.set_page_config(
    page_title="StockSense",
    page_icon="📈",
    layout="wide"
)

st.title("📈 StockSense")
st.caption("Indian Stock Market Analytics powered by Apache Spark")

if not DAILY_FILE.exists() or not SUMMARY_FILE.exists():
    st.error("Analysis files are missing. Run this command first:")
    st.code("python analyze_stocks.py")
    st.stop()

daily = pd.read_csv(DAILY_FILE, parse_dates=["Date"])
summary = pd.read_csv(SUMMARY_FILE)

# Sidebar filters
st.sidebar.header("Dashboard Filters")

symbols = sorted(daily["Symbol"].dropna().unique())

selected_symbols = st.sidebar.multiselect(
    "Select stocks",
    symbols,
    default=symbols
)

if not selected_symbols:
    st.warning("Select at least one stock.")
    st.stop()

filtered = daily[daily["Symbol"].isin(selected_symbols)].copy()
filtered_summary = summary[
    summary["Symbol"].isin(selected_symbols)
].copy()

min_date = filtered["Date"].min().date()
max_date = filtered["Date"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[
        (filtered["Date"].dt.date >= start_date)
        & (filtered["Date"].dt.date <= end_date)
    ]

if filtered.empty:
    st.warning("No data available for this selection.")
    st.stop()

# KPI cards
best_stock = filtered_summary.sort_values(
    "Total_Return_Pct",
    ascending=False
).iloc[0]

c1, c2, c3, c4 = st.columns(4)

c1.metric("Stocks Selected", len(selected_symbols))
c2.metric("Records Displayed", f"{len(filtered):,}")
c3.metric(
    "Average Daily Volume",
    f"{filtered['Volume'].mean():,.0f}"
)
c4.metric(
    "Best Historical Return",
    f"{best_stock['Total_Return_Pct']:.2f}%",
    help="Return over the full downloaded history, not just the selected dates."
)

st.divider()

# Closing price chart
st.subheader("Closing Price Comparison")

price_chart = filtered.pivot_table(
    index="Date",
    columns="Symbol",
    values="Close"
)

st.line_chart(price_chart)

# Daily returns chart
st.subheader("Daily Returns (%)")

return_chart = filtered.pivot_table(
    index="Date",
    columns="Symbol",
    values="Daily_Return_Pct"
)

st.line_chart(return_chart)

# Trading volume chart
st.subheader("Trading Volume")

volume_chart = filtered.pivot_table(
    index="Date",
    columns="Symbol",
    values="Volume"
)

st.line_chart(volume_chart)

# Summary table
st.subheader("Stock Performance Summary")

display_summary = filtered_summary[
    [
        "Symbol",
        "Records",
        "Latest_Date",
        "Latest_Close",
        "Average_Volume",
        "Volatility_Pct",
        "Total_Return_Pct"
    ]
].copy()

display_summary = display_summary.rename(columns={
    "Symbol": "Stock",
    "Records": "Trading Days",
    "Latest_Date": "Latest Data Date",
    "Latest_Close": "Latest Close (₹)",
    "Average_Volume": "Average Volume",
    "Volatility_Pct": "Daily Volatility (%)",
    "Total_Return_Pct": "Historical Return (%)"
})

st.dataframe(
    display_summary.round(2),
    use_container_width=True,
    hide_index=True
)

st.subheader("Historical Data")

st.dataframe(
    filtered.sort_values("Date", ascending=False),
    use_container_width=True,
    hide_index=True
)

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Selected Data as CSV",
    data=csv_data,
    file_name="stocksense_data.csv",
    mime="text/csv"
)

st.caption(
    "Educational analytics project. Historical returns and volatility "
    "do not guarantee future performance."
)
