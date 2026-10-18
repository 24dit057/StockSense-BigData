
import os
import pandas as pd
import yfinance as yf

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

# Project folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Indian stock symbols
TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
]

print("Downloading historical Indian stock data...")

all_stocks = []

for ticker in TICKERS:
    print(f"Downloading {ticker}...")

    data = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        print(f"Warning: No data received for {ticker}")
        continue

    # Handle possible multi-level column names
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    if "Datetime" in data.columns:
        data = data.rename(columns={"Datetime": "Date"})

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    data = data[required].copy()
    data["Symbol"] = ticker

    all_stocks.append(data)

if not all_stocks:
    raise RuntimeError(
        "No stock data downloaded. Check your internet connection "
        "and try again."
    )

# Combine all stocks and save the original dataset
combined = pd.concat(all_stocks, ignore_index=True)
combined.to_csv(
    os.path.join(BASE_DIR, "stocks.csv"),
    index=False
)

print(f"Downloaded {len(combined)} stock records.")
print("Starting Apache Spark...")

# Create a Spark session
spark = (
    SparkSession.builder
    .appName("StockSenseAnalytics")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Read the CSV using Spark
df = spark.read.csv(
    os.path.join(BASE_DIR, "stocks.csv"),
    header=True,
    inferSchema=True
)

df = (
    df.withColumn("Date", F.to_date("Date"))
    .withColumn("Close", F.col("Close").cast("double"))
    .withColumn("Volume", F.col("Volume").cast("long"))
    .filter(
        F.col("Date").isNotNull()
        & F.col("Close").isNotNull()
        & (F.col("Close") > 0)
    )
)

# Calculate the previous closing price for each stock
window = Window.partitionBy("Symbol").orderBy("Date")

daily = (
    df.withColumn(
        "Previous_Close",
        F.lag("Close").over(window)
    )
    .withColumn(
        "Daily_Return_Pct",
        F.when(
            F.col("Previous_Close").isNotNull()
            & (F.col("Previous_Close") > 0),
            (
                (F.col("Close") - F.col("Previous_Close"))
                / F.col("Previous_Close")
            ) * 100
        )
    )
)

# Find the first closing price for each stock
first_window = Window.partitionBy("Symbol").orderBy("Date")

first_prices = (
    daily.withColumn("rn", F.row_number().over(first_window))
    .filter(F.col("rn") == 1)
    .select(
        "Symbol",
        F.col("Close").alias("First_Close")
    )
)

# Find the latest closing price for each stock
latest_window = Window.partitionBy("Symbol").orderBy(
    F.col("Date").desc()
)

latest_prices = (
    daily.withColumn("rn", F.row_number().over(latest_window))
    .filter(F.col("rn") == 1)
    .select(
        "Symbol",
        F.col("Date").alias("Latest_Date"),
        F.col("Close").alias("Latest_Close")
    )
)

# Calculate summary metrics using Spark
metrics = daily.groupBy("Symbol").agg(
    F.count("*").alias("Records"),
    F.avg("Volume").alias("Average_Volume"),
    F.stddev_samp("Daily_Return_Pct").alias("Volatility_Pct")
)

summary = (
    metrics
    .join(first_prices, on="Symbol")
    .join(latest_prices, on="Symbol")
    .withColumn(
        "Total_Return_Pct",
        (
            (F.col("Latest_Close") - F.col("First_Close"))
            / F.col("First_Close")
        ) * 100
    )
    .orderBy("Symbol")
)

# Save Spark results as ordinary CSV files for Streamlit
daily_result = (
    daily.select(
        "Date",
        "Symbol",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Previous_Close",
        "Daily_Return_Pct"
    )
    .orderBy("Date", "Symbol")
    .toPandas()
)

summary_result = summary.toPandas()

daily_result.to_csv(
    os.path.join(OUTPUT_DIR, "daily_analysis.csv"),
    index=False
)

summary_result.to_csv(
    os.path.join(OUTPUT_DIR, "stock_summary.csv"),
    index=False
)

print("\nStock summary:")
summary.show(truncate=False)

print("\nAnalysis completed successfully!")
print("Original data: stocks.csv")
print("Dashboard data: analysis_output folder")

spark.stop()
