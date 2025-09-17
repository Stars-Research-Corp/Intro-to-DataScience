"""
Walk-along Project: Stock Data Analysis (Beginner Edition)

How to run this in JupyterLite (Pyodide kernel):
1. Open JupyterLite in your browser (no install needed).
2. Create a new notebook using the **Python (Pyodide)** kernel.
3. Upload your dataset file (e.g., stock_data_july_2025.csv) into the notebook’s working directory.
   - In JupyterLite: click the folder icon on the left → upload file.
4. Copy this code into a notebook cell.
5. Run the cell by pressing selecting the entire code and **Shift + Enter**.
   - If you change the dataset name or column names, update them in the code.
6. Explore outputs: tables will print below, and charts will show up inline.

What you’ll learn (in short):
- Pandas DataFrame basics (loading, inspecting, sorting)
- Feature engineering: creating new useful columns from raw data (e.g., Daily Change, moving averages)
- Vectorization: using whole-column math for speed and clarity (no loops)
- Simple visualization with matplotlib
- A tiny “data story” summary

Expected CSV columns (example):
Date, Open Price, Close Price, High Price, Low Price, Volume

Tip: If your column names differ, update them below.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1) Load data
# ----------------------------
# parse_dates converts text to real datetime objects; dayfirst=False assumes YYYY-MM-DD or MM/DD/YYYY
# na_values turns obvious missing tokens into NaN so Pandas can handle them
df = pd.read_csv(
    "stock_data_july_2025.csv",
    parse_dates=["Date"],
    na_values=["", "NA", "N/A", "null"]
)

# ----------------------------
# 2) Quick sanity checks
# ----------------------------
print("First 5 rows:")
print(df.head(), "\n")

print("Column info & data types:")
print(df.info(), "\n")

print("Basic stats for numeric columns:")
print(df.describe(numeric_only=True), "\n")

# Optional: Drop rows with critical missing values (keeps things simple for beginners)
needed = ["Date", "Open Price", "Close Price"]
df = df.dropna(subset=needed)

# ----------------------------
# 3) Sort by time (always good practice for time series)
# ----------------------------
df = df.sort_values(by="Date").reset_index(drop=True)

# ----------------------------
# 4) Feature engineering (ML concept!)
# ----------------------------
# - Feature engineering means creating extra columns that may capture useful patterns.
# - Vectorization: below we do math on entire columns at once (fast and clean), no loops needed.
# - Efficiency tip: engineered features can often *replace* raw columns (e.g., 'Daily Range' instead
#   of both 'High' and 'Low'), letting you drop unnecessary data. This reduces memory usage and
#   helps models focus only on the most relevant signals.

# Daily price change: how much the stock moved from open to close each day
df["Daily Change"] = df["Close Price"] - df["Open Price"]

# 5-day moving average of the closing price (smooths noise; a classic technical feature)
# rolling(...).mean() computes the average over sliding windows of length 5
df["MA_5"] = df["Close Price"].rolling(window=5, min_periods=1).mean()

# ----------------------------
# 5) Plotting (exploration)
# ----------------------------
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Close Price"], label="Closing Price")
plt.plot(df["Date"], df["MA_5"], label="5-Day MA", linestyle="--")
plt.xlabel("Date"); plt.ylabel("Price"); plt.title("Closing Price vs 5-Day Moving Average")
plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(12, 4))
plt.bar(df["Date"], df["Daily Change"])
plt.xlabel("Date"); plt.ylabel("Close - Open")
plt.title("Daily Price Changes")
plt.grid(True); plt.show()

# ----------------------------
# 6) Tiny data story (simple summary metrics)
# ----------------------------
avg_change = df["Daily Change"].mean()
max_close  = df["Close Price"].max()
min_close  = df["Close Price"].min()

print(f"Average daily price change: ${avg_change:.2f}")
print(f"Maximum closing price: ${max_close:.2f}")
print(f"Minimum closing price: ${min_close:.2f}")
print("\nInsight: The 5-day moving average smooths short-term wiggles to reveal trend direction.")
