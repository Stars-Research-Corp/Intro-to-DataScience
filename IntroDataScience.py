# Walk-along Python Project: Stock Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv('stock_data_july_2025.csv')

# Step 2: Inspect the data
print("First 5 rows of dataset:")
print(df.head())

print("\nData summary:")
print(df.info())

print("\nBasic statistics:")
print(df.describe())

# Step 3: Explore key columns
# Note: Column names are 'Open Price', 'Close Price', 'High Price', 'Low Price', etc.

# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date just in case
df = df.sort_values(by='Date')

# Step 4: Calculate daily price change (Close Price - Open Price)
df['Daily Change'] = df['Close Price'] - df['Open Price']

# Step 5: Calculate moving average of closing price (window = 5 days)
df['MA_5'] = df['Close Price'].rolling(window=5).mean()

# Step 6: Plot closing price and moving average
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close Price'], label='Closing Price')
plt.plot(df['Date'], df['MA_5'], label='5-Day Moving Average', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Closing Price and 5-Day Moving Average')
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Plot daily price changes as a bar chart
plt.figure(figsize=(12, 4))
plt.bar(df['Date'], df['Daily Change'], color='orange')
plt.xlabel('Date')
plt.ylabel('Daily Change (Close - Open)')
plt.title('Daily Price Changes')
plt.grid(True)
plt.show()

# Step 8: Simple data story takeaway printout
avg_change = np.mean(df['Daily Change'])
max_close = np.max(df['Close Price'])
min_close = np.min(df['Close Price'])

print(f"Average daily price change: ${avg_change:.2f}")
print(f"Maximum closing price: ${max_close:.2f}")
print(f"Minimum closing price: ${min_close:.2f}")

print("\nInsight: The 5-day moving average smooths out short-term fluctuations, giving a clearer trend direction.")