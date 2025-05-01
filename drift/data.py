import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
import datetime
import os

# -- Set FRED API Key --
os.environ["FRED_API_KEY"] = "YOUR_FRED_API_KEY"  # Replace with your actual key

start_date = datetime.datetime(2017, 1, 1)
end_date = datetime.datetime(2017, 12, 31)

# -- Download GOOG stock price --
goog_data = yf.download("GOOG", start=start_date, end=end_date, interval="1wk", auto_adjust=True)
goog = goog_data["Close"]
goog = pd.Series(goog.values.flatten(), index=goog.index)

# -- Download 10-Year Treasury Yield --
ir_data = yf.download("^TNX", start=start_date, end=end_date, interval="1wk", auto_adjust=True)
ir = ir_data["Close"]
ir = pd.Series(ir.values.flatten(), index=ir.index)

# -- Get CPI and CCI from FRED --
cpi = pdr.DataReader("CPIAUCSL", "fred", start_date, end_date)
cci = pdr.DataReader("UMCSENT", "fred", start_date, end_date)
cpi = cpi.resample("W").ffill().iloc[:, 0]
cci = cci.resample("W").ffill().iloc[:, 0]

# -- Align indexes to common timeframe --
df = pd.concat([goog, ir, cpi, cci], axis=1, join="inner")
df.columns = ["Price", "InterestRate", "CPI", "CCI"]

# -- Compute annualized average return (μ) --
returns = df["Price"].pct_change()
avg_return = returns.mean() * 52
df["AvgReturn"] = avg_return

# -- Save to CSV --
df.to_csv("goog_2017_data.csv")
print("✅ Saved goog_2017_data.csv successfully")
