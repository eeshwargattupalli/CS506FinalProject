import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
import datetime
import os
import numpy as np

# Set your own FRED api key from https://fred.stlouisfed.org/

os.environ["FRED_API_KEY"] = "a91935e847bfc02614351c2b582563d7"

def generate_stock_macro_data(stock_name: str, start_year: int, end_year: int, save_dir: str = "data"):
   #Creates .csv to test on
    os.makedirs(save_dir, exist_ok=True)

    start_date = datetime.datetime(start_year, 1, 1)
    end_date = datetime.datetime(end_year, 12, 31)
    weekly_index = pd.date_range(start=start_date, end=end_date, freq="W-FRI")

    stock_daily = yf.download(stock_name, start=start_date, end=end_date, interval="1d", auto_adjust=True)
    
    stock_weekly = stock_daily["Close"].resample("W-FRI").last().reindex(weekly_index).ffill()

    ir_daily = yf.download("^TNX", start=start_date, end=end_date, interval="1d", auto_adjust=True)
    ir_weekly = ir_daily["Close"].resample("W-FRI").last().reindex(weekly_index).ffill()

    cpi = pdr.DataReader("CPIAUCSL", "fred", start_date, end_date)
    cci = pdr.DataReader("UMCSENT", "fred", start_date, end_date)
    cpi_weekly = cpi.resample("W-FRI").ffill().reindex(weekly_index).ffill()
    cci_weekly = cci.resample("W-FRI").ffill().reindex(weekly_index).ffill()

    df = pd.concat([stock_weekly, ir_weekly, cpi_weekly, cci_weekly], axis=1)
    df.columns = ["Price", "InterestRate", "CPI", "CCI"]
    df.dropna(inplace=True)

    df["LogReturn"] = np.log(df["Price"] / df["Price"].shift(1))
    df.dropna(inplace=True)

    stock_lower = stock_name.lower()
    filename = f"{stock_lower}_{start_year}-{end_year}_data.csv"
    path = os.path.join(save_dir, filename)
    df.to_csv(path)
    print(f"✅ Saved {len(df)} rows to {path}")


def generate_future_data(stock_name: str, end_year: int, years_ahead: int, save_dir: str = "data"):
        #Generaets the future weeks to test data on
        generate_stock_macro_data(stock_name, end_year + 1, end_year + years_ahead, save_dir)


