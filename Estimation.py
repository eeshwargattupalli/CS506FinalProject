from drift.data import generate_stock_macro_data, generate_future_data
import pandas as pd
import numpy as np
import subprocess
import os
import math

def main():
    
    # Choose the stock you want to model
    stock_name = "NFLX"
    
    
    
    # Choose the years you want to model on
    # Start year will be january/1/start_year, and end_year will be december/31/end_year
    start_year = 2023
    end_year = 2023
    
    #weeks you want to predict on then compare

    
    test_weeks = 72
    stock_lower = stock_name.lower()

    generate_stock_macro_data(stock_name, start_year, end_year)

    future_years_needed = math.ceil(test_weeks / 52)
    generate_future_data(stock_name, end_year, future_years_needed)

    subprocess.run(["python", "drift/predict.py", stock_name, str(start_year), str(end_year), str(test_weeks)], check=True)

if __name__ == "__main__":
    main()
