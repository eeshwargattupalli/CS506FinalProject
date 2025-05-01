import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sde import sde
import math

# --- CLI Arguments ---
# Usage: python predict.py STOCK START_YEAR END_YEAR TEST_WEEKS
stock_name = sys.argv[1] if len(sys.argv) > 1 else "NFLX"
start_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2016
end_year = int(sys.argv[3]) if len(sys.argv) > 3 else 2017
test_weeks = int(sys.argv[4]) if len(sys.argv) > 4 else 52  # number of weeks to test on

stock_lower = stock_name.lower()
train_file = f"./data/{stock_lower}_{start_year}-{end_year}_data.csv"
test_file = f"./data/{stock_lower}_{end_year + 1}-{end_year + math.ceil(test_weeks/52)}_data.csv"

# --- Load Data ---
df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file).dropna(subset=["CPI", "CCI", "InterestRate", "Price"])

df_test = df_test.iloc[:test_weeks]

# --- Train Neural Network ---
X_train, y_train = [], []
for i in range(len(df_train) - 1):
    features = df_train.loc[i, ["CPI", "CCI", "InterestRate"]].values
    log_ret = np.log(df_train.loc[i + 1, "Price"] / df_train.loc[i, "Price"])
    X_train.append(features)
    y_train.append(log_ret)

X_train = np.array(X_train)
y_train = np.array(y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = MLPRegressor(hidden_layer_sizes=(16, 8), alpha=0.001,
                     learning_rate_init=0.001, max_iter=5000, random_state=42)
model.fit(X_train_scaled, y_train)


X_test = df_test[["CPI", "CCI", "InterestRate"]].values
X_test_scaled = scaler.transform(X_test)

weekly_log_returns = model.predict(X_test_scaled)
weekly_log_returns = np.clip(weekly_log_returns, -0.01, 0.01)
mu_estimate = np.mean(weekly_log_returns) * 52

starting_price = df_test.iloc[0]["Price"]
predicted_prices_full = [starting_price]
for r in weekly_log_returns[1:]:
    predicted_prices_full.append(predicted_prices_full[-1] * np.exp(r))

df_test["PredictedPriceFull"] = predicted_prices_full
df_test["Date"] = pd.to_datetime(df_test["Unnamed: 0"] if "Unnamed: 0" in df_test.columns else df_test.index)

stockPrices = df_train["Price"].values.tolist()
stockOptions = [(1, 0.25)]
initial_price = df_test.iloc[0]["Price"]

sde_sim = sde(
    T="montecarlo",
    stockPrices=stockPrices,
    stockOptions=stockOptions,
    initialPoint=initial_price,
    brownianIncrements=len(df_test),
    v1=0,
    v2=0,
    totalTime=1,
    sims=0
)
sde_sim.mu = mu_estimate
s_path = sde_sim.bmflowkickIter()

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(df_test["Date"].iloc[:len(s_path)], df_test["Price"].iloc[:len(s_path)], label="True Price", linewidth=2)
plt.plot(df_test["Date"].iloc[:len(s_path)], s_path, label="SDE Simulated Price", linestyle=":", linewidth=2)
plt.title(f"📈 {stock_name} Prediction ({test_weeks} weeks)", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
