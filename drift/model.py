import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gbm.visual import visual

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "nflx_2017_data.csv"))
df = pd.read_csv(data_path)

# Aggregate macroeconomic features 
macro_features = df[["CPI", "CCI", "InterestRate"]].mean().values.reshape(1, -1)
true_mu = df["AvgReturn"].iloc[0]

#  Scale and train neural network 
scaler = StandardScaler()
X = scaler.fit_transform(macro_features)
y = np.array([true_mu])

model = MLPRegressor(hidden_layer_sizes=(8, 4), max_iter=5000, random_state=42)
model.fit(X, y)

# Predict drift µ 
predicted_mu = model.predict(X)[0] + 0.03
print(f"True µ: {true_mu:.4f}")
print(f"Predicted µ: {predicted_mu:.4f}")

returns = df["Price"].pct_change().dropna()
historical_vol = returns.std() * np.sqrt(52)
print(f"Historical σ (volatility): {historical_vol:.4f}")

prices = df["Price"].values.tolist()
prices.reverse()

So = np.array([prices[0]])
BrownianIncrements = 500
TotalTime = 1.0
sims = 25

(v1, v1name) = (0, "t")
(v2, v2name) = (0, "stock price")

mod = visual("montecarlo", prices, [], So, BrownianIncrements, v1, v1name, v2name, v2, TotalTime, sims, TotalTime/BrownianIncrements)
mod.sys.mu = predicted_mu
mod.sys.sigma = historical_vol 
mod.graph()
