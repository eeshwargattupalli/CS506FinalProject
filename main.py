from gbm.sde import sde
from gbm.visual import visual
import matplotlib.pyplot as plt
import numpy as np

######## DATA ########
goog_options = [(5750, 0.0156), (4640, 0.0625), (3374, .0625), (1258, 0.1250), (3343, 0.1250), (1162, 0.1250), (80, .1250), (19, .25), (31, .25), (5, .25), (1, .25), (1, .25), (2, .5), (5, .5), (3, .5), (2, .5), (4, .5)]

goog_s_prices = [188.79, 186.78, 191.96, 183.42, 180.26, 178.17, 175.75, 173.76, 176.13, 177.09, 170.10, 168.80, 173.49, 155.54, 159.01, 153.77, 152.09, 151.60, 142.01, 136.14, 137.92, 145.12, 141.60, 150.05, 143.38, 153.62, 147.80, 144.08, 137.23, 140.77, 142.56, 133.69, 136.48, 133.17, 138.06, 136.78, 133.91, 130.22, 123.26, 136.58, 138.42, 138.57, 131.70, 131.10, 138.14, 137.04, 136.64, 130.54, 127.96, 130.02, 128.39, 132.86, 120.17]
goog_s_prices.reverse()

So = np.array([goog_s_prices[0]]) # Initial point
TotalTime = 1.0 # Total time interval
BrownianIncrements = 500 # Total number of brownian increments
sims = 25 # Number of individual simulations (only pertains to montecarlo)

(v1, v1name) = (0, "t") # Horizontal axis
(v2, v2name) = (0, "stock price") # Vertical axis


#Possibilities: animation, montecarlo, deterministic
SimType = "montecarlo"
mod = visual(SimType, goog_s_prices, goog_options, So, BrownianIncrements, v1, v1name, v2name, v2, TotalTime, sims, TotalTime/BrownianIncrements)
mod.graph()