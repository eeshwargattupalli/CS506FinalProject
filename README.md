# CS506FinalProject

Project Proposal:

Description: We can look at historical stock prices and options prices in order to predict the trajectory of future stock price movements. Using this information, we can make informed predictions about the payoffs of different financial securities as they are dependent on multiple macroeconomic variables such as stock market performance but also interest rates and inflation.

Goal: Accurately predict payoffs of different financial securities using stock price models.

Data: We will collect historical stock data and options data using Yahoo! finance API, and we will collect census data using government provided data.

Data Modeling For the stock price projections, we will use a stochastic forward euler’s scheme and Monte-carlo simulations (to approximate average behavior of a random system). We will attempt to use deep learning techniques to forecast the direction of other macroeconomic variables using historical data. If this ends up being too ambitious, we can perform a multiple linear regression or use clustering techniques by inputting the same historical data.

Data Visualization: We can plot Monte-carlo simulations in matplotlib, plot expected versus actual behavior, confidence bands for regression techniques, histograms to compare predictions across different parameter sets, etc.

Test Plan: We can collect any 52-week set of data (for stocks and options) and then compare our predictions to any point in time that is ahead of the time that the last point of data that was collected to test our predictions. The same principle applies for collecting census data but since this data is collected and released less often we will most likely have to collect data over several years as opposed to a single year.

To view simulations, run "python3 main.py" in terminal

Midterm report video: https://youtu.be/g9nxOWQNnqA