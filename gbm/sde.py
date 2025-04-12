import numpy as np
import random as r

class sde:
    # throughout the class, x represents the current state of the system given by an n-dimensional vector

    def __init__(self, T, stockPrices, stockOptions, initialPoint, brownianIncrements, v1, v2, totalTime=0, sims=0, dt=0):
        self.sprices = stockPrices
        self.options = stockOptions
        self.So = initialPoint
        self.impVol = self.ip()
        self.n = len(self.So)
        self.simType = T
        self.T = totalTime
        self.N = brownianIncrements
        if self.simType == "montecarlo" or "deterministic": 
            self.dt = self.T/self.N
        else:
            self.dt = dt
        self.mu = self.drift()
        self.sigma = self.initSigma()
        self.sims = sims
        self.xVar = v1
        self.yVar = v2

    def drift(self):
        Rt = 0    
        p = self.sprices
        for t in range(1, len(p)):
            Rt += (p[t] - p[t-1]) / p[t-1] #daily returns
        
        Rt = (1/(len(p)-1)) * Rt * 252 #averaging the daily returns then annualizing
        
        return (Rt / 100)

    def ip(self):
        wip = 0 #weighted implied volatility
        o = self.options
        wtotal = 0 #total weights
        
        for w, ip in o:
            
            wip += w * ip 
            wtotal += w 

        return wip / wtotal 
    def initSigma(self):
        sig = 0
        match self.simType:
            case "deterministic":
                return sig
            case "montecarlo":
                sig = self.impVol
            case "animation":
                sig = self.impVol
            case _:
                raise Exception("Not a valid type.")
        return sig
    
    def sig(self, x):
        return self.sigma * x
    def l(self, x):
        return 3
    def f(self, x):
        s = x
        return self.mu * s
    
    def bmflowkickSingle(self, s, dt, i): #used for indefinite animation case
        Wt = np.random.normal(0, 1) #brownian motion
        upd = s[i-1] + self.f(s[i-1]) * dt + self.sig(s[i-1]) * Wt * np.sqrt(dt)
        s = np.vstack([s, upd])
        return np.asarray(s)
    
    def bmflowkickIter(self): #retrieve animation or image over set interval
        dt = self.T/self.N
        s = np.zeros(self.N) #s is an N x n matrix
        s[0] = self.So #initial state
        for i in range(1, self.N):
            Wt = np.random.normal(0, 1) #brownian motion
            s[i] = s[i-1] + self.f(s[i-1]) * dt + self.sig(s[i-1]) * Wt * np.sqrt(dt)
        return s

      


