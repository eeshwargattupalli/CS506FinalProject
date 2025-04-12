import numpy as np
import matplotlib.pyplot as plt
import random as r
import matplotlib.animation as animation
from gbm.sde import sde
import math

class visual:
    def __init__(self, T, stockPrices, stockOptions, initialPoint, brownianIncrements, v1, v1name, v2name, v2, totalTime=0,sims=0, dt=0):
        self.type = T # tells us whether we want an indefinite animation or a montecarlo simulation 
        self.fig, self.ax = plt.subplots(figsize=(10,6))
        self.x_data = []
        self.y_data = []
        self.line, = self.ax.plot([],[],'r', linewidth=1)
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(0, initialPoint + 5)
        self.HorizName = v1name
        self.VertName = v2name
        self.sys = sde(T, stockPrices, stockOptions, initialPoint, brownianIncrements, v1, v2, totalTime, sims) #object representing the system
        self.s = [0]
        self.s[0] = initialPoint
        self.dt = self.sys.dt   
    
    def init_animation(self): 
        self.line.set_data([], [])
        self.ax.set_xlabel(self.HorizName)
        self.ax.set_ylabel(self.VertName)
        return self.line,

    def updateInd(self, frame):
        if frame!=0:
            self.s = self.sys.bmflowkickSingle(self.s, self.dt, frame)
        self.x_data.append(frame/self.sys.N)
        self.y_data.append(self.s[frame])
        
        if len(self.x_data) > 1 and len(self.y_data) > 1:
            self.ax.set_xlim(min(self.x_data) - 1, max(self.x_data) + 1)
            self.ax.set_ylim(min(self.y_data) - 1, max(self.y_data) + 1)
            self.ax.figure.canvas.draw()

        self.line.set_data(self.x_data, self.y_data)

        return self.line,

    def mcSim(self):
        (max, min) = (0, math.inf)
        for i in range(self.sys.sims):
            sMatrix = self.sys.bmflowkickIter()
            H = np.linspace(0, len(sMatrix)*self.dt, len(sMatrix))
            V = sMatrix
            if np.max(V) > max:
                max = np.max(V)
            if np.min(V) < min:
                min = np.min(V)
            plt.xlim(0, np.max(H))
            plt.ylim(min, max)
            plt.plot(H, V)
            plt.plot()
        plt.xlabel(self.HorizName)
        plt.ylabel(self.VertName)
        plt.title('Brownian Motion Montecarlo Simulation')
        plt.show()

    def animate(self):
        ani = animation.FuncAnimation(self.fig, self.updateInd, init_func=self.init_animation, blit=True, interval=50, save_count=5000)
        paused = False

        def toggle_pause(event): # Function allows for pausing
            nonlocal paused
            if event.key == 'p':
                if paused:
                    ani.event_source.start()
                    paused = False
                else: 
                    ani.event_source.stop()
                    paused = True
        
        self.fig.canvas.mpl_connect('key_press_event', toggle_pause)
        plt.pause(0.01)
        plt.show()

    def graph(self):
        if self.type == "montecarlo":
            self.mcSim()
        elif self.type == "animation":
            self.animate()
        else:
            print("Type not supported")

