import matplotlib.pyplot as plt
import numpy as np
class PortfolioStrategy(object):
    """Base class for PortFolio Strategies
    
    Attributes:
    nav:  np.array of net asset values
    num_periods: number of periods
    """
    def __init__(self):
        return
    """
    def plot(self, choice = 'nav'):
        indices = range(0,self.num_periods)
        def plot_nav(): 
            plt.plot(indices,self.nav)
            plt.xlabel('time')
            plt.show()
            
        def plot_bond_and_equity():
            p_bond = plt.bar(indices, self.bond_holdings, color = 'blue')
            p_equity = plt.bar(indices, self.equity_holdings, bottom = self.bond_holdings, color = 'red')
            
            plt.legend((p_bond,p_equity),('Bond Value','Equity Value'),loc = 3)
            plt.xlabel('time')
            plt.show()
            
        def plot_floor_and_cushion():
            p_floor = plt.bar(indices, self.floor_ts, color = 'blue')
            p_cushion = plt.bar(indices, self.nav - self.floor_ts, bottom = self.floor_ts, color = 'red')
            plt.legend((p_floor,p_cushion),('Floor Value','Cushion Value'),loc = 3)
            plt.xlabel('time')
            plt.show()
        switch = {'nav':plot_nav, 'bond_and_equity' : plot_bond_and_equity, 'floor_and_cushion':plot_floor_and_cushion}
        switch.get(choice,plot_nav)()
    """