import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from strategy import PortfolioStrategy

class ConstantProportion(PortfolioStrategy):
    """A constant proportion strategy
    
    Attributes:
    -----------
    weights: np.array of weights, shape = (n,)
    returns: np.array of returns, shape = (T,n)"""
    def __init__(self,weights,returns):
        self.weights = weights
        self.returns = returns
        
        self.num_periods,n = returns.shape
        self.portfolio_returns = np.dot(returns,weights.reshape((n,1))).ravel()
        self.nav = (1+self.portfolio_returns).cumprod()

if __name__ == '__main__':
    returns = np.array([[1,2,3],[2,3,4],[4,5,6]])/100
    weights = np.array([0.4,0.4,0.2])

    mix = ConstantProportion(weights,returns)
    mix.plot()