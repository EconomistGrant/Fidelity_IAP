import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import numpy as np
import matplotlib.pyplot as plt
from strategy import PortfolioStrategy
class DynamicCPPI(PortfolioStrategy):
    """Implementing dynamic CPPI strategy

    Compared with vanilla CPPI strategy, the floor value will not increase at the risk-free rate, but 
    rather determined based on a fixed proportion of the maximum portfolio value ever reached        
    Attributes:
    -----------
        equity_returns: a numpy array of risky returns
        bond_returns:   a numpy array of risk-free rates
        floor:          floor as a proportion of maximum portfolio value ever reached
        multiple:       proportion of cushion held in equity
        
        floor_ts:       np.array of floor values growing with risk-free rate
        equity_holdings:np.array of equity value hold 
        bond_holdings:  np.array of bond value hold
        nav:            np.array of total portfolio value"""
    def __init__(self,
                 equity_returns:np.array,
                 bond_returns: np.array,
                 floor: float,
                 multiple: float):
        self.floor = floor
        self.multiple = multiple
        self.bond_returns = bond_returns
        self.equity_returns = equity_returns
        
        assert len(equity_returns) == len(bond_returns), 'Inputs should have the same dimension'
        self.num_periods = len(equity_returns)
        self.floor_ts = np.zeros(self.num_periods)
        self.bond_holdings = np.zeros(self.num_periods)
        self.equity_holdings = np.zeros(self.num_periods)
        self.nav = np.ones(self.num_periods)
        
        self.floor_ts[0] = self.floor
        self.max_nav = 1

        for t in range(1,self.num_periods):
            # Set positions for time t at the end of time t-1
            prev_nav = self.nav[t-1]
            prev_floor = self.floor_ts[t-1]
            #TODO: consider different types of floor strategy
            cushion = prev_nav - prev_floor
            if cushion > 0:
                equity_holding = cushion * self.multiple
            else: 
                equity_holding = 0
            bond_holding = prev_nav - equity_holding
            
            self.equity_holdings[t] = equity_holding
            self.bond_holdings[t] = bond_holding

            self.nav[t] = self.nav[t-1] + bond_holding*bond_returns[t] + equity_holding*equity_returns[t]
            
            if self.nav[t] > self.max_nav:
                self.max_nav = self.nav[t]
            self.floor_ts[t] = self.max_nav * floor            

if __name__ == '__main__':
    simulated_bond_returns = np.array([-2,-3,-4,-3,-2,-1,-2,-3,-4,-3])/100
    simulated_equity_returns = np.array([9,-2,-8,-1,-7,-3,-6,-2,-5,0])/100
    multiple = 0.5
    floor = 0.8

    dcppi = DynamicCPPI(simulated_equity_returns,simulated_bond_returns,floor,multiple)
    dcppi.plot()
    dcppi.plot('bond_and_equity')
    dcppi.plot('floor_and_cushion')
    print(dcppi.equity_holdings)