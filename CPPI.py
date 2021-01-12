import numpy as np
import matplotlib.pyplot as plt
from strategy import PortfolioStrategy
class CPPI(PortfolioStrategy):
    """Implementing vanilla CPPI strategy
    
    The investor have a target floor value that grows with risk-free rate
    Start with 1, at any time, he will determine the floor value of next period by "guessing" the risk-free rate 
    of next period, and then determine cushion and amount of equity shares accordingly
        
    Attributes:
    -----------
        equity_returns: a numpy array of risky returns
        bond_returns:   a numpy array of risk-free rates
        floor:          initial floor as a proportion of initial portfolio value
        multiple:       proportion of cushion held in equity

        floor_ts:       np.array of floor values growing with risk-free rate
        equity_ts:      np.array of equity value hold 
        bond_ts:        np.array of bond value hold
        nav_ts:         np.array of total portfolio value
    """
    def __init__(self,
                 equity_returns:np.array,
                 bond_returns: np.array,
                 initial_floor: float,
                 multiple: float):
        self.initial_floor = initial_floor
        self.multiple = multiple
        self.bond_returns = bond_returns
        self.equity_returns = equity_returns

        assert len(equity_returns) == len(bond_returns), 'Inputs should have the same dimension'
        self.num_periods = len(equity_returns)
        self.floor_ts = np.zeros(self.num_periods)
        self.bond_holdings = np.zeros(self.num_periods)
        self.equity_holdings = np.zeros(self.num_periods)
        self.nav = np.ones(self.num_periods)

        self.floor_ts[0] = self.initial_floor
        
        for t in range(1,self.num_periods):
            # Set positions for time t at the end of time t-1
            prev_nav = self.nav[t-1]
            prev_floor = self.floor_ts[t-1]
            #TODO: consider different types of floor strategy
            cushion = prev_nav - prev_floor
            equity_holding = cushion * self.multiple
            bond_holding = prev_nav - equity_holding
            
            self.equity_holdings[t] = equity_holding
            self.bond_holdings[t] = bond_holding
            
            # Realized values at the end of time t but before setting positions for time t + 1
            realized_floor = self.floor_ts[t-1] * (1+self.bond_returns[t])
            self.floor_ts[t] = realized_floor

            self.nav[t] = self.nav[t-1] + bond_holding*bond_returns[t] + equity_holding*equity_returns[t]
            

if __name__ == '__main__':
    simulated_bond_returns = np.array([2,3,4,3,2,1,2,3,4,3])/100
    simulated_equity_returns = np.array([9,-2,8,-1,7,-3,6,-2,5,0])/100
    multiple = 0.5
    initial_floor = 0.8

    cppi = CPPI(simulated_equity_returns,simulated_bond_returns,initial_floor,multiple)
    cppi.plot()
    cppi.plot('bond_and_equity')
    cppi.plot('floor_and_cushion')