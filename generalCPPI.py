import os
import sys
_path = os.path.dirname(os.path.abspath(__file__))
if _path not in sys.path:
    sys.path.append(_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from strategy import PortfolioStrategy
class gCPPI(PortfolioStrategy):
    """General CPPI class that is going to replace old version, give the flexibility to use different 
    floor and multiple strategies
    
    Floor strategies: 
    1. Vanilla: grows with risk free rate
    2. Dynamic Floor CPPI(TIPP): a proportion of the max historical portfolio value
    3. Double Floor CPPI(Margin CPPI): Floor + Margin + Cushion
    4. Dynamic Double CPPI

    Multiple strategies:
    1. Constant: rule-of-thumb constant or time-invariant VaR / Expected Shortfall / Conditional
    2. Conditional (Time-Variant): GARCH VaR -> normal dist -> quantile / Conditional AutoRegressive VAR / 
    Implied from option pricing"""
    def __init__(self,
                 risky_asset_returns:np.array,
                 rf_asset_returns: np.array,
                 max_leverage: float = 1.0):
        assert len(risky_asset_returns) == len(rf_asset_returns), 'Inputs should have the same dimension'
        self.risky_asset_returns = risky_asset_returns
        self.rf_asset_returns = rf_asset_returns
        self.max_leverage = max_leverage

        self.num_periods = len(risky_asset_returns)
        self.floor = np.zeros(self.num_periods)
        self.exposure = np.zeros(self.num_periods)
        self.rf_holding = np.zeros(self.num_periods) 
        self.nav = np.ones(self.num_periods)


    def run(self, multiple_strategy, floor_strategy,**kwargs):
        assert multiple_strategy in ["vanilla","constant", "EWMA", "GARCH"]
        assert floor_strategy in ["vanilla","dynamic","double","dynamic double","d2","dd"]

        if multiple_strategy == "vanilla" or "constant":
            multiple = kwargs.get("multiple",5)
            def _get_multiple(t):
                return multiple

        # return self.floor[t-1] which is the end of (t-1) floor value
        if floor_strategy == "vanilla":
            floor_ratio = kwargs.get("floor",0.8)
            def _get_floor(t):
                #print("cppi")
                if t == 1:
                    self.floor[0] = floor_ratio
                self.floor[t] = self.floor[t-1] * (1+self.rf_asset_returns[t])
                #print(self.floor[t-1] * (1+self.rf_asset_returns[t]))
                return self.floor[t-1]
        elif floor_strategy == "dynamic":
            floor_ratio = kwargs.get("floor",0.8)
            def _get_floor(t):
                #print("dcppi")
                if t == 1:
                    self.floor[0] = floor_ratio
                    return self.floor[0]
                self.floor[t-1] = max(self.floor[t-2],self.nav[t-1]*floor_ratio)
                return self.floor[t-1]



        for t in range(1,self.num_periods):
            floor = _get_floor(t)
            multiple = _get_multiple(t)
            
            prev_nav = self.nav[t-1]

            cushion = prev_nav - floor
            exposure = max(0, min(cushion * multiple, self.max_leverage * prev_nav))
            rf_holding = prev_nav - exposure
            
            """
            print("t:" +str(t))
            print(multiple)
            print(floor)
            print(cushion)
            print(exposure)
            print("________")
            """
            self.exposure[t] = exposure
            self.rf_holding[t] = rf_holding
            self.nav[t] = self.nav[t-1] + rf_holding * self.rf_asset_returns[t] + exposure * self.risky_asset_returns[t]



if __name__ == '__main__':
    simulated_bond_returns = np.array([2,3,4,3,2,1,2,3,4,3])/100
    simulated_equity_returns = np.array([9,-2,8,-1,7,-3,6,-2,5,0])/100
    multiple = 0.5
    floor = 0.8

    cppi = gCPPI(simulated_equity_returns,simulated_bond_returns)
    cppi.run(multiple_strategy = "constant", floor_strategy = "vanilla",multiple = multiple, floor = floor)
    print(cppi.nav[-1])
    

    dcppi = gCPPI(simulated_equity_returns,simulated_bond_returns)
    dcppi.run(multiple_strategy = "constant", floor_strategy = "dynamic",multiple = multiple, floor = floor)
    print(dcppi.nav)