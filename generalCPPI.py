import os
import sys
_path = os.path.dirname(os.path.abspath(__file__))
if _path not in sys.path:
    sys.path.append(_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

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
        assert multiple_strategy in ["vanilla", "constant", "time-variant", "input_vol"]
        assert floor_strategy in ["vanilla","dynamic","double","dynamic double","d2","dd"]

        if multiple_strategy in ["vanilla","constant"]:
            multiple = kwargs.get("multiple",5)
            def _get_multiple(t):
                return multiple
        elif multiple_strategy in ["time-variant","input_vol"]:
            assert "vol" in kwargs, 'input volatility array as vol'
            self.vol = kwargs.get("vol",None)
            assert len(self.vol) == len(self.risky_asset_returns), "vol and asset return should have same dimension"
            level = kwargs.get("level",0.01)
            def _get_multiple(t):
                σ = self.vol[t-1]
                critical_value = σ*stats.norm.ppf(level)
                return 1 / abs(critical_value)        
    
        # return self.floor[t-1] which is the end of (t-1) floor value
        if floor_strategy == "vanilla":
            floor_ratio = kwargs.get("floor",0.7)
            def _get_floor(t):
                #print("cppi")
                if t == 1:
                    self.floor[0] = floor_ratio
                    return self.floor[0]
                self.floor[t-1] = self.floor[t-2] * (1+self.rf_asset_returns[t-1])
                #print(self.floor[t-1] * (1+self.rf_asset_returns[t]))
                return self.floor[t-1]
        elif floor_strategy == "dynamic":
            floor_ratio = kwargs.get("floor",0.7)
            def _get_floor(t):
                #print("dcppi")
                if t == 1:
                    self.floor[0] = floor_ratio
                    return self.floor[0]
                self.floor[t-1] = max(self.floor[t-2],self.nav[t-1]*floor_ratio)
                return self.floor[t-1]
        elif floor_strategy == "double":
            self.floor_ratio = kwargs.get("floor", 0.7)
            self.margin_ratio = kwargs.get("margin",0.1)
            def _get_floor(t):
                if t == 1:
                    self.floor[0] = self.floor_ratio
                    self.margin = np.zeros(self.num_periods)
                    self.margin[0] = self.margin_ratio
                    self.cushion_threshold = (1 - self.floor_ratio - self.margin_ratio) / 2
                    return self.floor[t-1] + self.margin[t-1]
                self.floor[t-1] = self.floor[t-2] * self.rf_asset_returns[t-1]
                self.margin[t-1] = self.floor[t-2] * self.rf_asset_returns[t-1]
                cushion = self.nav[t-1] - self.floor[t-1] - self.margin[t-1]
                if cushion < self.cushion_threshold:
                    self.margin[t-1] = self.margin[t-1] / 2
                    self.cushion_threshold = self.cushion_threshold / 2
                    self.margin_ratio = self.margin_ratio / 2
                    #print(self.margin_ratio)
                return self.floor[t-1] + self.margin[t-1]
        elif floor_strategy in ["dynamic double", "d2", "dd"]:
            self.floor_ratio = kwargs.get("floor", 0.7)
            self.margin_ratio = kwargs.get("margin",0.1)
            def _get_floor(t):
                if t == 1:
                    self.floor[0] = self.floor_ratio
                    self.margin = np.zeros(self.num_periods)
                    self.margin[0] = self.margin_ratio
                    self.cushion_threshold = (1 - self.floor_ratio - self.margin_ratio) / 2
                    return self.floor[t-1] + self.margin[t-1]
                self.floor[t-1] = max(self.floor[t-2],self.nav[t-1]*self.floor_ratio)
                self.margin[t-1] = max(self.margin[t-2],self.nav[t-1]*self.margin_ratio)
                cushion = self.nav[t-1] - self.floor[t-1] - self.margin[t-1]
                if cushion < self.cushion_threshold:
                    self.margin[t-1] = self.margin[t-1] / 2
                    self.cushion_threshold = self.cushion_threshold / 2
                    self.margin_ratio = self.margin_ratio / 2
                    #print(self.margin_ratio)
                return self.floor[t-1] + self.margin[t-1]

        for t in range(1,self.num_periods):
            effective_floor = _get_floor(t)
            multiple = _get_multiple(t)
            
            prev_nav = self.nav[t-1]

            cushion = prev_nav - effective_floor
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

    def plot(self, indices:pd.Series, choice = 'nav',):
        def plot_nav(): 
            plt.plot(indices,self.nav)
            plt.xlabel('time')
            plt.show()
            
        def plot_bond_and_equity():
            p_bond = plt.bar(indices, self.rf_holding, color = 'blue')
            p_equity = plt.bar(indices, self.exposure, bottom = self.rf_holding, color = 'red')
            
            plt.legend((p_bond,p_equity),('RF Holding','Risky Holding'),loc = 3)
            plt.xlabel('time')
            plt.show()
            
        def plot_floor_and_cushion():
            p_floor = plt.bar(indices, self.floor, color = 'blue')
            p_cushion = plt.bar(indices, self.nav - self.floor, bottom = self.floor, color = 'red')
            plt.legend((p_floor,p_cushion),('Floor Value','Cushion Value'),loc = 3)
            plt.xlabel('time')
            plt.show()

        def plot_floor_and_margin_and_cushion():
            p_floor = plt.bar(indices, self.floor, color = 'blue')
            p_cushion = plt.bar(indices, self.cushion, color = 'blue')
            p_cushion = plt.bar(indices, self.nav - self.cushion - self.floor, bottom = self.floor, color = 'red')
            plt.legend((p_floor,p_cushion),('Floor Value','Cushion Value'),loc = 3)
            plt.xlabel('time')
            plt.show()


        switch = {'nav':plot_nav, 'bond_and_equity' : plot_bond_and_equity, 'floor_and_cushion':plot_floor_and_cushion, 'floor_and_margin_and_cushion',plot_floor_and_margin_and_cushion}
        switch.get(choice,plot_nav)()


if __name__ == '__main__':
    simulated_bond_returns = np.array([-2,-3,-4,-3,-2,-1,2,3,4,3])/100
    simulated_equity_returns = np.array([-9,-2,-8,-1,-7,-3,-6,-2,5,0])/100
    
    multiple = 4
    floor = 0.8
    
    """
    cppi = gCPPI(simulated_equity_returns,simulated_bond_returns)
    cppi.run(multiple_strategy = "constant", floor_strategy = "vanilla",multiple = multiple, floor = floor)
    print(cppi.nav[-1])
    

    dcppi = gCPPI(simulated_equity_returns,simulated_bond_returns)
    dcppi.run(multiple_strategy = "constant", floor_strategy = "dynamic",multiple = multiple, floor = floor)
    print(dcppi.nav)
    
    d2cppi = gCPPI(simulated_equity_returns,simulated_bond_returns)
    d2cppi.run(multiple_strategy = "constant", floor_strategy = "d2",multiple = 5, floor = floor, margin = 0.1)
    """
    vol = np.array([9,10,11,10,10,5,5,5,20,20])/100
    volcppi = gCPPI(simulated_equity_returns,simulated_bond_returns)
    volcppi.run(multiple_strategy = "input_vol", floor_strategy = "vanilla",multiple = multiple, floor = floor,vol = vol)
    