import os
import sys
_path = os.path.dirname(os.path.abspath(__file__))
if _path not in sys.path:
    sys.path.append(_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from constant_proportion import ConstantProportion

class CommodityCPPI(object):
    """A class for commodity CPPI strategy
    Compared with general CPPI (risk-free CPPI)：
    1. replace rf with commodity
    2. restrict equity holdings between 80% - 100%, commodity to be 20% - 0%, and then mix up with bond, cash, foreign equity(outside the scope of strategy itself)
    3. Floor will be hit eventually. So we need to consider how to change floor strategies. 
        1) 70% of Rolling 6-month peak
        2) Reset floor every 6 months
        2) After falling below floor for 3 periods, reset"""
    def __init__(self, risky_asset_returns, commodity_returns, floor_ratio, floor_type, multiple = 5):
        assert floor_type in ['rolling peak', 'active reset', 'passive reset']
        assert len(risky_asset_returns) == len(commodity_returns)
        self.risky_asset_returns = risky_asset_returns
        self.commodity_returns = commodity_returns
        self.floor_ratio = floor_ratio
        self.floor_type = floor_type
        self.multiple = multiple

        self.num_periods = len(risky_asset_returns)
        self.floor = np.zeros(self.num_periods)
        self.equity_exposure = np.zeros(self.num_periods)
        self.commodity_exposure = np.zeros(self.num_periods)
        self.portfolio_value = np.ones(self.num_periods)

    def _get_floor(self,t):
        if self.floor_type == 'rolling peak':
            if t <= 12:
                self.floor[t-1] = self.floor_ratio
            else: 
                self.floor[t-1] = max(self.portfolio_value[t-12:t]) * self.floor_ratio
            return self.floor[t-1]
        elif self.floor_type == 'active reset':
            if t == 1:
                self.floor[t-1] = self.floor_ratio
            elif t % 12 == 1:
                self.floor[t-1] = self.portfolio_value[t-1] * self.floor_ratio
            else: 
                self.floor[t-1] = self.floor[t-2]
            return self.floor[t-1]  
        else: #passive reset
            if t <= 3:
                self.floor[t-1] = self.floor_ratio
            elif self.portfolio_value[t-4] < self.floor[t-4] and self.portfolio_value[t-3] < self.floor[t-3] and self.portfolio_value[t-2] < self.floor[t-2]:
                self.floor[t-1] = self.portfolio_value[t-1]*self.floor_ratio
            else:
                self.floor[t-1] = self.floor[t-2]
            return self.floor[t-1]
    def run(self):
        self.returns = []
        for t in range(1,self.num_periods):
            floor = self._get_floor(t)
            multiple = self.multiple
            cushion = self.portfolio_value[t-1] - floor

            equity_exposure = max(0.8 * self.portfolio_value[t-1], min(cushion * multiple, self.portfolio_value[t-1]))
            commodity_exposure = self.portfolio_value[t-1] - equity_exposure

            self.equity_exposure[t] = equity_exposure
            self.commodity_exposure[t] = commodity_exposure

            self.portfolio_value[t] = self.portfolio_value[t - 1] + self.equity_exposure[t]*self.risky_asset_returns[t] + self.commodity_exposure[t] * self.commodity_returns[t]
            self.returns.append(self.portfolio_value[t]/self.portfolio_value[t - 1] - 1)
            

if __name__ == '__main__':
    #%% data
    START = pd.to_datetime('20000101')
    END = pd.to_datetime('20191231')
    
    data = pd.read_excel("data/hist_data.xlsx")
    data.columns = ["time","Stock","Bond","Cash"]
    data["time"] = pd.to_datetime(data["time"])
    data = data.set_index('time')
    data = data.resample('1M').last()
    data_in_range = data[data.index >= START]
    data_in_range = data_in_range[data_in_range.index <= END]

    
    gold = pd.read_csv('data/GOLD.csv', header = None)
    gold.columns = ['Time','price']
    gold['Time'] = pd.to_datetime(gold['Time'])
    gold = gold.set_index('Time')['price'].resample('1M').last()
    gold = pd.to_numeric(gold.str.replace(',',''), errors='coerce')
    gold_returns = gold.pct_change()[1:] * 100
    gold_returns = gold_returns[gold_returns.index >= START]
    gold_returns = gold_returns[gold_returns.index <=END]
    #%% all equity benchmark
    all_equity = ConstantProportion(np.array([1,0]),data_in_range[['Stock','Bond']].values/100)
    #%% gold CPPI
    data_in_range = data[data.index >= START]
    risky_asset_return = data_in_range['Stock'].values / 100
    
    index = data_in_range.index
    
    gcppi = CommodityCPPI(risky_asset_return, gold_returns.values/100, 0.7,'rolling peak', multiple = 4)
    gcppi.run()
    #plt.plot(gcppi.floor)
    #plt.plot(commodity_cppi.portfolio_value)
    plt.plot(index,all_equity.nav, label = 'all_equity')
    plt.plot(index,gcppi.portfolio_value, label = 'gold_cppi')
    plt.plot(index,gcppi.commodity_exposure, label = 'gold_holding')
    plt.plot(index[:-1],gcppi.floor[:-1], label = 'floor', color = 'black')
    #plt.plot(all_equity.nav)
    plt.legend()