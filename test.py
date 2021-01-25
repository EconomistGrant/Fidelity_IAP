# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 21:06:39 2021

@author: a
"""
#%% import modules
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from generalCPPI import gCPPI
from OBPI import OBPI
from constant_proportion import ConstantProportion

def SR(nav:np.array, rf_rate:np.array):
  assert len(nav) == len(rf_rate)
  returns = np.zeros(len(nav)-1)
  for i in range(len(nav) - 1):
    returns[i] = nav[i+1] / nav[i] - 1 - rf_rate[i+1]
  annualized_mean = returns.mean() * 12
  annualized_std = returns.std() * np.sqrt(12)
  return annualized_mean / annualized_std

#%% data loading
data = pd.read_excel("data/hist_data.xlsx")
data.columns = ["time","Stock","Bond","Cash"]
data["time"] = pd.to_datetime(data["time"])
data = data.set_index('time')
data = data.resample('1M').last()
# each data in the original table represents return of the month starting with that date
# We adjust to represent the month ending with that date

VIX = pd.read_csv("data/VIX.csv")
VIX['Date'] = pd.to_datetime(VIX['Date'])
VIX = VIX.set_index('Date').resample('1M').last()['vix'] / 100 /np.sqrt(12)

data_in_range = data[data.index >= VIX.index[0]]

risky_asset_return = data_in_range['Stock'].values /100
rf_asset_return = data_in_range['Cash'].values / 100
time_index = data_in_range.index
vix = VIX.values
#%% All Equity Benchmark
all_equity = ConstantProportion(np.array([1,0]),data_in_range[['Stock','Bond']].values/100)
plt.plot(all_equity.nav)

print(SR(all_equity.nav, rf_asset_return))
plt.show()

#%% vanilla CPPI
cppi = gCPPI(risky_asset_return,rf_asset_return)
multiple_strategy = "constant"            #@param {type:"string"}
floor_strategy = "vanilla"                #@param {type:"string"}
multiple = 5                              #@param {type:"number"}
floor = 0.7                               #@param {type:"number"}

cppi.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor)

print(SR(cppi.nav, rf_asset_return))
cppi.plot('nav')
cppi.plot('floor_and_cushion',time_index)
cppi.plot('bond_and_equity',time_index)

#%% dynamic CPPI
dcppi = gCPPI(risky_asset_return,rf_asset_return)
multiple_strategy = "constant"            #@param {type:"string"}
floor_strategy = "dynamic"                #@param {type:"string"}
multiple = 5                              #@param {type:"number"}
floor = 0.7                               #@param {type:"number"}

dcppi.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor)

dcppi.plot('nav')
dcppi.plot('floor_and_cushion',time_index)
dcppi.plot('bond_and_equity',time_index)

#%% dynamic double floor
d2cppi = gCPPI(risky_asset_return,rf_asset_return)
multiple_strategy = "constant"            #@param {type:"string"}
floor_strategy = "dynamic double"         #@param {type:"string"}
multiple = 5                              #@param {type:"number"}
floor = 0.7                               #@param {type:"number"}
cushion = 0.1                             #@param {type:"number"}


d2cppi.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy,multiple = multiple, floor = floor, cushion = 0.1)
#plt.plot(d2cppi.nav)

print(SR(d2cppi.nav, rf_asset_return))

d2cppi.plot('nav')
d2cppi.plot('floor_and_margin_and_cushion',time_index)
d2cppi.plot('bond_and_equity',time_index)

#%% dynamic double floor + VIX-based volatility
d2cppi_implied_vol = gCPPI(risky_asset_return,rf_asset_return)
multiple_strategy = "time-variant"        #@param {type:"string"}
floor_strategy = "dynamic double"         #@param {type:"string"}
floor = 0.7                               #@param {type:"number"}
cushion = 0.1                             #@param {type:"number"}
vol = vix                                 #@param {type:"raw"}
level = 0.01                              #@param {type:"number"}

d2cppi_implied_vol.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, floor = floor, cushion = cushion, vol = vol, level = level)

#plt.plot(d2cppi.nav)

print(SR(d2cppi.nav, rf_asset_return))

d2cppi_implied_vol.plot('nav')
d2cppi_implied_vol.plot('floor_and_margin_and_cushion',time_index)
d2cppi_implied_vol.plot('bond_and_equity',time_index)


above_floor_rate = (d2cppi_implied_vol.nav > d2cppi_implied_vol.floor).sum() / len(d2cppi_implied_vol.nav)