# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 19:52:11 2021

@author: a
"""

#%% import modules
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from generalCPPI import gCPPI
from commodityCPPI import CommodityCPPI
from constant_proportion import ConstantProportion
#%% data
START = pd.to_datetime('19920101')
END = pd.to_datetime('20191231')

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

data_in_range = data[data.index >= START]
data_in_range = data_in_range[data_in_range.index <= END]

risky_asset_return = data_in_range['Stock'].values /100
rf_asset_return = data_in_range['Cash'].values / 100
time_index = data_in_range.index
vix = VIX.values

BCOM = pd.read_csv('data/BCOM.csv')
BCOM['Time'] = pd.to_datetime(BCOM['Time'], format = '%b %y')
BCOM = BCOM[::-1][['Time','Close']].set_index('Time').resample('1M').last()['Close'] # Start in Feb 91
BCOM_returns = BCOM.pct_change()[1:] * 100
BCOM_returns = BCOM_returns[BCOM_returns.index >= START]
BCOM_returns = BCOM_returns[BCOM_returns.index <= END]

GSCI = pd.read_csv('data/GSCI.csv')
GSCI['Time'] = pd.to_datetime(GSCI['Time'], format = '%b %y')
GSCI = GSCI[::-1][['Time','Close']].set_index('Time').resample('1M').last()['Close'] # Start in Feb 91
GSCI = pd.to_numeric(GSCI.str.replace(',',''), errors='coerce')
GSCI_returns = GSCI.pct_change()[1:] * 100
GSCI_returns = GSCI_returns[GSCI_returns.index >= START]
GSCI_returns = GSCI_returns[GSCI_returns.index <= END]

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
plt.plot(all_equity.nav)
plt.show()
#%% gold CPPI
data_in_range = data[data.index >= START]
risky_asset_return = data_in_range['Stock'].values / 100

index = data_in_range.index

gcppi = CommodityCPPI(risky_asset_return, gold_returns.values/100, 0.7,'rolling peak', multiple = 5)
gcppi.run()
#plt.plot(gcppi.floor)
#plt.plot(commodity_cppi.portfolio_value)
plt.plot(index,all_equity.nav, label = 'all_equity')
plt.plot(index,gcppi.portfolio_value, label = 'gold_cppi')
plt.plot(index,gcppi.commodity_exposure, label = 'gold_holding')
plt.plot(index[:-1],gcppi.floor[:-1], label = 'floor', color = 'black')
#plt.plot(all_equity.nav)
plt.legend()


#%% commodity cppi
start = pd.to_datetime('19920101')
data_in_range = data[data.index >= start]
commodity_in_range = commodity_returns[commodity_returns.index>=start]
risky_asset_return = data_in_range['Stock'].values / 100

ccppi = CommodityCPPI(risky_asset_return, commodity_in_range.values/100, 0.7, 'active reset', multiple = 5)
ccppi.run()
#plt.plot(ccppi.floor)
plt.plot(ccppi.portfolio_value, label = 'c')
plt.plot(all_equity.nav)
plt.legend()

#%% GSCI CPPI
data_in_range = data[data.index >= start]
GSCI_in_range = GSCI_returns[GSCI_returns.index>=start]
risky_asset_return = data_in_range['Stock'].values / 100

gsci_cppi = CommodityCPPI(risky_asset_return, GSCI_in_range.values/100, 0.7, 'rolling peak', multiple = 5)
gsci_cppi.run()
#plt.plot(ccppi.floor)
plt.plot(gsci_cppi.portfolio_value, label = 'gsci')
plt.plot(all_equity.nav)
plt.legend()




#%% examine candidancy
stock = data_in_range['Stock']
bond = data_in_range['Bond']
cash = data_in_range['Cash']

#%%% benchmark
assets = [stock, bond, cash, gold_returns, BCOM_returns,GSCI_returns]
text = ['stock', 'bond', 'cash', 'gold', 'BCOM','GSCI']

for i in range(6):  
    asset = assets[i]
    x = asset.std()
    y = asset.mean()
    plt.scatter(x,y);plt.annotate(text[i], xy = (x,y), xytext = (x+0.05,y+0.01))

plt.axhline(y=0,ls="-",c="black")
plt.title('benchmark')
plt.xlabel('std');plt.ylabel('mean');plt.show()


#%%% after 4% drop
next_return = (stock < -4).shift(1)
next_index = (stock < -4).shift(1)
next_index[0] = False


for i in range(6):
    print(text[i])    
    asset = assets[i]
    x = asset[next_index].std()
    y = asset[next_index].mean()
    plt.scatter(x,y);plt.annotate(text[i], xy = (x,y), xytext = (x+0.05,y+0.01))

plt.title('next month of 4% drop in equity market')
plt.axhline(y=0,ls="-",c="black")
plt.xlabel('std');plt.ylabel('mean');plt.show()


#%%% after 2% drop
next_return = (stock < -2).shift(1)
next_index = (stock < -2).shift(1)
next_index[0] = False


for i in range(6):
    print(text[i])    
    asset = assets[i]
    x = asset[next_index].std()
    y = asset[next_index].mean()
    plt.scatter(x,y);plt.annotate(text[i], xy = (x,y), xytext = (x+0.05,y+0.01))

plt.title('next month of 2% drop in equity market')
plt.axhline(y=0,ls="-",c="black")
plt.xlabel('std');plt.ylabel('mean');plt.show()

#%%% after 6% drop
next_return = (stock < -6).shift(1)
next_index = (stock < -6).shift(1)
next_index[0] = False


for i in range(6):
    print(text[i])    
    asset = assets[i]
    x = asset[next_index].std()
    y = asset[next_index].mean()
    plt.scatter(x,y);plt.annotate(text[i], xy = (x,y), xytext = (x+0.05,y+0.01))

plt.title('next month of 6% drop in equity market')
plt.axhline(y=0,ls="-",c="black")
plt.xlabel('std');plt.ylabel('mean');plt.show()
