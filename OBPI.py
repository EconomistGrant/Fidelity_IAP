#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 21:59:48 2021

@author: fuyutang
"""
import numpy as np
import pandas as pd
import math
from scipy.stats import norm
import matplotlib.pyplot as plt


class OBPI(object):
	
	def __init__(self, equity_price: np.array, risk_free: np.array, vol: np.array,
			  floor: float, principal: float):
		"""
		Parameters
		----------
		floor : float
			A percentage between 90% and 100%, which means an investor is willing to recover
			a percentage p of her initial investment.

		Returns
		-------
		None.

		"""
		
		self.floor = floor
		self.equity_price = equity_price
		self.risk_free = risk_free
		self.vol = vol
		self.principal = principal
		
		assert len(equity_price) == len(risk_free)
		self.num_periods = len(equity_price)
		self.bond_holdings = np.zeros(self.num_periods)
		self.equity_holdings = np.zeros(self.num_periods)
		self.port_val = np.zeros(self.num_periods)
		
		self.floor_ts = np.zeros(self.num_periods)
		r0 = self.risk_free[0]
		self.floor_ts[0] = self.floor * self.principal * np.exp(-r0*(self.num_periods-1))
		
		self.port_val[0] = self.principal
		
		for t in range(0, self.num_periods-1):
			
			S = self.equity_price[t]
			r = self.risk_free[t]
			tao = self.num_periods - 1 - t
			sigma = self.vol[t]
			lower = 0  #wait for change
			upper = 4000 #wait for change
			
			#calculate strike k which can guarantee the floor protection
			K = self.cal_K(S, r, tao, sigma, lower, upper)
			d1 = (math.log(S/K) + (r + sigma ** 2 / 2) * tao) / (sigma * np.sqrt(tao))
			equity_weight = S * np.exp(-r * tao) * norm.cdf(d1) / (S + self.option_price(S, K, sigma, r, tao)[1])
			
			self.equity_holdings[t] = equity_weight * self.port_val[t]
			self.bond_holdings[t] = self.port_val[t] - self.equity_holdings[t]
			equity_return = (self.equity_price[t+1] - self.equity_price[t]) / self.equity_price[t]
			self.floor_ts[t+1] = self.floor_ts[t] * (1 + self.risk_free[t])
			
			self.port_val[t+1] = self.port_val[t] + self.equity_holdings[t] * equity_return + self.bond_holdings[t] * self.risk_free[t]

	
	def cal_K(self, S, r, tao, sigma, lower, upper):
		
		while(True):
			mid = (lower + upper) / 2
			thres = (1 - self.floor * np.exp(-r * tao)) / self.floor
			tmp = self.option_price(S, mid, sigma, r, tao)[0] / mid
			
			if abs(tmp - thres) < 1e-7:
				#print("sucessfully")
				return mid
			if tmp < thres:
				upper = mid
			else:
				lower = mid
			
	def option_price(self, S, K, sigma, r, tao):
		"""
		Returns
		-------
		call option price & put option price

		"""
		d1 = (math.log(S/K) + (r + sigma ** 2 / 2) * tao) / (sigma * np.sqrt(tao))
		d2 = d1 - sigma * np.sqrt(tao)
		c = S * norm.cdf(d1) - K * np.exp(-r * tao) * norm.cdf(d2)
		p = K * np.exp(-r * tao) * norm.cdf(-d2) - S * norm.cdf(-d1)
		
		return c, p
	
	def plot(self, choice="port"):
		
		indices = range(0, self.num_periods)
		def plot_port():
			plt.plot(indices, self.port_val)
			plt.xlabel("time")
			plt.show()
		
		def plot_bond_and_equity():
			p_bond = plt.bar(indices, self.bond_holdings, color="blue")
			p_equity = plt.bar(indices, self.equity_holdings, bottom = self.bond_holdings, color = "red")
			
			plt.legend((p_bond, p_equity), ("Bond Value", "Equity Value"), loc = 3)
			plt.xlabel("time")
			plt.show()
			
		def plot_floor_and_cushion():
			p_floor = plt.bar(indices, self.floor_ts, color="blue")
			p_cushion = plt.bar(indices, self.port_val - self.floor_ts, bottom = self.floor_ts, color="red")
			plt.legend((p_floor, p_cushion), ("floor value", "cushion value"), loc=3)
			plt.xlabel("time")
			plt.show()
			
		switch = {'port':plot_port, 'bond_and_equity' : plot_bond_and_equity, 'floor_and_cushion':plot_floor_and_cushion}
		switch.get(choice,plot_port)()
		

if __name__ == '__main__':
	#simulated_risk_free = np.array([2,3,4,3,2,1,2,3,4,3])/100
	#simulated_equity_price = np.array([99,100,97,96,95,94,93,92,91,90])
	#vol = np.array([0.15,0.13,0.14,0.15,0.13,0.12,0.10,0.11,0.12,0.13])
	#floor = 0.8
	#principal = 3000
	
	data2 = pd.read_csv("data/OBPI_hist_data.csv")
	data2.columns = ["time", "Stock", "Cash"]
	data2["time"] = pd.to_datetime(data2["time"])
	data2["return"] = (data2["Stock"] - data2["Stock"].shift(1))/data2["Stock"].shift(1)
	data2["return_sqr"] = data2["return"] * data2["return"]
	data2["sigma"] = (data2["return_sqr"].rolling(10).sum())/10
	data2["sigma"] = data2["sigma"].map(lambda x: np.sqrt(x))
	
	
	equity_price = data2["Stock"].values[-360:]
	rf_returns = data2["Cash"].values[-360:] / 100
	vol = data2["sigma"].values[-360:] / 100
	floor = 0.8
	principal = 1000000
	
	obpi = OBPI(equity_price, rf_returns, vol, floor, principal)
	obpi.plot()
	obpi.plot("floor_and_cushion")
	#obpi.plot("floor_and_cushion")
	
	

	

	