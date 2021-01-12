#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 21:59:48 2021

@author: fuyutang
"""
import numpy as np
import math
from scipy.stats import norm
from strategy import PortfolioStrategy

class OBPI(PortfolioStrategy):
	
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
		
		self.port_val[0] = self.principal
		
		for t in range(0, self.num_periods):
			
			S = self.equity_price[t]
			r = self.risk_free[t]
			tao = self.num_periods - 1 - t
			sigma = self.vol[t]
			lower = 0  #wait for change
			upper = 3000 #wait for change
			
			K = self.cal_K(S, r, tao, sigma, lower, upper)
			d1 = (math.log(S/K) + (r + sigma ** 2 / 2) * tao) / (sigma * np.sqrt(tao))
			equity_weight = S * np.exp(-r * tao) * norm.cdf(d1) / (S + self.option_price(S, K, sigma, r, tao)[1])
			
			self.equity_holdings[t] = equity_weight * self.port_val[t]
			self.bond_holdings[t] = self.port_val[t] - self.equity_holdings[t]
			
			equity_return = (self.equity_price[t+1] - self.equity_price[t]) / self.equity_price[t]
			
			self.port_val[t+1] = self.port_val[t] + self.equity_holdings[t] * equity_return + self.bond_holdings[t] * self.risk_free[t]

	
	def cal_K(self, S, r, tao, sigma, lower, upper):
		
		while(True):
			mid = (lower + upper) / 2
			thres = (1 - self.floor * np.exp(-r * tao)) / self.floor
			tmp = self.option_price(S, mid, sigma, r, tao)[0] / mid
			
			if tmp - thres < 1e-7:
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
	

	