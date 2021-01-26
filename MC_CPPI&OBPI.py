#!/usr/bin/env python
# coding: utf-8

# ### MC_Main (CPPI&OBPI)

# In[20]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
import math
import seaborn as sns
from datetime import datetime


# In[2]:


class gCPPI(object):
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
                if t == 1:
                    self.multiple = np.zeros(self.num_periods)
                σ = self.vol[t-1]
                critical_value = σ*stats.norm.ppf(level)
                multiple = 1 / abs(critical_value)
                self.multiple[t-1] = multiple
                return multiple        
    
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

    def plot(self, choice = 'nav',indices = None, percent = True):
        if indices is None:
            indices = np.array(range(self.num_periods))
        def plot_nav(): 
            plt.plot(indices,self.nav)
            plt.show()
            
        def plot_bond_and_equity(percent = percent):
            rf_holding = self.rf_holding[1:]
            exposure = self.exposure[1:]
            if percent:
                rf_holding = rf_holding/self.nav[:-1]
                exposure = exposure/self.nav[:-1]
            p_bond = plt.plot(indices[1:], rf_holding, color = 'blue', label = 'RF Holding')
            p_equity = plt.plot(indices[1:], exposure, color = 'red', label = 'Risky Holding')
            plt.legend()
            plt.show()
            
        def plot_floor_and_cushion():
            p_floor = plt.plot(indices[:-1], self.floor[:-1], color = 'blue', label = 'floor')
            p_cushion = plt.plot(indices[:-1], self.nav[:-1],color = 'red', label = 'floor + cushion(nav)')
            plt.legend()
            plt.show()

        def plot_floor_and_margin_and_cushion():
            p_floor = plt.plot(indices[:-1], self.floor[:-1], color = 'blue', label = 'floor')
            p_margin = plt.plot(indices[:-1],self.margin[:-1]+self.floor[:-1], color = 'green', label = 'floor + margin')
            p_cushion = plt.plot(indices[:-1], self.nav[:-1],color = 'red', label = 'floor + margin + cushion(nav)')
            plt.legend()
            plt.show()


        switch = {'nav':plot_nav, 'bond_and_equity' : plot_bond_and_equity, 'floor_and_cushion':plot_floor_and_cushion, 'floor_and_margin_and_cushion':plot_floor_and_margin_and_cushion}
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
    
    vol = np.array([9,10,11,10,10,5,5,5,20,20])/100
    volcppi = gCPPI(simulated_equity_returns,simulated_bond_returns)
    volcppi.run(multiple_strategy = "input_vol", floor_strategy = "vanilla",multiple = multiple, floor = floor,vol = vol)
    """
    
    


# In[3]:


class gOBPI():
	
	def __init__(self, 
			  months: np.array, 
			  risky_asset_prices: np.array,
			  risky_asset_returns: np.array,
			  rf_asset_returns: np.array,
			  lower,
			  upper,
			  hist_vol,
			  vol_surface,
			  period = 6):
		
		#the length of input array must be larger than period and be multiple of period
		
		assert len(risky_asset_prices) == len(rf_asset_returns)
		self.months = months
		self.risky_asset_prices = risky_asset_prices
		self.risky_asset_returns = risky_asset_returns
		self.rf_asset_returns = rf_asset_returns
		self.lower = lower
		self.upper = upper
		self.period = period
		self.hist_vol = hist_vol
		self.vol_surface = vol_surface
		
		self.num_periods = len(risky_asset_returns)
		#for each period, the index of maturity in num_periods
		self.maturities_ind = np.arange(period-1, len(self.months)-1+period,period)
		self.floor = np.ones(self.num_periods)
		self.exposure = np.zeros(self.num_periods)
		self.rf_holding = np.zeros(self.num_periods)
		self.nav = np.ones(self.num_periods)
		self.best_nav = 1.0
		
		self.annualized_vols = np.zeros(self.num_periods)
		self.exercise_no = 0
		#the current chose option
		self.K = 0
	

		
	
	def run(self, vol_type, implement_type, floor_strategy, **kwargs):
		
		assert vol_type in ["hist", "implied"]
		assert implement_type in ["put_based", "call_based"]
		assert floor_strategy in ["vanilla", "dynamic"]

		def _get_vol(t, op_type, K, tao):
			
			if vol_type == "hist":
				return self.hist_vol[t-1] * np.sqrt(12) #because the input is std from monthly data
			else:
				if op_type == "put_based":
					vols = self.vol_surface[self.vol_surface["cp_flag"] == "P"]
				if op_type == "call_based":
					vols = self.vol_surface[self.vol_surface["cp_flag"] == "C"]
				month = self.months[t-1]
				exmonth = self.months[t-1+tao]
				expected_date_to_mat = tao * 30
				print(expected_date_to_mat)
				
				vols = vols[vols["month"] == month]
				vols = vols[vols["exmonth"] == exmonth]
				#vols.drop_duplicates(subset=["date", "month"], inplace=True)
				#vols.drop_duplicates(subset=["exdate", "exmonth"], inplace=True)
				#vols["date_to_mat"] = vols["exdate"].map(lambda x: datetime.strptime(x, "%Y%m%d")) - vols["date"].map(lambda x: datetime.strptime(x, "%Y%m%d"))
				#vols["date_to_mat"] = vols.map(lambad x: int(str(x)[0:2]))
				vols = vols[vols["dates_to_mat"] == expected_date_to_mat]
				
				if vols.shape[0] == 0:
					return self.hist_vol[t-1] * np.sqrt(12)
				
				vols["strike_diff"] = (vols["strike_price"] - K).map(lambda x: abs(x))
				ind = vols["strike_diff"].idxmin()
				return self.vol_surface.iloc[ind]["impl_volatility"]
		
		if floor_strategy == "vanilla":
			floor_ratio = kwargs.get("floor", 0.8)
			def _update_target_floor(t, r, tao):
				
								
				if (t-1) % self.period == 0:
					if t == 1:
						self.maturity_floor = self.nav[t-1] * floor_ratio
					else:
						if self.nav[t-1] > self.maturity_floor:
							#self.maturity_floor = max(self.maturity_floor, self.best_nav * floor_ratio)
							self.maturity_floor = self.maturity_floor * np.exp(6 * r)
						
					self.p = self.maturity_floor / self.nav[t-1]
					print("P:", self.p)
					"""
					if self.p > 1:
						
						#self.maturity_floor = self.nav[t-1] * floor_ratio
						#self.p = floor_ratio
						print(True)
						print(self.maturity_floor)
						print(self.nav[t-1])
						print(t)
						#print(self.rf_asset_returns[t])
					"""

				#print(self.maturity_floor)
				self.floor[t] = self.maturity_floor * np.exp(-r * tao)
			
			
				

		
		if floor_strategy == "dynamic":
			#Determine the expected floor for time t at the end of time t - 1
			floor_ratio = kwargs.get("floor", 0.8)
			def _update_target_floor(t, r, tao):

				
				if (t-1) % self.period == 0:
					if t == 1:
						self.maturity_floor = self.nav[t-1] * floor_ratio
					else:
						#self.maturity_floor = max(self.maturity_floor, self.best_nav * floor_ratio)
						self.maturity_floor = self.nav[t-1] * floor_ratio
						
					self.p = self.maturity_floor / self.nav[t-1]
					#print("P:", self.p)
					"""
					if self.p > 1:
						
						#self.maturity_floor = self.nav[t-1] * floor_ratio
						#self.p = floor_ratio
						print(True)
						print(self.maturity_floor)
						print(self.nav[t-1])
						print(t)
						#print(self.rf_asset_returns[t])
					"""

				#print(self.maturity_floor)
						
				self.floor[t] = self.maturity_floor * np.exp(-r * tao)
					

		if implement_type == "put_based":
			
			floor_ratio = kwargs.get("floor", 0.8)
			def _implement(S, r, sigma, tao, t):
				
				
				if tao == 0:
					#update exercise
					if self.K > S:
						self.exercise_no += 1
					
					self.exposure[t] = self.exposure[t-1]
					self.rf_holding[t] = self.rf_holding[t-1]
					
				else:
					
					#calculate strike price K
					thres = (1 - self.p * np.exp(-r * tao))/ self.p
					K = self.calc_K(S, r * 12, tao/12, sigma * np.sqrt(12), self.lower, self.upper, thres)
					self.K = K
					
					if K == self.upper:
						self.exposure[t] = 0
						self.rf_holding[t] = self.nav[t-1]
						
					else:
						#hist or implied vol
						annualized_vol = _get_vol(t, implement_type, K, tao)
						self.annualized_vols[t-1] = annualized_vol 
						
						#print(self.option_price(S, K, annualized_vol, r*12, tao/12)[1])
						
						q = self.nav[t-1] / (S + self.option_price(S, K, annualized_vol, r*12, tao/12)[1])
						d1 = (math.log(S/K) + (r*12 + annualized_vol ** 2 / 2) * tao/12) / (annualized_vol * np.sqrt(tao/12))
						d2 = d1 - annualized_vol * np.sqrt(tao/12)
						
						equity_holding = norm.cdf(d1) * S * q
						bond_holding = K * norm.cdf(-d2) * np.exp(-r*tao) * q
						
						#equity_weight = (equity_holding + bond_holding)
						
						
						#print(equity_holding)
						#print(bond_holding)
						
						self.exposure[t] = equity_holding
						self.rf_holding[t] = bond_holding
					
		if implement_type == "call_based":
			floor_ratio = kwargs.get("floor", 0.8)
			def _implement(S, r, sigma, tao, t):
							
				if tao == 0:
					if S > self.K:
						self.exercise_no += 1
					self.exposure[t] = self.exposure[t-1]
					self.rf_holding[t] = self.rf_holding[t-1]
					
				else:
					thres = (self.nav[t-1] - self.floor[t-1])/self.maturity_floor
					K = self.calc_K(S,r * 12, tao/12, sigma * np.sqrt(12), self.lower, self.upper, thres)
					self.K = K
					n = self.maturity_floor / K
					
					
					if K == self.upper:
						self.exposure[t] = 0
						self.rf_holding[t] = self.nav[t-1]
					else:
	
						#hist or implied vol
						annualized_vol = _get_vol(t, implement_type, K, tao)
						self.annualized_vols[t-1] = annualized_vol 
							
						#print(self.option_price(S, K, annualized_vol, r*12, tao/12)[1])
							
						d1 = (math.log(S/K) + (r*12 + annualized_vol ** 2 / 2) * tao/12) / (annualized_vol * np.sqrt(tao/12))
						d2 = d1 - annualized_vol * np.sqrt(tao/12)
						
						equity_holding = n * norm.cdf(d1) * S
						rf_holding = self.floor[t-1] - K * np.exp(-r * tao) * norm.cdf(d2) * n
						
						#problem
						self.exposure[t] = equity_holding #/ (equity_holding + rf_holding) * self.nav[t-1]
						self.rf_holding[t] = rf_holding #/ (equity_holding + rf_holding) * self.nav[t-1]
					
					
					
						#print(equity_holding+rf_holding)
						#print(self.nav[t-1])
				
		
		floor_ratio = kwargs.get("floor", 0.8)
		self.maturity_floor = self.nav[0] * floor_ratio
		self.p = floor_ratio
				
				
		
		for t in range(1, self.num_periods):
			
			#Set positions for time t at the end of time t - 1
			
			cur_maturity_ind = math.floor(int(t-1) / int(self.period))
			cur_maturity = self.maturities_ind[cur_maturity_ind]
					
			S = self.risky_asset_prices[t-1]
			r = self.rf_asset_returns[t-1]
			sigma = self.hist_vol[t-1]
			tao = (cur_maturity - t + 1) #months to maturity
			
			_update_target_floor(t, r, tao) #r is monthly and tao is months to maturity

			#implement the strategy
			_implement(S, r, sigma, tao, t)
			
			self.nav[t] = self.nav[t-1] + self.exposure[t] * self.risky_asset_returns[t] + self.rf_holding[t] * self.rf_asset_returns[t]
			if self.nav[t] > self.best_nav:
				self.best_nav = self.nav[t]

	
	def calc_K(self, S, r, tao, sigma, lower, upper, thres):
		
		if thres < 0:
			
			K = upper
			return K
		
		while(True):
			mid = (lower + upper) / 2
			tmp = self.option_price(S, mid, sigma, r, tao)[0] / mid
			
			if abs(tmp - thres) < 1e-7:
				return mid
			if tmp < thres:
				upper = mid
			else:
				lower = mid
				
	def option_price(self, S, K, sigma, r, tao):

		d1 = (math.log(S/K) + (r + sigma ** 2 / 2) * tao) / (sigma * np.sqrt(tao))
		d2 = d1 - sigma * np.sqrt(tao)
		c = S * norm.cdf(d1) - K * np.exp(-r * tao) * norm.cdf(d2)
		p = K * np.exp(-r * tao) * norm.cdf(-d2) - S * norm.cdf(-d1)
		
		return c, p
	
	def plot(self, choice="nav"):
		
		indices = range(0, self.num_periods)
		xs = [datetime.strptime(str(m), '%Y%m').date() for m in self.months]
		def plot_port():
			scale_price = self.risky_asset_prices/self.risky_asset_prices[0]
			plt.plot(xs, self.nav, color = "red", label="portfolio_value")
			plt.plot(xs, scale_price, color = "blue", label="market_risky_return")
			plt.legend()
			plt.xlabel("time")
			plt.show()
		
		def plot_bond_and_equity():
			plt.plot(xs, self.exposure, color = "red", label = "exposure")
			plt.plot(xs, self.rf_holding, color = "blue", label = "rf_holding")
			plt.legend()
			plt.xlabel("time")
			plt.show()
			
		def plot_floor_and_cushion():
			plt.plot(xs, self.nav, color = "red", label = "portfolio_value")
			plt.plot(xs, self.floor, color = "blue", label = "floor")
			plt.legend()
			plt.xlabel("time")
			plt.show()
		"""	
		def plot_implicit_cost():
			plt.plot(indices, self.implicit_cost)
			plt.xlabel("time")
			plt.show()
		"""
			
		switch = {'port':plot_port, 'bond_and_equity' : plot_bond_and_equity, 'floor_and_cushion':plot_floor_and_cushion}
		switch.get(choice,plot_port)()


# In[4]:


def SR(nav:np.array, rf_rate:np.array):
    assert len(nav) == len(rf_rate)
    returns = np.zeros(len(nav)-1)
    for i in range(len(nav) - 1):
        returns[i] = nav[i+1] / nav[i] - 1 - rf_rate[i+1]
    annualized_mean = returns.mean() * 12
    annualized_std = returns.std() * np.sqrt(12)
    return annualized_mean / annualized_std


# #### Historical Data Input

# In[5]:


data = pd.read_excel("hist_data.xlsx")
data.columns = ["time","Stock","Bond","Cash"]
data["time"] = pd.to_datetime(data["time"])
data = data.set_index('time')
data = data.resample('1M').last()

data['Stock'] = data['Stock'].values /100
data['Bond'] = data['Bond'].values /100
data['Cash'] = data['Cash'].values / 100


# #### Set Parameters

# In[6]:


#data['Stock_mu'] = data['Stock'].rolling(12).apply(lambda x:np.mean(x),raw = False)/12
#data['Stock_sigma'] = data['Stock'].rolling(12).apply(lambda x:np.std(x),raw = False)
#mu_max_stock = max(data['Stock_mu'].iloc[-360:])
#mu_min_stock = min(data['Stock_mu'].iloc[-360:])
#mubp1 = (mu_max_stock - mu_min_stock)/3+mu_min_stock
#mubp2 = (mu_max_stock - mu_min_stock)*2/3+mu_min_stock
#sigma_max_stock = max(data['Stock_sigma'].iloc[-360:])
#sigma_min_stock = min(data['Stock_sigma'].iloc[-360:])
#sigmabp1 = (sigma_max_stock - sigma_min_stock)/3+sigma_min_stock
#sigmabp2 = (sigma_max_stock - sigma_min_stock)*2/3+sigma_min_stock

#data['Bond_mu'] = data['Bond'].rolling(12).apply(lambda x:np.mean(x),raw = False)/12
#data['Bond_sigma'] = data['Bond'].rolling(12).apply(lambda x:np.std(x),raw = False)
#mu_max_bond = max(data['Bond_mu'].iloc[-360:])
#mu_min_bond = min(data['Bond_mu'].iloc[-360:])
#sigma_max_bond = max(data['Bond_sigma'].iloc[-360:])
#sigma_min_bond = min(data['Bond_sigma'].iloc[-360:])

#data['Cash_mu'] = data['Cash'].rolling(12).apply(lambda x:np.mean(x),raw = False)/12
#data['Cash_sigma'] = data['Cash'].rolling(12).apply(lambda x:np.std(x),raw = False)
#mu_max_cash = max(data['Cash_mu'].iloc[-360:])
#mu_min_cash = min(data['Cash_mu'].iloc[-360:])
#sigma_max_cash = max(data['Cash_sigma'].iloc[-360:])
#sigma_min_cash = min(data['Cash_sigma'].iloc[-360:])

mu_stock = np.mean(data['Stock'].rolling(12).apply(lambda x:np.mean(x),raw = False)/12)
sigma_stock = np.std(data['Stock'].iloc[-360:])
mu_max_stock = mu_stock + 0.005
mu_min_stock = mu_stock - 0.005
sigma_max_stock = sigma_stock * 5
sigma_min_stock = sigma_stock 

mu_bond = np.mean(data['Bond'].iloc[-360:])
sigma_bond = np.std(data['Bond'].iloc[-360:])
mu_max_bond = mu_bond * 2
mu_min_bond = mu_bond / 2
sigma_max_bond = sigma_bond * 2
sigma_min_bond = sigma_bond / 2

mu_cash = np.mean(data['Cash'].iloc[-360:])
sigma_cash = np.std(data['Cash'].iloc[-360:])
mu_max_cash = mu_cash * 2
mu_min_cash = mu_cash / 2
sigma_max_cash = sigma_cash * 2
sigma_min_cash = sigma_cash / 2
print(mu_stock,sigma_stock)

mubp1 = (mu_max_stock - mu_min_stock)/3+mu_min_stock
mubp2 = (mu_max_stock - mu_min_stock)*2/3+mu_min_stock
sigmabp1 = (sigma_max_stock - sigma_min_stock)/3+sigma_min_stock
sigmabp2 = (sigma_max_stock - sigma_min_stock)*2/3+sigma_min_stock


# In[7]:


data['Stock_sigma'] = data['Stock'].rolling(12).apply(lambda x:np.std(x),raw = False)
plt.plot(data.index,data['Stock_sigma'])


# Stock: -0.005-0.005, 0.05-0.2
# Bond: 0.002-0.008, 0.005-0.02
# Cash: 0.001-0.006, 0.001-0.002

# In[8]:


print(mu_min_stock,mu_max_stock,sigma_min_stock,sigma_max_stock)
print(mu_min_bond,mu_max_bond,sigma_min_bond,sigma_max_bond)
print(mu_min_cash,mu_max_cash,sigma_min_cash,sigma_max_cash)


# #### MC Simulation

# In[9]:


def MCdata(simnum,T,mu_min,mu_max,sigma_min,sigma_max):
    
    simdata = np.zeros((simnum,T))
    sigmalist = []
    
    for i in range(simnum):
        mu = random.uniform(mu_min,mu_max)
        sigma = random.uniform(sigma_min,sigma_max)
        sigmalist.append(sigma)
        simdata[i] = np.random.normal(mu, sigma, T)
    
    return simdata, sigmalist


# In[10]:


simnum = 1000
T = 360
#mu_min_stock = 0.015
#mu_max_stock = 0.020
#sigma_min_stock = 0.090
#sigma_max_stock = 0.12
#simstock = MCdata(simnum,T,mu_min_stock,mu_max_stock,sigma_min_stock,sigma_max_stock)[0]
#mu_min_bond = 0.002
#mu_max_bond = 0.008
#sigma_min_bond = 0.005
#sigma_max_bond = 0.020
random.seed(321)
simbond = MCdata(simnum,T,mu_min_bond,mu_max_bond,sigma_min_bond,sigma_max_bond)[0]
#mu_min_cash = 0.001
#mu_max_cash = 0.006
#sigma_min_cash = 0.001
#sigma_max_cash = 0.002
random.seed(321)
simcash = MCdata(simnum,T,mu_min_cash,mu_max_cash,sigma_min_cash,sigma_max_cash)[0]

#simhmhs,sigmahmhs = MCdata(simnum,T,0.003,0.005,0.150,0.200)
#simhmms,sigmahmms = MCdata(simnum,T,0.003,0.005,0.100,0.150)
#simhmls,sigmahmls = MCdata(simnum,T,0.003,0.005,0.050,0.100)
#simmmhs,sigmammhs = MCdata(simnum,T,-0.003,0.003,0.150,0.200)
#simmmms,sigmammms = MCdata(simnum,T,-0.003,0.003,0.100,0.150)
#simmmls,sigmammls = MCdata(simnum,T,-0.003,0.003,0.050,0.100)
#simlmhs,sigmalmhs = MCdata(simnum,T,-0.005,-0.003,0.150,0.200)
#simlmms,sigmalmms = MCdata(simnum,T,-0.005,-0.003,0.100,0.150)
#simlmls,sigmalmls = MCdata(simnum,T,-0.005,-0.003,0.050,0.100)
random.seed(321)
simhmhs,sigmahmhs = MCdata(simnum,T,mubp2,mu_max_stock,sigmabp2,sigma_max_stock)
random.seed(321)
simhmms,sigmahmms = MCdata(simnum,T,mubp2,mu_max_stock,sigmabp1,sigmabp2)
random.seed(321)
simhmls,sigmahmls = MCdata(simnum,T,mubp2,mu_max_stock,sigma_min_stock,sigmabp1)
random.seed(321)
simmmhs,sigmammhs = MCdata(simnum,T,mubp1,mubp2,sigmabp2,sigma_max_stock)
random.seed(321)
simmmms,sigmammms = MCdata(simnum,T,mubp1,mubp2,sigmabp1,sigmabp2)
random.seed(321)
simmmls,sigmammls = MCdata(simnum,T,mubp1,mubp2,sigma_min_stock,sigmabp1)
random.seed(321)
simlmhs,sigmalmhs = MCdata(simnum,T,mu_min_stock,mubp2,sigmabp2,sigma_max_stock)
random.seed(321)
simlmms,sigmalmms = MCdata(simnum,T,mu_min_stock,mubp2,sigmabp1,sigmabp2)
random.seed(321)
simlmls,sigmalmls = MCdata(simnum,T,mu_min_stock,mubp2,sigma_min_stock,sigmabp1)


# #### A0 - Basic CPPI

# In[11]:


multiple = 5
floor = 0.7
cppi_hmhs=np.empty(shape=[0,T],dtype=float)
cppi_hmms=np.empty(shape=[0,T],dtype=float)
cppi_hmls=np.empty(shape=[0,T],dtype=float)
cppi_mmhs=np.empty(shape=[0,T],dtype=float)
cppi_mmms=np.empty(shape=[0,T],dtype=float)
cppi_mmls=np.empty(shape=[0,T],dtype=float)
cppi_lmhs=np.empty(shape=[0,T],dtype=float)
cppi_lmms=np.empty(shape=[0,T],dtype=float)
cppi_lmls=np.empty(shape=[0,T],dtype=float)
for i in range(simnum):
    cppihmhs = gCPPI(simhmhs[i],simcash[i])
    cppihmhs.run(multiple_strategy = "constant", floor_strategy = "vanilla",multiple = multiple, floor = floor)
    cppi_hmhs = np.append(cppi_hmhs,[cppihmhs.nav],axis=0)
    cppihmms = gCPPI(simhmms[i],simcash[i])
    cppihmms.run(multiple_strategy = "constant", floor_strategy = "vanilla",multiple = multiple, floor = floor)
    cppi_hmms = np.append(cppi_hmms,[cppihmms.nav],axis=0)
    cppihmls = gCPPI(simhmls[i],simcash[i])
    cppihmls.run(multiple_strategy = "constant", floor_strategy = "vanilla",multiple = multiple, floor = floor)
    cppi_hmls = np.append(cppi_hmls,[cppihmls.nav],axis=0)
    cppimmhs = gCPPI(simmmhs[i],simcash[i])
    cppimmhs.run(multiple_strategy = "constant", floor_strategy = "vanilla",multiple = multiple, floor = floor)
    cppi_mmhs = np.append(cppi_mmhs,[cppimmhs.nav],axis=0)
    cppimmms = gCPPI(simmmms[i],simcash[i])
    cppimmms.run(multiple_strategy = "constant", floor_strategy = "vanilla",multiple = multiple, floor = floor)
    cppi_mmms = np.append(cppi_mmms,[cppimmms.nav],axis=0)
    cppimmls = gCPPI(simmmls[i],simcash[i])
    cppimmls.run(multiple_strategy = "constant", floor_strategy = "vanilla",multiple = multiple, floor = floor)
    cppi_mmls = np.append(cppi_mmls,[cppimmls.nav],axis=0)
    cppilmhs = gCPPI(simlmhs[i],simcash[i])
    cppilmhs.run(multiple_strategy = "constant", floor_strategy = "vanilla",multiple = multiple, floor = floor)
    cppi_lmhs = np.append(cppi_lmhs,[cppilmhs.nav],axis=0)
    cppilmms = gCPPI(simlmms[i],simcash[i])
    cppilmms.run(multiple_strategy = "constant", floor_strategy = "vanilla",multiple = multiple, floor = floor)
    cppi_lmms = np.append(cppi_lmms,[cppilmms.nav],axis=0)
    cppilmls = gCPPI(simlmls[i],simcash[i])
    cppilmls.run(multiple_strategy = "constant", floor_strategy = "vanilla",multiple = multiple, floor = floor)
    cppi_lmls = np.append(cppi_lmls,[cppilmls.nav],axis=0)
cppi_mc = pd.DataFrame({'Date':np.arange(T),
                        'CPPI_hmhs':np.mean(cppi_hmhs,axis=0),'CPPI_hmms':np.mean(cppi_hmms,axis=0),'CPPI_hmls':np.mean(cppi_hmls,axis=0),
                        'CPPI_mmhs':np.mean(cppi_mmhs,axis=0),'CPPI_mmms':np.mean(cppi_mmms,axis=0),'CPPI_mmls':np.mean(cppi_mmls,axis=0),
                        'CPPI_lmhs':np.mean(cppi_lmhs,axis=0),'CPPI_lmms':np.mean(cppi_lmms,axis=0),'CPPI_lmls':np.mean(cppi_lmls,axis=0)})


# In[12]:


plt.plot(cppi_mc['Date'],cppi_mc.iloc[:,1:])
plt.legend(cppi_mc.columns[1:])
plt.show()


# #### A1 - Dynamic CPPI

# In[13]:


multiple_strategy = "constant"            #@param {type:"string"}
floor_strategy = "dynamic"                #@param {type:"string"}
multiple = 5                              #@param {type:"number"}
floor = 0.7                               #@param {type:"number"}
dcppi_hmhs=np.empty(shape=[0,T],dtype=float)
dcppi_hmms=np.empty(shape=[0,T],dtype=float)
dcppi_hmls=np.empty(shape=[0,T],dtype=float)
dcppi_mmhs=np.empty(shape=[0,T],dtype=float)
dcppi_mmms=np.empty(shape=[0,T],dtype=float)
dcppi_mmls=np.empty(shape=[0,T],dtype=float)
dcppi_lmhs=np.empty(shape=[0,T],dtype=float)
dcppi_lmms=np.empty(shape=[0,T],dtype=float)
dcppi_lmls=np.empty(shape=[0,T],dtype=float)
for i in range(simnum):  
    dcppihmhs = gCPPI(simhmhs[i],simcash[i])
    dcppihmhs.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor)
    dcppi_hmhs= np.append(dcppi_hmhs,[dcppihmhs.nav],axis=0)
    dcppihmms = gCPPI(simhmms[i],simcash[i])
    dcppihmms.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor)
    dcppi_hmms= np.append(dcppi_hmms,[dcppihmhs.nav],axis=0)
    dcppihmls = gCPPI(simhmls[i],simcash[i])
    dcppihmls.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor)
    dcppi_hmls= np.append(dcppi_hmls,[dcppihmls.nav],axis=0)
    dcppimmhs = gCPPI(simmmhs[i],simcash[i])
    dcppimmhs.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor)
    dcppi_mmhs= np.append(dcppi_mmhs,[dcppimmhs.nav],axis=0)
    dcppimmms = gCPPI(simmmms[i],simcash[i])
    dcppimmms.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor)
    dcppi_mmms= np.append(dcppi_mmms,[dcppimmms.nav],axis=0)
    dcppimmls = gCPPI(simmmls[i],simcash[i])
    dcppimmls.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor)
    dcppi_mmls= np.append(dcppi_mmls,[dcppimmls.nav],axis=0)
    dcppilmhs = gCPPI(simlmhs[i],simcash[i])
    dcppilmhs.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor)
    dcppi_lmhs= np.append(dcppi_lmhs,[dcppilmhs.nav],axis=0)
    dcppilmms = gCPPI(simlmms[i],simcash[i])
    dcppilmms.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor)
    dcppi_lmms= np.append(dcppi_lmms,[dcppilmms.nav],axis=0)
    dcppilmls = gCPPI(simlmls[i],simcash[i])
    dcppilmls.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor)
    dcppi_lmls= np.append(dcppi_lmls,[dcppilmls.nav],axis=0)
dcppi_mc = pd.DataFrame({'Date':np.arange(T),
                        'dCPPI_hmhs':np.mean(dcppi_hmhs,axis=0),'dCPPI_hmms':np.mean(dcppi_hmms,axis=0),'dCPPI_hmls':np.mean(dcppi_hmls,axis=0),
                        'dCPPI_mmhs':np.mean(dcppi_mmhs,axis=0),'dCPPI_mmms':np.mean(dcppi_mmms,axis=0),'dCPPI_mmls':np.mean(dcppi_mmls,axis=0),
                        'dCPPI_lmhs':np.mean(dcppi_lmhs,axis=0),'dCPPI_lmms':np.mean(dcppi_lmms,axis=0),'dCPPI_lmls':np.mean(dcppi_lmls,axis=0)})


# In[14]:


plt.plot(dcppi_mc['Date'],dcppi_mc.iloc[:,1:])
plt.legend(dcppi_mc.columns[1:])
plt.show()


# #### A2 - Dynamic Double CPPI

# In[15]:


multiple_strategy = "constant"            #@param {type:"string"}
floor_strategy = "dynamic double"         #@param {type:"string"}
multiple = 5                              #@param {type:"number"}
floor = 0.7                               #@param {type:"number"}
cushion = 0.1                             #@param {type:"number"}


d2cppi_hmhs=np.empty(shape=[0,T],dtype=float)
d2cppi_hmms=np.empty(shape=[0,T],dtype=float)
d2cppi_hmls=np.empty(shape=[0,T],dtype=float)
d2cppi_mmhs=np.empty(shape=[0,T],dtype=float)
d2cppi_mmms=np.empty(shape=[0,T],dtype=float)
d2cppi_mmls=np.empty(shape=[0,T],dtype=float)
d2cppi_lmhs=np.empty(shape=[0,T],dtype=float)
d2cppi_lmms=np.empty(shape=[0,T],dtype=float)
d2cppi_lmls=np.empty(shape=[0,T],dtype=float)
for i in range(simnum):  
    d2cppihmhs = gCPPI(simhmhs[i],simcash[i])
    d2cppihmhs.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor, cushion = cushion)
    d2cppi_hmhs= np.append(d2cppi_hmhs,[d2cppihmhs.nav],axis=0)
    d2cppihmms = gCPPI(simhmms[i],simcash[i])
    d2cppihmms.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor, cushion = cushion)
    d2cppi_hmms= np.append(d2cppi_hmms,[d2cppihmhs.nav],axis=0)
    d2cppihmls = gCPPI(simhmls[i],simcash[i])
    d2cppihmls.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor, cushion = cushion)
    d2cppi_hmls= np.append(d2cppi_hmls,[d2cppihmls.nav],axis=0)
    d2cppimmhs = gCPPI(simmmhs[i],simcash[i])
    d2cppimmhs.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor, cushion = cushion)
    d2cppi_mmhs= np.append(d2cppi_mmhs,[d2cppimmhs.nav],axis=0)
    d2cppimmms = gCPPI(simmmms[i],simcash[i])
    d2cppimmms.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor, cushion = cushion)
    d2cppi_mmms= np.append(d2cppi_mmms,[d2cppimmms.nav],axis=0)
    d2cppimmls = gCPPI(simmmls[i],simcash[i])
    d2cppimmls.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor, cushion = cushion)
    d2cppi_mmls= np.append(d2cppi_mmls,[d2cppimmls.nav],axis=0)
    d2cppilmhs = gCPPI(simlmhs[i],simcash[i])
    d2cppilmhs.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor, cushion = cushion)
    d2cppi_lmhs= np.append(d2cppi_lmhs,[d2cppilmhs.nav],axis=0)
    d2cppilmms = gCPPI(simlmms[i],simcash[i])
    d2cppilmms.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor, cushion = cushion)
    d2cppi_lmms= np.append(d2cppi_lmms,[d2cppilmms.nav],axis=0)
    d2cppilmls = gCPPI(simlmls[i],simcash[i])
    d2cppilmls.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor, cushion = cushion)
    d2cppi_lmls= np.append(d2cppi_lmls,[d2cppilmls.nav],axis=0)
d2cppi_mc = pd.DataFrame({'Date':np.arange(T),
                        'd2CPPI_hmhs':np.mean(d2cppi_hmhs,axis=0),'d2CPPI_hmms':np.mean(d2cppi_hmms,axis=0),'d2CPPI_hmls':np.mean(d2cppi_hmls,axis=0),
                        'd2CPPI_mmhs':np.mean(d2cppi_mmhs,axis=0),'d2CPPI_mmms':np.mean(d2cppi_mmms,axis=0),'d2CPPI_mmls':np.mean(d2cppi_mmls,axis=0),
                        'd2CPPI_lmhs':np.mean(d2cppi_lmhs,axis=0),'d2CPPI_lmms':np.mean(d2cppi_lmms,axis=0),'d2CPPI_lmls':np.mean(d2cppi_lmls,axis=0)})


# In[16]:


plt.plot(d2cppi_mc['Date'],d2cppi_mc.iloc[:,1:])
plt.legend(d2cppi_mc.columns[1:])
plt.show()


# #### A3 - TIme-Variant Multiple CPPI

# In[17]:


multiple_strategy = "time-variant"        #@param {type:"string"}
floor_strategy = "dynamic double"         #@param {type:"string"}
floor = 0.7                               #@param {type:"number"}
cushion = 0.1                             #@param {type:"number"}
#input_vol = sigma                         #@param {type:"raw"}
level = 0.01                              #@param {type:"number"}

d2vcppi_hmhs=np.empty(shape=[0,T],dtype=float)
d2vcppi_hmms=np.empty(shape=[0,T],dtype=float)
d2vcppi_hmls=np.empty(shape=[0,T],dtype=float)
d2vcppi_mmhs=np.empty(shape=[0,T],dtype=float)
d2vcppi_mmms=np.empty(shape=[0,T],dtype=float)
d2vcppi_mmls=np.empty(shape=[0,T],dtype=float)
d2vcppi_lmhs=np.empty(shape=[0,T],dtype=float)
d2vcppi_lmms=np.empty(shape=[0,T],dtype=float)
d2vcppi_lmls=np.empty(shape=[0,T],dtype=float)

for i in range(simnum): 
    d2vcppihmhs = gCPPI(simhmhs[i],simcash[i])
    input_vol = np.full((T,1),sigmahmhs[i])
    d2vcppihmhs.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor, cushion = cushion, vol = input_vol, level = level)
    d2vcppi_hmhs= np.append(d2vcppi_hmhs,[d2vcppihmhs.nav],axis=0)
    d2vcppihmms = gCPPI(simhmms[i],simcash[i])
    input_vol = np.full((T,1),sigmahmms[i])
    d2vcppihmms.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor, cushion = cushion, vol = input_vol, level = level)
    d2vcppi_hmms= np.append(d2vcppi_hmms,[d2vcppihmhs.nav],axis=0)
    d2vcppihmls = gCPPI(simhmls[i],simcash[i])
    input_vol = np.full((T,1),sigmahmls[i])
    d2vcppihmls.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor, cushion = cushion, vol = input_vol, level = level)
    d2vcppi_hmls= np.append(d2vcppi_hmls,[d2vcppihmls.nav],axis=0)
    d2vcppimmhs = gCPPI(simmmhs[i],simcash[i])
    input_vol = np.full((T,1),sigmammhs[i])
    d2vcppimmhs.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor, cushion = cushion, vol = input_vol, level = level)
    d2vcppi_mmhs= np.append(d2vcppi_mmhs,[d2vcppimmhs.nav],axis=0)
    d2vcppimmms = gCPPI(simmmms[i],simcash[i])
    input_vol = np.full((T,1),sigmammms[i])
    d2vcppimmms.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor, cushion = cushion, vol = input_vol, level = level)
    d2vcppi_mmms= np.append(d2vcppi_mmms,[d2vcppimmms.nav],axis=0)
    d2vcppimmls = gCPPI(simmmls[i],simcash[i])
    input_vol = np.full((T,1),sigmammls[i])
    d2vcppimmls.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor, cushion = cushion, vol = input_vol, level = level)
    d2vcppi_mmls= np.append(d2vcppi_mmls,[d2vcppimmls.nav],axis=0)
    d2vcppilmhs = gCPPI(simlmhs[i],simcash[i])
    input_vol = np.full((T,1),sigmalmhs[i])
    d2vcppilmhs.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor, cushion = cushion, vol = input_vol, level = level)
    d2vcppi_lmhs= np.append(d2vcppi_lmhs,[d2vcppilmhs.nav],axis=0)
    d2vcppilmms = gCPPI(simlmms[i],simcash[i])
    input_vol = np.full((T,1),sigmalmms[i])
    d2vcppilmms.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor, cushion = cushion, vol = input_vol, level = level)
    d2vcppi_lmms= np.append(d2vcppi_lmms,[d2vcppilmms.nav],axis=0)
    d2vcppilmls = gCPPI(simlmls[i],simcash[i])
    input_vol = np.full((T,1),sigmalmls[i])
    d2vcppilmls.run(multiple_strategy = multiple_strategy, floor_strategy = floor_strategy, multiple = multiple, floor = floor, cushion = cushion, vol = input_vol, level = level)
    d2vcppi_lmls= np.append(d2vcppi_lmls,[d2vcppilmls.nav],axis=0)
d2vcppi_mc = pd.DataFrame({'Date':np.arange(T),
                           'd2vCPPI_hmhs':d2vcppihmhs.nav,'d2vCPPI_hmms':d2vcppihmms.nav,'d2vCPPI_hmls':d2vcppihmls.nav,
                           'd2vCPPI_mmhs':d2vcppimmhs.nav,'d2vCPPI_mmms':d2vcppimmms.nav,'d2vCPPI_mmls':d2vcppimmls.nav,
                           'd2vCPPI_lmhs':d2vcppilmhs.nav,'d2vCPPI_lmms':d2vcppilmms.nav,'d2vCPPI_lmls':d2vcppilmls.nav})


# In[18]:


plt.plot(d2vcppi_mc['Date'],d2vcppi_mc.iloc[:,1:])
plt.legend(d2vcppi_mc.columns[1:])
plt.show()


# #### B0 - Basic OBPI (Put-based)

# In[21]:


if __name__ == '__main__':
	#simulated_rf_rates = np.array([2,3,4,3,2,1,2,3,4,3])/100
	#simulated_equity_prices = np.array([99,100,97,96,95,94,93,92,91,90])
	#hist_vol = np.array([0.15,0.13,0.14,0.15,0.13,0.12,0.10,0.11,0.12,0.13])
	#percent = 0.8
	#principal = 3000

	data2 = pd.read_csv("monthly_data2.csv")
	data2.columns = ["time", "Stock", "Return", "Cash"]
	data2["time"] = pd.to_datetime(data2["time"])
	data2["time"] = data2["time"].map(lambda x: str(x))
	data2["month"] = data2["time"].map(lambda x: x[0:4]+x[5:7])
	data2["time"] = data2["time"].map(lambda x: x[0:4]+x[5:7]+x[8:10])
	#data2["time"]= data2["time"].map(lambda x: int(x))
	#data2.drop_duplicates(subset=["month"], keep="last", inplace=True)
	
	
	#data2["return"] = (data2["Stock"] - data2["Stock"].shift(1))/data2["Stock"].shift(1)
	data2["sigma"] = ((data2["Return"] * data2["Return"]).rolling(10).sum())/10
	#data2["sigma"] = (data2["return_sqr"].rolling(10).sum())/10
	data2["sigma"] = data2["sigma"].map(lambda x: np.sqrt(x))
	
	#cash_data = pd.read_csv("data/Cash.csv")
	#cash_data.index = data2.index
	#data2["Cash"] = cash_data["Cash"]
	
	#data2.dropna(how="any", inplace=True)
	
	
	equity_prices = data2["Stock"].values[-252:]
	equity_returns = data2["Return"].values[-252:]
	rf_returns = data2["Cash"].values[-252:] / 100
	hist_vol = data2["sigma"].values[-252:] 
	dates = data2["time"].values[-252:]
	
	#floor_ratio = 1.0
	
	#vol_surface = pd.read_csv("data/monthly_data2.csv")
	vol_30 = pd.read_csv("data/option_data/option_price_30.csv",dtype = {"date": str, "exdate":str})
	vol_30["dates_to_mat"] = 30
	vol_60 = pd.read_csv("data/option_data/option_price_60.csv",dtype = {"date": str, "exdate":str})
	vol_60["dates_to_mat"] = 60
	vol_90 = pd.read_csv("data/option_data/option_price_90.csv",dtype = {"date": str, "exdate":str})
	vol_90["dates_to_mat"] = 90
	vol_120 = pd.read_csv("data/option_data/option_price_120.csv",dtype = {"date": str, "exdate":str})
	vol_120["dates_to_mat"] = 120
	vol_150 = pd.read_csv("data/option_data/option_price_150.csv",dtype = {"date": str, "exdate":str})
	vol_150["dates_to_mat"] = 150
	
	vol_surface = pd.concat([vol_30,vol_60,vol_90,vol_120,vol_150],axis=0)
	vol_surface["month"] = vol_surface["date"].map(lambda x: x[0:6])
	vol_surface["exmonth"] = vol_surface["exdate"].map(lambda x: x[0:6])
	vol_surface["strike_price"] = vol_surface["strike_price"] / 1000
	vol_surface.dropna(how="any", inplace=True)
	

	months = [date[0:6] for date in dates]

	
	
	obpi = gOBPI(months, equity_prices, equity_returns, rf_returns, 0, 10000, hist_vol, vol_surface)
	obpi.run(vol_type="hist", implement_type = "put_based", floor_strategy = "dynamic", floor = 0.95)
	obpi.plot()
	#obpi.plot("floor_and_cushion")
	obpi.plot("floor_and_cushion")
	
	print(obpi.nav.min())
	print(obpi.floor.min())
	print((obpi.nav > obpi.floor).sum() / len(obpi.nav))
	
	ind = obpi.maturities_ind
	port = obpi.nav[ind]
	floor = obpi.floor[ind]
	print(((port > floor).sum()) / len(floor))


# #### Ratios

# In[22]:


def psr(nav,floor):
    psr = (nav > floor).sum() / len(nav)
    return psr

def psrm(nav,floor,ind):
    port = nav[ind]
    floor = floor[ind]
    return ((port > floor).sum()) / len(floor)

def wabdm(nav,floor):
	diff = (nav - floor) / floor
	for i in range(0, len(diff)):
		if diff[i] > 0:
			diff[i] = 0
	avr_break = diff.sum() / 42
	return avr_break


# In[23]:


s0 = data2["Stock"][0]
bsimnum = 100
T = 252
#mu_min_stock = 0.015
#mu_max_stock = 0.020
#sigma_min_stock = 0.090
#sigma_max_stock = 0.12
#simstock = MCdata(simnum,T,mu_min_stock,mu_max_stock,sigma_min_stock,sigma_max_stock)[0]
#mu_min_bond = 0.002
#mu_max_bond = 0.008
#sigma_min_bond = 0.005
#sigma_max_bond = 0.020
random.seed(321)
bsimbond = MCdata(simnum,T,mu_min_bond,mu_max_bond,sigma_min_bond,sigma_max_bond)[0]
#mu_min_cash = 0.001
#mu_max_cash = 0.006
#sigma_min_cash = 0.001
#sigma_max_cash = 0.002
random.seed(321)
bsimcash = MCdata(simnum,T,mu_min_cash,mu_max_cash,sigma_min_cash,sigma_max_cash)[0]

#simhmhs,sigmahmhs = MCdata(simnum,T,0.003,0.005,0.150,0.200)
#simhmms,sigmahmms = MCdata(simnum,T,0.003,0.005,0.100,0.150)
#simhmls,sigmahmls = MCdata(simnum,T,0.003,0.005,0.050,0.100)
#simmmhs,sigmammhs = MCdata(simnum,T,-0.003,0.003,0.150,0.200)
#simmmms,sigmammms = MCdata(simnum,T,-0.003,0.003,0.100,0.150)
#simmmls,sigmammls = MCdata(simnum,T,-0.003,0.003,0.050,0.100)
#simlmhs,sigmalmhs = MCdata(simnum,T,-0.005,-0.003,0.150,0.200)
#simlmms,sigmalmms = MCdata(simnum,T,-0.005,-0.003,0.100,0.150)
#simlmls,sigmalmls = MCdata(simnum,T,-0.005,-0.003,0.050,0.100)
random.seed(321)
simhmhs,sigmahmhs = MCdata(simnum,T,mubp2,mu_max_stock,sigmabp2,sigma_max_stock)
random.seed(321)
simhmms,sigmahmms = MCdata(simnum,T,mubp2,mu_max_stock,sigmabp1,sigmabp2)
random.seed(321)
simhmls,sigmahmls = MCdata(simnum,T,mubp2,mu_max_stock,sigma_min_stock,sigmabp1)
random.seed(321)
simmmhs,sigmammhs = MCdata(simnum,T,mubp1,mubp2,sigmabp2,sigma_max_stock)
random.seed(321)
simmmms,sigmammms = MCdata(simnum,T,mubp1,mubp2,sigmabp1,sigmabp2)
random.seed(321)
simmmls,sigmammls = MCdata(simnum,T,mubp1,mubp2,sigma_min_stock,sigmabp1)
random.seed(321)
simlmhs,sigmalmhs = MCdata(simnum,T,mu_min_stock,mubp2,sigmabp2,sigma_max_stock)
random.seed(321)
simlmms,sigmalmms = MCdata(simnum,T,mu_min_stock,mubp2,sigmabp1,sigmabp2)
random.seed(321)
simlmls,sigmalmls = MCdata(simnum,T,mu_min_stock,mubp2,sigma_min_stock,sigmabp1)


# #### B1 - Put_based OBPI (Different Floors)

# In[24]:


vol_type = "hist"
implement_type = "put_based"
floor_strategy = "dynamic"
floor = 0.99

obpip99_hmhs=np.empty(shape=[0,T],dtype=float)
obpip99_hmms=np.empty(shape=[0,T],dtype=float)
obpip99_hmls=np.empty(shape=[0,T],dtype=float)
obpip99_mmhs=np.empty(shape=[0,T],dtype=float)
obpip99_mmms=np.empty(shape=[0,T],dtype=float)
obpip99_mmls=np.empty(shape=[0,T],dtype=float)
obpip99_lmhs=np.empty(shape=[0,T],dtype=float)
obpip99_lmms=np.empty(shape=[0,T],dtype=float)
obpip99_lmls=np.empty(shape=[0,T],dtype=float)
for i in range(bsimnum):
    obpihmhs = gOBPI(months, s0 * np.cumprod(1+simhmhs[i]), simhmhs[i], bsimcash[i], 0, 10000, np.full((T,1),sigmahmhs[i]), [])
    obpihmhs.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip99_hmhs = np.append(obpip99_hmhs,[obpihmhs.nav], axis=0)
    obpihmms = gOBPI(months, s0 * np.cumprod(1+simhmms[i]), simhmms[i], bsimcash[i], 0, 10000, np.full((T,1),sigmahmms[i]), [])
    obpihmms.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip99_hmms = np.append(obpip99_hmms,[obpihmms.nav], axis=0)
    obpihmls = gOBPI(months, s0 * np.cumprod(1+simhmls[i]), simhmls[i], bsimcash[i], 0, 10000, np.full((T,1),sigmahmls[i]), [])
    obpihmls.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip99_hmls = np.append(obpip99_hmls,[obpihmls.nav], axis=0)
    obpimmhs = gOBPI(months, s0 * np.cumprod(1+simmmhs[i]), simmmhs[i], bsimcash[i], 0, 10000, np.full((T,1),sigmammhs[i]), [])
    obpimmhs.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip99_mmhs = np.append(obpip99_mmhs,[obpimmhs.nav], axis=0)
    obpimmms = gOBPI(months, s0 * np.cumprod(1+simmmms[i]), simmmms[i], bsimcash[i], 0, 10000, np.full((T,1),sigmammms[i]), [])
    obpimmms.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip99_mmms = np.append(obpip99_mmms,[obpimmms.nav], axis=0)
    obpimmls = gOBPI(months, s0 * np.cumprod(1+simmmls[i]), simmmls[i], bsimcash[i], 0, 10000, np.full((T,1),sigmammls[i]), [])
    obpimmls.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip99_mmls = np.append(obpip99_mmls,[obpimmls.nav], axis=0)
    obpilmhs = gOBPI(months, s0 * np.cumprod(1+simlmhs[i]), simlmhs[i], bsimcash[i], 0, 10000, np.full((T,1),sigmalmhs[i]), [])
    obpilmhs.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip99_lmhs = np.append(obpip99_lmhs,[obpilmhs.nav], axis=0)
    obpilmms = gOBPI(months, s0 * np.cumprod(1+simlmms[i]), simlmms[i], bsimcash[i], 0, 10000, np.full((T,1),sigmalmms[i]), [])
    obpilmms.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip99_lmms = np.append(obpip99_lmms,[obpilmms.nav], axis=0)
    obpilmls = gOBPI(months, s0 * np.cumprod(1+simlmls[i]), simlmls[i], bsimcash[i], 0, 10000, np.full((T,1),sigmalmls[i]), [])
    obpilmls.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip99_lmls = np.append(obpip99_lmls,[obpilmls.nav], axis=0)


# In[25]:


obpip99_mc = pd.DataFrame({'Date':np.arange(T),
                           'OBPIp99_hmhs':np.mean(obpip99_hmhs,axis=0),'OBPIp99_hmms':np.mean(obpip99_hmms,axis=0),'OBPIp99_hmls':np.mean(obpip99_hmls,axis=0),
                           'OBPIp99_mmhs':np.mean(obpip99_mmhs,axis=0),'OBPIp99_mmms':np.mean(obpip99_mmms,axis=0),'OBPIp99_mmls':np.mean(obpip99_mmls,axis=0),
                           'OBPIp99_lmhs':np.mean(obpip99_lmhs,axis=0),'OBPIp99_lmms':np.mean(obpip99_lmms,axis=0),'OBPIp99_lmls':np.mean(obpip99_lmls,axis=0)})


# In[26]:


plt.plot(obpip99_mc['Date'],obpip99_mc.iloc[:,1:])
plt.legend(obpip99_mc.columns[1:])
plt.show()


# In[27]:


vol_type = "hist"
implement_type = "put_based"
floor_strategy = "dynamic"
floor = 0.98

obpip98_hmhs=np.empty(shape=[0,T],dtype=float)
obpip98_hmms=np.empty(shape=[0,T],dtype=float)
obpip98_hmls=np.empty(shape=[0,T],dtype=float)
obpip98_mmhs=np.empty(shape=[0,T],dtype=float)
obpip98_mmms=np.empty(shape=[0,T],dtype=float)
obpip98_mmls=np.empty(shape=[0,T],dtype=float)
obpip98_lmhs=np.empty(shape=[0,T],dtype=float)
obpip98_lmms=np.empty(shape=[0,T],dtype=float)
obpip98_lmls=np.empty(shape=[0,T],dtype=float)
for i in range(bsimnum):
    obpihmhs = gOBPI(months, s0 * np.cumprod(1+simhmhs[i]), simhmhs[i], bsimcash[i], 0, 10000, np.full((T,1),sigmahmhs[i]), [])
    obpihmhs.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip98_hmhs = np.append(obpip98_hmhs,[obpihmhs.nav], axis=0)
    obpihmms = gOBPI(months, s0 * np.cumprod(1+simhmms[i]), simhmms[i], bsimcash[i], 0, 10000, np.full((T,1),sigmahmms[i]), [])
    obpihmms.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip98_hmms = np.append(obpip98_hmms,[obpihmms.nav], axis=0)
    obpihmls = gOBPI(months, s0 * np.cumprod(1+simhmls[i]), simhmls[i], bsimcash[i], 0, 10000, np.full((T,1),sigmahmls[i]), [])
    obpihmls.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip98_hmls = np.append(obpip98_hmls,[obpihmls.nav], axis=0)
    obpimmhs = gOBPI(months, s0 * np.cumprod(1+simmmhs[i]), simmmhs[i], bsimcash[i], 0, 10000, np.full((T,1),sigmammhs[i]), [])
    obpimmhs.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip98_mmhs = np.append(obpip98_mmhs,[obpimmhs.nav], axis=0)
    obpimmms = gOBPI(months, s0 * np.cumprod(1+simmmms[i]), simmmms[i], bsimcash[i], 0, 10000, np.full((T,1),sigmammms[i]), [])
    obpimmms.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip98_mmms = np.append(obpip98_mmms,[obpimmms.nav], axis=0)
    obpimmls = gOBPI(months, s0 * np.cumprod(1+simmmls[i]), simmmls[i], bsimcash[i], 0, 10000, np.full((T,1),sigmammls[i]), [])
    obpimmls.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip98_mmls = np.append(obpip98_mmls,[obpimmls.nav], axis=0)
    obpilmhs = gOBPI(months, s0 * np.cumprod(1+simlmhs[i]), simlmhs[i], bsimcash[i], 0, 10000, np.full((T,1),sigmalmhs[i]), [])
    obpilmhs.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip98_lmhs = np.append(obpip98_lmhs,[obpilmhs.nav], axis=0)
    obpilmms = gOBPI(months, s0 * np.cumprod(1+simlmms[i]), simlmms[i], bsimcash[i], 0, 10000, np.full((T,1),sigmalmms[i]), [])
    obpilmms.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip98_lmms = np.append(obpip98_lmms,[obpilmms.nav], axis=0)
    obpilmls = gOBPI(months, s0 * np.cumprod(1+simlmls[i]), simlmls[i], bsimcash[i], 0, 10000, np.full((T,1),sigmalmls[i]), [])
    obpilmls.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip98_lmls = np.append(obpip98_lmls,[obpilmls.nav], axis=0)


# In[28]:


obpip98_mc = pd.DataFrame({'Date':np.arange(T),
                           'OBPIp98_hmhs':np.mean(obpip98_hmhs,axis=0),'OBPIp98_hmms':np.mean(obpip98_hmms,axis=0),'OBPIp98_hmls':np.mean(obpip98_hmls,axis=0),
                           'OBPIp98_mmhs':np.mean(obpip98_mmhs,axis=0),'OBPIp98_mmms':np.mean(obpip98_mmms,axis=0),'OBPIp98_mmls':np.mean(obpip98_mmls,axis=0),
                           'OBPIp98_lmhs':np.mean(obpip98_lmhs,axis=0),'OBPIp98_lmms':np.mean(obpip98_lmms,axis=0),'OBPIp98_lmls':np.mean(obpip98_lmls,axis=0)})


# In[29]:


plt.plot(obpip98_mc['Date'],obpip98_mc.iloc[:,1:])
plt.legend(obpip98_mc.columns[1:])
plt.show()


# In[30]:


vol_type = "hist"
implement_type = "put_based"
floor_strategy = "dynamic"
floor = 0.95

obpip95_hmhs=np.empty(shape=[0,T],dtype=float)
obpip95_hmms=np.empty(shape=[0,T],dtype=float)
obpip95_hmls=np.empty(shape=[0,T],dtype=float)
obpip95_mmhs=np.empty(shape=[0,T],dtype=float)
obpip95_mmms=np.empty(shape=[0,T],dtype=float)
obpip95_mmls=np.empty(shape=[0,T],dtype=float)
obpip95_lmhs=np.empty(shape=[0,T],dtype=float)
obpip95_lmms=np.empty(shape=[0,T],dtype=float)
obpip95_lmls=np.empty(shape=[0,T],dtype=float)
for i in range(bsimnum):
    obpihmhs = gOBPI(months, s0 * np.cumprod(1+simhmhs[i]), simhmhs[i], bsimcash[i], 0, 10000, np.full((T,1),sigmahmhs[i]), [])
    obpihmhs.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip95_hmhs = np.append(obpip95_hmhs,[obpihmhs.nav], axis=0)
    obpihmms = gOBPI(months, s0 * np.cumprod(1+simhmms[i]), simhmms[i], bsimcash[i], 0, 10000, np.full((T,1),sigmahmms[i]), [])
    obpihmms.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip95_hmms = np.append(obpip95_hmms,[obpihmms.nav], axis=0)
    obpihmls = gOBPI(months, s0 * np.cumprod(1+simhmls[i]), simhmls[i], bsimcash[i], 0, 10000, np.full((T,1),sigmahmls[i]), [])
    obpihmls.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip95_hmls = np.append(obpip95_hmls,[obpihmls.nav], axis=0)
    obpimmhs = gOBPI(months, s0 * np.cumprod(1+simmmhs[i]), simmmhs[i], bsimcash[i], 0, 10000, np.full((T,1),sigmammhs[i]), [])
    obpimmhs.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip95_mmhs = np.append(obpip95_mmhs,[obpimmhs.nav], axis=0)
    obpimmms = gOBPI(months, s0 * np.cumprod(1+simmmms[i]), simmmms[i], bsimcash[i], 0, 10000, np.full((T,1),sigmammms[i]), [])
    obpimmms.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip95_mmms = np.append(obpip95_mmms,[obpimmms.nav], axis=0)
    obpimmls = gOBPI(months, s0 * np.cumprod(1+simmmls[i]), simmmls[i], bsimcash[i], 0, 10000, np.full((T,1),sigmammls[i]), [])
    obpimmls.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip95_mmls = np.append(obpip95_mmls,[obpimmls.nav], axis=0)
    obpilmhs = gOBPI(months, s0 * np.cumprod(1+simlmhs[i]), simlmhs[i], bsimcash[i], 0, 10000, np.full((T,1),sigmalmhs[i]), [])
    obpilmhs.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip95_lmhs = np.append(obpip95_lmhs,[obpilmhs.nav], axis=0)
    obpilmms = gOBPI(months, s0 * np.cumprod(1+simlmms[i]), simlmms[i], bsimcash[i], 0, 10000, np.full((T,1),sigmalmms[i]), [])
    obpilmms.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip95_lmms = np.append(obpip95_lmms,[obpilmms.nav], axis=0)
    obpilmls = gOBPI(months, s0 * np.cumprod(1+simlmls[i]), simlmls[i], bsimcash[i], 0, 10000, np.full((T,1),sigmalmls[i]), [])
    obpilmls.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpip95_lmls = np.append(obpip95_lmls,[obpilmls.nav], axis=0)


# In[31]:


obpip95_mc = pd.DataFrame({'Date':np.arange(T),
                           'OBPIp95_hmhs':np.mean(obpip95_hmhs,axis=0),'OBPIp95_hmms':np.mean(obpip95_hmms,axis=0),'OBPIp95_hmls':np.mean(obpip95_hmls,axis=0),
                           'OBPIp95_mmhs':np.mean(obpip95_mmhs,axis=0),'OBPIp95_mmms':np.mean(obpip95_mmms,axis=0),'OBPIp95_mmls':np.mean(obpip95_mmls,axis=0),
                           'OBPIp95_lmhs':np.mean(obpip95_lmhs,axis=0),'OBPIp95_lmms':np.mean(obpip95_lmms,axis=0),'OBPIp95_lmls':np.mean(obpip95_lmls,axis=0)})


# In[32]:


plt.plot(obpip95_mc['Date'],obpip95_mc.iloc[:,1:])
plt.legend(obpip95_mc.columns[1:])
plt.show()


# #### B2 - Call_based OBPI (Different Floors)

# In[33]:


vol_type = "hist"
implement_type = "call_based"
floor_strategy = "dynamic"
floor = 0.99

obpic99_hmhs=np.empty(shape=[0,T],dtype=float)
obpic99_hmms=np.empty(shape=[0,T],dtype=float)
obpic99_hmls=np.empty(shape=[0,T],dtype=float)
obpic99_mmhs=np.empty(shape=[0,T],dtype=float)
obpic99_mmms=np.empty(shape=[0,T],dtype=float)
obpic99_mmls=np.empty(shape=[0,T],dtype=float)
obpic99_lmhs=np.empty(shape=[0,T],dtype=float)
obpic99_lmms=np.empty(shape=[0,T],dtype=float)
obpic99_lmls=np.empty(shape=[0,T],dtype=float)
for i in range(bsimnum):
    obpihmhs = gOBPI(months, s0 * np.cumprod(1+simhmhs[i]), simhmhs[i], bsimcash[i], 0, 10000, np.full((T,1),sigmahmhs[i]), [])
    obpihmhs.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic99_hmhs = np.append(obpic99_hmhs,[obpihmhs.nav], axis=0)
    obpihmms = gOBPI(months, s0 * np.cumprod(1+simhmms[i]), simhmms[i], bsimcash[i], 0, 10000, np.full((T,1),sigmahmms[i]), [])
    obpihmms.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic99_hmms = np.append(obpic99_hmms,[obpihmms.nav], axis=0)
    obpihmls = gOBPI(months, s0 * np.cumprod(1+simhmls[i]), simhmls[i], bsimcash[i], 0, 10000, np.full((T,1),sigmahmls[i]), [])
    obpihmls.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic99_hmls = np.append(obpic99_hmls,[obpihmls.nav], axis=0)
    obpimmhs = gOBPI(months, s0 * np.cumprod(1+simmmhs[i]), simmmhs[i], bsimcash[i], 0, 10000, np.full((T,1),sigmammhs[i]), [])
    obpimmhs.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic99_mmhs = np.append(obpic99_mmhs,[obpimmhs.nav], axis=0)
    obpimmms = gOBPI(months, s0 * np.cumprod(1+simmmms[i]), simmmms[i], bsimcash[i], 0, 10000, np.full((T,1),sigmammms[i]), [])
    obpimmms.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic99_mmms = np.append(obpic99_mmms,[obpimmms.nav], axis=0)
    obpimmls = gOBPI(months, s0 * np.cumprod(1+simmmls[i]), simmmls[i], bsimcash[i], 0, 10000, np.full((T,1),sigmammls[i]), [])
    obpimmls.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic99_mmls = np.append(obpic99_mmls,[obpimmls.nav], axis=0)
    obpilmhs = gOBPI(months, s0 * np.cumprod(1+simlmhs[i]), simlmhs[i], bsimcash[i], 0, 10000, np.full((T,1),sigmalmhs[i]), [])
    obpilmhs.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic99_lmhs = np.append(obpic99_lmhs,[obpilmhs.nav], axis=0)
    obpilmms = gOBPI(months, s0 * np.cumprod(1+simlmms[i]), simlmms[i], bsimcash[i], 0, 10000, np.full((T,1),sigmalmms[i]), [])
    obpilmms.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic99_lmms = np.append(obpic99_lmms,[obpilmms.nav], axis=0)
    obpilmls = gOBPI(months, s0 * np.cumprod(1+simlmls[i]), simlmls[i], bsimcash[i], 0, 10000, np.full((T,1),sigmalmls[i]), [])
    obpilmls.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic99_lmls = np.append(obpic99_lmls,[obpilmls.nav], axis=0)


# In[34]:


obpic99_mc = pd.DataFrame({'Date':np.arange(T),
                           'OBPIc99_hmhs':np.mean(obpic99_hmhs,axis=0),'OBPIc99_hmms':np.mean(obpic99_hmms,axis=0),'OBPIc99_hmls':np.mean(obpic99_hmls,axis=0),
                           'OBPIc99_mmhs':np.mean(obpic99_mmhs,axis=0),'OBPIc99_mmms':np.mean(obpic99_mmms,axis=0),'OBPIc99_mmls':np.mean(obpic99_mmls,axis=0),
                           'OBPIc99_lmhs':np.mean(obpic99_lmhs,axis=0),'OBPIc99_lmms':np.mean(obpic99_lmms,axis=0),'OBPIc99_lmls':np.mean(obpic99_lmls,axis=0)})


# In[35]:


plt.plot(obpic99_mc['Date'],obpic99_mc.iloc[:,1:])
plt.legend(obpic99_mc.columns[1:])
plt.show()


# In[36]:


vol_type = "hist"
implement_type = "call_based"
floor_strategy = "dynamic"
floor = 0.98

obpic98_hmhs=np.empty(shape=[0,T],dtype=float)
obpic98_hmms=np.empty(shape=[0,T],dtype=float)
obpic98_hmls=np.empty(shape=[0,T],dtype=float)
obpic98_mmhs=np.empty(shape=[0,T],dtype=float)
obpic98_mmms=np.empty(shape=[0,T],dtype=float)
obpic98_mmls=np.empty(shape=[0,T],dtype=float)
obpic98_lmhs=np.empty(shape=[0,T],dtype=float)
obpic98_lmms=np.empty(shape=[0,T],dtype=float)
obpic98_lmls=np.empty(shape=[0,T],dtype=float)
for i in range(bsimnum):
    obpihmhs = gOBPI(months, s0 * np.cumprod(1+simhmhs[i]), simhmhs[i], bsimcash[i], 0, 10000, np.full((T,1),sigmahmhs[i]), [])
    obpihmhs.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic98_hmhs = np.append(obpic98_hmhs,[obpihmhs.nav], axis=0)
    obpihmms = gOBPI(months, s0 * np.cumprod(1+simhmms[i]), simhmms[i], bsimcash[i], 0, 10000, np.full((T,1),sigmahmms[i]), [])
    obpihmms.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic98_hmms = np.append(obpic98_hmms,[obpihmms.nav], axis=0)
    obpihmls = gOBPI(months, s0 * np.cumprod(1+simhmls[i]), simhmls[i], bsimcash[i], 0, 10000, np.full((T,1),sigmahmls[i]), [])
    obpihmls.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic98_hmls = np.append(obpic98_hmls,[obpihmls.nav], axis=0)
    obpimmhs = gOBPI(months, s0 * np.cumprod(1+simmmhs[i]), simmmhs[i], bsimcash[i], 0, 10000, np.full((T,1),sigmammhs[i]), [])
    obpimmhs.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic98_mmhs = np.append(obpic98_mmhs,[obpimmhs.nav], axis=0)
    obpimmms = gOBPI(months, s0 * np.cumprod(1+simmmms[i]), simmmms[i], bsimcash[i], 0, 10000, np.full((T,1),sigmammms[i]), [])
    obpimmms.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic98_mmms = np.append(obpic98_mmms,[obpimmms.nav], axis=0)
    obpimmls = gOBPI(months, s0 * np.cumprod(1+simmmls[i]), simmmls[i], bsimcash[i], 0, 10000, np.full((T,1),sigmammls[i]), [])
    obpimmls.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic98_mmls = np.append(obpic98_mmls,[obpimmls.nav], axis=0)
    obpilmhs = gOBPI(months, s0 * np.cumprod(1+simlmhs[i]), simlmhs[i], bsimcash[i], 0, 10000, np.full((T,1),sigmalmhs[i]), [])
    obpilmhs.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic98_lmhs = np.append(obpic98_lmhs,[obpilmhs.nav], axis=0)
    obpilmms = gOBPI(months, s0 * np.cumprod(1+simlmms[i]), simlmms[i], bsimcash[i], 0, 10000, np.full((T,1),sigmalmms[i]), [])
    obpilmms.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic98_lmms = np.append(obpic98_lmms,[obpilmms.nav], axis=0)
    obpilmls = gOBPI(months, s0 * np.cumprod(1+simlmls[i]), simlmls[i], bsimcash[i], 0, 10000, np.full((T,1),sigmalmls[i]), [])
    obpilmls.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic98_lmls = np.append(obpic98_lmls,[obpilmls.nav], axis=0)


# In[37]:


obpic98_mc = pd.DataFrame({'Date':np.arange(T),
                           'OBPIc98_hmhs':np.mean(obpic98_hmhs,axis=0),'OBPIc98_hmms':np.mean(obpic98_hmms,axis=0),'OBPIc98_hmls':np.mean(obpic98_hmls,axis=0),
                           'OBPIc98_mmhs':np.mean(obpic98_mmhs,axis=0),'OBPIc98_mmms':np.mean(obpic98_mmms,axis=0),'OBPIc98_mmls':np.mean(obpic98_mmls,axis=0),
                           'OBPIc98_lmhs':np.mean(obpic98_lmhs,axis=0),'OBPIc98_lmms':np.mean(obpic98_lmms,axis=0),'OBPIc98_lmls':np.mean(obpic98_lmls,axis=0)})


# In[38]:


plt.plot(obpic98_mc['Date'],obpic98_mc.iloc[:,1:])
plt.legend(obpic98_mc.columns[1:])
plt.show()


# In[39]:


vol_type = "hist"
implement_type = "call_based"
floor_strategy = "dynamic"
floor = 0.95

obpic95_hmhs=np.empty(shape=[0,T],dtype=float)
obpic95_hmms=np.empty(shape=[0,T],dtype=float)
obpic95_hmls=np.empty(shape=[0,T],dtype=float)
obpic95_mmhs=np.empty(shape=[0,T],dtype=float)
obpic95_mmms=np.empty(shape=[0,T],dtype=float)
obpic95_mmls=np.empty(shape=[0,T],dtype=float)
obpic95_lmhs=np.empty(shape=[0,T],dtype=float)
obpic95_lmms=np.empty(shape=[0,T],dtype=float)
obpic95_lmls=np.empty(shape=[0,T],dtype=float)
for i in range(bsimnum):
    obpihmhs = gOBPI(months, s0 * np.cumprod(1+simhmhs[i]), simhmhs[i], bsimcash[i], 0, 10000, np.full((T,1),sigmahmhs[i]), [])
    obpihmhs.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic95_hmhs = np.append(obpic95_hmhs,[obpihmhs.nav], axis=0)
    obpihmms = gOBPI(months, s0 * np.cumprod(1+simhmms[i]), simhmms[i], bsimcash[i], 0, 10000, np.full((T,1),sigmahmms[i]), [])
    obpihmms.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic95_hmms = np.append(obpic95_hmms,[obpihmms.nav], axis=0)
    obpihmls = gOBPI(months, s0 * np.cumprod(1+simhmls[i]), simhmls[i], bsimcash[i], 0, 10000, np.full((T,1),sigmahmls[i]), [])
    obpihmls.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic95_hmls = np.append(obpic95_hmls,[obpihmls.nav], axis=0)
    obpimmhs = gOBPI(months, s0 * np.cumprod(1+simmmhs[i]), simmmhs[i], bsimcash[i], 0, 10000, np.full((T,1),sigmammhs[i]), [])
    obpimmhs.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic95_mmhs = np.append(obpic95_mmhs,[obpimmhs.nav], axis=0)
    obpimmms = gOBPI(months, s0 * np.cumprod(1+simmmms[i]), simmmms[i], bsimcash[i], 0, 10000, np.full((T,1),sigmammms[i]), [])
    obpimmms.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic95_mmms = np.append(obpic95_mmms,[obpimmms.nav], axis=0)
    obpimmls = gOBPI(months, s0 * np.cumprod(1+simmmls[i]), simmmls[i], bsimcash[i], 0, 10000, np.full((T,1),sigmammls[i]), [])
    obpimmls.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic95_mmls = np.append(obpic95_mmls,[obpimmls.nav], axis=0)
    obpilmhs = gOBPI(months, s0 * np.cumprod(1+simlmhs[i]), simlmhs[i], bsimcash[i], 0, 10000, np.full((T,1),sigmalmhs[i]), [])
    obpilmhs.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic95_lmhs = np.append(obpic95_lmhs,[obpilmhs.nav], axis=0)
    obpilmms = gOBPI(months, s0 * np.cumprod(1+simlmms[i]), simlmms[i], bsimcash[i], 0, 10000, np.full((T,1),sigmalmms[i]), [])
    obpilmms.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic95_lmms = np.append(obpic95_lmms,[obpilmms.nav], axis=0)
    obpilmls = gOBPI(months, s0 * np.cumprod(1+simlmls[i]), simlmls[i], bsimcash[i], 0, 10000, np.full((T,1),sigmalmls[i]), [])
    obpilmls.run(vol_type=vol_type, implement_type = implement_type, floor_strategy = floor_strategy, floor = floor)
    obpic95_lmls = np.append(obpic95_lmls,[obpilmls.nav], axis=0)


# In[40]:


obpic95_mc = pd.DataFrame({'Date':np.arange(T),
                           'OBPIc95_hmhs':np.mean(obpic95_hmhs,axis=0),'OBPIc95_hmms':np.mean(obpic95_hmms,axis=0),'OBPIc95_hmls':np.mean(obpic95_hmls,axis=0),
                           'OBPIc95_mmhs':np.mean(obpic95_mmhs,axis=0),'OBPIc95_mmms':np.mean(obpic95_mmms,axis=0),'OBPIc95_mmls':np.mean(obpic95_mmls,axis=0),
                           'OBPIc95_lmhs':np.mean(obpic95_lmhs,axis=0),'OBPIc95_lmms':np.mean(obpic95_lmms,axis=0),'OBPIc95_lmls':np.mean(obpic95_lmls,axis=0)})


# In[41]:


plt.plot(obpic95_mc['Date'],obpic95_mc.iloc[:,1:])
plt.legend(obpic95_mc.columns[1:])
plt.show()


# #### C0 - NAV Features

# In[42]:


fig = plt.figure(figsize = (8,8)) 
plt.subplots_adjust(wspace=0.4, hspace=0.4)

ax1 = fig.add_subplot(331)
plt.subplot(331)
plt.scatter(np.std(cppi_hmhs[:,-1]),np.mean(cppi_hmhs[:,-1]),marker = '<')
plt.scatter(np.std(dcppi_hmhs[:,-1]),np.mean(dcppi_hmhs[:,-1]),marker = '+')
plt.scatter(np.std(d2cppi_hmhs[:,-1]),np.mean(d2cppi_hmhs[:,-1]),marker = '*')
plt.scatter(np.std(d2vcppi_hmhs[:,-1]),np.mean(d2vcppi_hmhs[:,-1]),marker = 'o')
plt.scatter(np.std(obpip95_hmhs[:,-1]),np.mean(obpip95_hmhs[:,-1]),marker = 'd')
plt.scatter(np.std(obpip98_hmhs[:,-1]),np.mean(obpip98_hmhs[:,-1]),marker = 'D')
plt.scatter(np.std(obpip99_hmhs[:,-1]),np.mean(obpip99_hmhs[:,-1]),marker = 'x')
plt.scatter(np.std(obpic95_hmhs[:,-1]),np.mean(obpic95_hmhs[:,-1]),marker = '_')
plt.scatter(np.std(obpic98_hmhs[:,-1]),np.mean(obpic98_hmhs[:,-1]),marker = '|')
plt.scatter(np.std(obpic99_hmhs[:,-1]),np.mean(obpic99_hmhs[:,-1]),marker = 'X')

ax2 = fig.add_subplot(332)
plt.subplot(332)
plt.scatter(np.std(cppi_hmms[:,-1]),np.mean(cppi_hmms[:,-1]),marker = '<')
plt.scatter(np.std(dcppi_hmms[:,-1]),np.mean(dcppi_hmms[:,-1]),marker = '+')
plt.scatter(np.std(d2cppi_hmms[:,-1]),np.mean(d2cppi_hmms[:,-1]),marker = '*')
plt.scatter(np.std(d2vcppi_hmms[:,-1]),np.mean(d2vcppi_hmms[:,-1]),marker = 'o')
plt.scatter(np.std(obpip95_hmms[:,-1]),np.mean(obpip95_hmms[:,-1]),marker = 'd')
plt.scatter(np.std(obpip98_hmms[:,-1]),np.mean(obpip98_hmms[:,-1]),marker = 'D')
plt.scatter(np.std(obpip99_hmms[:,-1]),np.mean(obpip99_hmms[:,-1]),marker = 'x')
plt.scatter(np.std(obpic95_hmms[:,-1]),np.mean(obpic95_hmms[:,-1]),marker = '_')
plt.scatter(np.std(obpic98_hmms[:,-1]),np.mean(obpic98_hmms[:,-1]),marker = '|')
plt.scatter(np.std(obpic99_hmms[:,-1]),np.mean(obpic99_hmms[:,-1]),marker = 'X')

ax3 = fig.add_subplot(333)
plt.subplot(333)
plt.scatter(np.std(cppi_hmls[:,-1]),np.mean(cppi_hmls[:,-1]),marker = '<')
plt.scatter(np.std(dcppi_hmls[:,-1]),np.mean(dcppi_hmls[:,-1]),marker = '+')
plt.scatter(np.std(d2cppi_hmls[:,-1]),np.mean(d2cppi_hmls[:,-1]),marker = '*')
plt.scatter(np.std(d2vcppi_hmls[:,-1]),np.mean(d2vcppi_hmls[:,-1]),marker = 'o')
plt.scatter(np.std(obpip95_hmls[:,-1]),np.mean(obpip95_hmls[:,-1]),marker = 'd')
plt.scatter(np.std(obpip98_hmls[:,-1]),np.mean(obpip98_hmls[:,-1]),marker = 'D')
plt.scatter(np.std(obpip99_hmls[:,-1]),np.mean(obpip99_hmls[:,-1]),marker = 'x')
plt.scatter(np.std(obpic95_hmls[:,-1]),np.mean(obpic95_hmls[:,-1]),marker = '_')
plt.scatter(np.std(obpic98_hmls[:,-1]),np.mean(obpic98_hmls[:,-1]),marker = '|')
plt.scatter(np.std(obpic99_hmls[:,-1]),np.mean(obpic99_hmls[:,-1]),marker = 'X')

ax4 = fig.add_subplot(334)
plt.subplot(334)
plt.scatter(np.std(cppi_mmhs[:,-1]),np.mean(cppi_mmhs[:,-1]),marker = '<')
plt.scatter(np.std(dcppi_mmhs[:,-1]),np.mean(dcppi_mmhs[:,-1]),marker = '+')
plt.scatter(np.std(d2cppi_mmhs[:,-1]),np.mean(d2cppi_mmhs[:,-1]),marker = '*')
plt.scatter(np.std(d2vcppi_mmhs[:,-1]),np.mean(d2vcppi_mmhs[:,-1]),marker = 'o')
plt.scatter(np.std(obpip95_mmhs[:,-1]),np.mean(obpip95_mmhs[:,-1]),marker = 'd')
plt.scatter(np.std(obpip98_mmhs[:,-1]),np.mean(obpip98_mmhs[:,-1]),marker = 'D')
plt.scatter(np.std(obpip99_mmhs[:,-1]),np.mean(obpip99_mmhs[:,-1]),marker = 'x')
plt.scatter(np.std(obpic95_mmhs[:,-1]),np.mean(obpic95_mmhs[:,-1]),marker = '_')
plt.scatter(np.std(obpic98_mmhs[:,-1]),np.mean(obpic98_mmhs[:,-1]),marker = '|')
plt.scatter(np.std(obpic99_mmhs[:,-1]),np.mean(obpic99_mmhs[:,-1]),marker = 'X')

ax5 = fig.add_subplot(335)
plt.subplot(335)
plt.scatter(np.std(cppi_mmms[:,-1]),np.mean(cppi_mmms[:,-1]),marker = '<')
plt.scatter(np.std(dcppi_mmms[:,-1]),np.mean(dcppi_mmms[:,-1]),marker = '+')
plt.scatter(np.std(d2cppi_mmms[:,-1]),np.mean(d2cppi_mmms[:,-1]),marker = '*')
plt.scatter(np.std(d2vcppi_mmms[:,-1]),np.mean(d2vcppi_mmms[:,-1]),marker = 'o')
plt.scatter(np.std(obpip95_mmms[:,-1]),np.mean(obpip95_mmms[:,-1]),marker = 'd')
plt.scatter(np.std(obpip98_mmms[:,-1]),np.mean(obpip98_mmms[:,-1]),marker = 'D')
plt.scatter(np.std(obpip99_mmms[:,-1]),np.mean(obpip99_mmms[:,-1]),marker = 'x')
plt.scatter(np.std(obpic95_mmms[:,-1]),np.mean(obpic95_mmms[:,-1]),marker = '_')
plt.scatter(np.std(obpic98_mmms[:,-1]),np.mean(obpic98_mmms[:,-1]),marker = '|')
plt.scatter(np.std(obpic99_mmms[:,-1]),np.mean(obpic99_mmms[:,-1]),marker = 'X')


ax6 = fig.add_subplot(336)
plt.subplot(336)
plt.scatter(np.std(cppi_mmls[:,-1]),np.mean(cppi_mmls[:,-1]),marker = '<')
plt.scatter(np.std(dcppi_mmls[:,-1]),np.mean(dcppi_mmls[:,-1]),marker = '+')
plt.scatter(np.std(d2cppi_mmls[:,-1]),np.mean(d2cppi_mmls[:,-1]),marker = '*')
plt.scatter(np.std(d2vcppi_mmls[:,-1]),np.mean(d2vcppi_mmls[:,-1]),marker = 'o')
plt.scatter(np.std(obpip95_mmls[:,-1]),np.mean(obpip95_mmls[:,-1]),marker = 'd')
plt.scatter(np.std(obpip98_mmls[:,-1]),np.mean(obpip98_mmls[:,-1]),marker = 'D')
plt.scatter(np.std(obpip99_mmls[:,-1]),np.mean(obpip99_mmls[:,-1]),marker = 'x')
plt.scatter(np.std(obpic95_mmls[:,-1]),np.mean(obpic95_mmls[:,-1]),marker = '_')
plt.scatter(np.std(obpic98_mmls[:,-1]),np.mean(obpic98_mmls[:,-1]),marker = '|')
plt.scatter(np.std(obpic99_mmls[:,-1]),np.mean(obpic99_mmls[:,-1]),marker = 'X')

ax7 = fig.add_subplot(337)
plt.subplot(337)
plt.scatter(np.std(cppi_lmhs[:,-1]),np.mean(cppi_lmhs[:,-1]),marker = '<')
plt.scatter(np.std(dcppi_lmhs[:,-1]),np.mean(dcppi_lmhs[:,-1]),marker = '+')
plt.scatter(np.std(d2cppi_lmhs[:,-1]),np.mean(d2cppi_lmhs[:,-1]),marker = '*')
plt.scatter(np.std(d2vcppi_lmhs[:,-1]),np.mean(d2vcppi_lmhs[:,-1]),marker = 'o')
plt.scatter(np.std(obpip95_lmhs[:,-1]),np.mean(obpip95_lmhs[:,-1]),marker = 'd')
plt.scatter(np.std(obpip98_lmhs[:,-1]),np.mean(obpip98_lmhs[:,-1]),marker = 'D')
plt.scatter(np.std(obpip99_lmhs[:,-1]),np.mean(obpip99_lmhs[:,-1]),marker = 'x')
plt.scatter(np.std(obpic95_lmhs[:,-1]),np.mean(obpic95_lmhs[:,-1]),marker = '_')
plt.scatter(np.std(obpic98_lmhs[:,-1]),np.mean(obpic98_lmhs[:,-1]),marker = '|')
plt.scatter(np.std(obpic99_lmhs[:,-1]),np.mean(obpic99_lmhs[:,-1]),marker = 'X')

ax8 = fig.add_subplot(338)
plt.subplot(338)
plt.scatter(np.std(cppi_lmms[:,-1]),np.mean(cppi_lmms[:,-1]),marker = '<')
plt.scatter(np.std(dcppi_lmms[:,-1]),np.mean(dcppi_lmms[:,-1]),marker = '+')
plt.scatter(np.std(d2cppi_lmms[:,-1]),np.mean(d2cppi_lmms[:,-1]),marker = '*')
plt.scatter(np.std(d2vcppi_lmms[:,-1]),np.mean(d2vcppi_lmms[:,-1]),marker = 'o')
plt.scatter(np.std(obpip95_lmms[:,-1]),np.mean(obpip95_lmms[:,-1]),marker = 'd')
plt.scatter(np.std(obpip98_lmms[:,-1]),np.mean(obpip98_lmms[:,-1]),marker = 'D')
plt.scatter(np.std(obpip99_lmms[:,-1]),np.mean(obpip99_lmms[:,-1]),marker = 'x')
plt.scatter(np.std(obpic95_lmms[:,-1]),np.mean(obpic95_lmms[:,-1]),marker = '_')
plt.scatter(np.std(obpic98_lmms[:,-1]),np.mean(obpic98_lmms[:,-1]),marker = '|')
plt.scatter(np.std(obpic99_lmms[:,-1]),np.mean(obpic99_lmms[:,-1]),marker = 'X')

ax9 = fig.add_subplot(339)
plt.subplot(339)
plt.scatter(np.std(cppi_lmls[:,-1]),np.mean(cppi_lmls[:,-1]),marker = '<')
plt.scatter(np.std(dcppi_lmls[:,-1]),np.mean(dcppi_lmls[:,-1]),marker = '+')
plt.scatter(np.std(d2cppi_lmls[:,-1]),np.mean(d2cppi_lmls[:,-1]),marker = '*')
plt.scatter(np.std(d2vcppi_lmls[:,-1]),np.mean(d2vcppi_lmls[:,-1]),marker = 'o')
plt.scatter(np.std(obpip95_lmls[:,-1]),np.mean(obpip95_lmls[:,-1]),marker = 'd')
plt.scatter(np.std(obpip98_lmls[:,-1]),np.mean(obpip98_lmls[:,-1]),marker = 'D')
plt.scatter(np.std(obpip99_lmls[:,-1]),np.mean(obpip99_lmls[:,-1]),marker = 'x')
plt.scatter(np.std(obpic95_lmls[:,-1]),np.mean(obpic95_lmls[:,-1]),marker = '_')
plt.scatter(np.std(obpic98_lmls[:,-1]),np.mean(obpic98_lmls[:,-1]),marker = '|')
plt.scatter(np.std(obpic99_lmls[:,-1]),np.mean(obpic99_lmls[:,-1]),marker = 'X')

plt.legend(['cppi','dcppi','d2cppi','d2vcppi','obpip99','obpip98','obpip95','obpic99','obpip98','obpip95'],loc=4)
ax1.set_title('hmhs',fontsize=10)
ax2.set_title('hmms',fontsize=10)
ax3.set_title('hmls',fontsize=10)
ax4.set_title('mmhs',fontsize=10)
ax5.set_title('mmms',fontsize=10)
ax6.set_title('mmls',fontsize=10)
ax7.set_title('lmhs',fontsize=10)
ax8.set_title('lmms',fontsize=10)
ax9.set_title('lmls',fontsize=10)
ax1.set_xlabel('$\sigma$')
ax1.set_ylabel('$\mu$')
ax2.set_xlabel('$\sigma$')
ax2.set_ylabel('$\mu$')
ax3.set_xlabel('$\sigma$')
ax3.set_ylabel('$\mu$')
ax4.set_xlabel('$\sigma$')
ax4.set_ylabel('$\mu$')
ax5.set_xlabel('$\sigma$')
ax5.set_ylabel('$\mu$')
ax6.set_xlabel('$\sigma$')
ax6.set_ylabel('$\mu$')
ax7.set_xlabel('$\sigma$')
ax7.set_ylabel('$\mu$')
ax8.set_xlabel('$\sigma$')
ax8.set_ylabel('$\mu$')
ax9.set_xlabel('$\sigma$')
ax9.set_ylabel('$\mu$')

#plt.xlabel('$\sigma$ of NAV')  
#plt.ylabel('Mean NAV')

plt.show()


# #### D0 - SR Plots

# In[44]:


sr_cppi_hmhs = []
sr_dcppi_hmhs = []
sr_d2cppi_hmhs = []
sr_d2vcppi_hmhs = []
sr_obpip99_hmhs = []
sr_obpip98_hmhs = []
sr_obpip95_hmhs = []
sr_obpic99_hmhs = []
sr_obpic98_hmhs = []
sr_obpic95_hmhs = []
sr_cppi_hmms = []
sr_dcppi_hmms = []
sr_d2cppi_hmms = []
sr_d2vcppi_hmms = []
sr_obpip99_hmms = []
sr_obpip98_hmms = []
sr_obpip95_hmms = []
sr_obpic99_hmms = []
sr_obpic98_hmms = []
sr_obpic95_hmms = []
sr_cppi_hmls = []
sr_dcppi_hmls = []
sr_d2cppi_hmls = []
sr_d2vcppi_hmls = []
sr_obpip99_hmls = []
sr_obpip98_hmls = []
sr_obpip95_hmls = []
sr_obpic99_hmls = []
sr_obpic98_hmls = []
sr_obpic95_hmls = []
sr_cppi_mmhs = []
sr_dcppi_mmhs = []
sr_d2cppi_mmhs = []
sr_d2vcppi_mmhs = []
sr_obpip99_mmhs = []
sr_obpip98_mmhs = []
sr_obpip95_mmhs = []
sr_obpic99_mmhs = []
sr_obpic98_mmhs = []
sr_obpic95_mmhs = []
sr_cppi_mmms = []
sr_dcppi_mmms = []
sr_d2cppi_mmms = []
sr_d2vcppi_mmms = []
sr_obpip99_mmms = []
sr_obpip98_mmms = []
sr_obpip95_mmms = []
sr_obpic99_mmms = []
sr_obpic98_mmms = []
sr_obpic95_mmms = []
sr_cppi_mmls = []
sr_dcppi_mmls = []
sr_d2cppi_mmls = []
sr_d2vcppi_mmls = []
sr_obpip99_mmls = []
sr_obpip98_mmls = []
sr_obpip95_mmls = []
sr_obpic99_mmls = []
sr_obpic98_mmls = []
sr_obpic95_mmls = []
sr_cppi_lmhs = []
sr_dcppi_lmhs = []
sr_d2cppi_lmhs = []
sr_d2vcppi_lmhs = []
sr_obpip99_lmhs = []
sr_obpip98_lmhs = []
sr_obpip95_lmhs = []
sr_obpic99_lmhs = []
sr_obpic98_lmhs = []
sr_obpic95_lmhs = []
sr_cppi_lmms = []
sr_dcppi_lmms = []
sr_d2cppi_lmms = []
sr_d2vcppi_lmms = []
sr_obpip99_lmms = []
sr_obpip98_lmms = []
sr_obpip95_lmms = []
sr_obpic99_lmms = []
sr_obpic98_lmms = []
sr_obpic95_lmms = []
sr_cppi_lmls = []
sr_dcppi_lmls = []
sr_d2cppi_lmls = []
sr_d2vcppi_lmls = []
sr_obpip99_lmls = []
sr_obpip98_lmls = []
sr_obpip95_lmls = []
sr_obpic99_lmls = []
sr_obpic98_lmls = []
sr_obpic95_lmls = []

for i in range(simnum):
    sr_cppi_hmhs.append(SR(cppi_hmhs[i],simcash[i]))
    sr_dcppi_hmhs.append(SR(cppi_hmhs[i],simcash[i]))
    sr_d2cppi_hmhs.append(SR(cppi_hmhs[i],simcash[i]))
    sr_d2vcppi_hmhs.append(SR(cppi_hmhs[i],simcash[i]))
    sr_cppi_hmms.append(SR(cppi_hmms[i],simcash[i]))
    sr_dcppi_hmms.append(SR(cppi_hmms[i],simcash[i]))
    sr_d2cppi_hmms.append(SR(cppi_hmms[i],simcash[i]))
    sr_d2vcppi_hmms.append(SR(cppi_hmms[i],simcash[i]))
    sr_cppi_hmls.append(SR(cppi_hmls[i],simcash[i]))
    sr_dcppi_hmls.append(SR(cppi_hmls[i],simcash[i]))
    sr_d2cppi_hmls.append(SR(cppi_hmls[i],simcash[i]))
    sr_d2vcppi_hmls.append(SR(cppi_hmls[i],simcash[i]))    
    sr_cppi_mmhs.append(SR(cppi_mmhs[i],simcash[i]))
    sr_dcppi_mmhs.append(SR(cppi_mmhs[i],simcash[i]))
    sr_d2cppi_mmhs.append(SR(cppi_mmhs[i],simcash[i]))
    sr_d2vcppi_mmhs.append(SR(cppi_mmhs[i],simcash[i]))
    sr_cppi_mmms.append(SR(cppi_mmms[i],simcash[i]))
    sr_dcppi_mmms.append(SR(cppi_mmms[i],simcash[i]))
    sr_d2cppi_mmms.append(SR(cppi_mmms[i],simcash[i]))
    sr_d2vcppi_mmms.append(SR(cppi_mmms[i],simcash[i]))
    sr_cppi_mmls.append(SR(cppi_mmls[i],simcash[i]))
    sr_dcppi_mmls.append(SR(cppi_mmls[i],simcash[i]))
    sr_d2cppi_mmls.append(SR(cppi_mmls[i],simcash[i]))
    sr_d2vcppi_mmls.append(SR(cppi_mmls[i],simcash[i]))
    sr_cppi_lmhs.append(SR(cppi_lmhs[i],simcash[i]))
    sr_dcppi_lmhs.append(SR(cppi_lmhs[i],simcash[i]))
    sr_d2cppi_lmhs.append(SR(cppi_lmhs[i],simcash[i]))
    sr_d2vcppi_lmhs.append(SR(cppi_lmhs[i],simcash[i]))  
    sr_cppi_lmms.append(SR(cppi_lmms[i],simcash[i]))
    sr_dcppi_lmms.append(SR(cppi_lmms[i],simcash[i]))
    sr_d2cppi_lmms.append(SR(cppi_lmms[i],simcash[i]))
    sr_d2vcppi_lmms.append(SR(cppi_lmms[i],simcash[i]))
    sr_cppi_lmls.append(SR(cppi_lmls[i],simcash[i]))
    sr_dcppi_lmls.append(SR(cppi_lmls[i],simcash[i]))
    sr_d2cppi_lmls.append(SR(cppi_lmls[i],simcash[i]))
    sr_d2vcppi_lmls.append(SR(cppi_lmls[i],simcash[i]))
    

sr_cppi = pd.DataFrame({("cppi","hmhs"):sr_cppi_hmhs,("dcppi","hmhs"):sr_dcppi_hmhs,("d2cppi","hmhs"):sr_d2cppi_hmhs,("d2vcppi","hmhs"):sr_d2cppi_hmhs,
                        ("cppi","hmms"):sr_cppi_hmms,("dcppi","hmms"):sr_dcppi_hmms,("d2cppi","hmms"):sr_d2cppi_hmms,("d2vcppi","hmms"):sr_d2cppi_hmms,
                        ("cppi","hmls"):sr_cppi_hmls,("dcppi","hmls"):sr_dcppi_hmls,("d2cppi","hmls"):sr_d2cppi_hmls,("d2vcppi","hmls"):sr_d2cppi_hmls,
                        ("cppi","mmhs"):sr_cppi_mmhs,("dcppi","mmhs"):sr_dcppi_mmhs,("d2cppi","mmhs"):sr_d2cppi_mmhs,("d2vcppi","mmhs"):sr_d2cppi_mmhs,
                        ("cppi","mmms"):sr_cppi_mmms,("dcppi","mmms"):sr_dcppi_mmms,("d2cppi","mmms"):sr_d2cppi_mmms,("d2vcppi","mmms"):sr_d2cppi_mmms,
                        ("cppi","mmls"):sr_cppi_mmls,("dcppi","mmls"):sr_dcppi_mmls,("d2cppi","mmls"):sr_d2cppi_mmls,("d2vcppi","mmls"):sr_d2cppi_mmls,
                        ("cppi","lmhs"):sr_cppi_lmhs,("dcppi","lmhs"):sr_dcppi_lmhs,("d2cppi","lmhs"):sr_d2cppi_lmhs,("d2vcppi","lmhs"):sr_d2cppi_lmhs,
                        ("cppi","lmms"):sr_cppi_lmms,("dcppi","lmms"):sr_dcppi_lmms,("d2cppi","lmms"):sr_d2cppi_lmms,("d2vcppi","lmms"):sr_d2cppi_lmms,
                        ("cppi","lmls"):sr_cppi_lmls,("dcppi","lmls"):sr_dcppi_lmls,("d2cppi","lmls"):sr_d2cppi_lmls,("d2vcppi","lmls"):sr_d2cppi_lmls})


# In[45]:


sr_cppi['simid'] = sr_cppi.index
sr_cppi1 = sr_cppi.melt(id_vars=["simid"])
sr_cppi1.columns = ['simid','strategy','MCtype','SR']
fig = plt.figure(figsize = (10,5)) 
ax = sns.boxplot(x="MCtype", y="SR", hue="strategy",
                 data=sr_cppi1, palette="Set3")


# In[46]:


for i in range(bsimnum):
    sr_obpip99_hmhs.append(SR(obpip99_hmhs[i],bsimcash[i]))
    sr_obpip98_hmhs.append(SR(obpip98_hmhs[i],bsimcash[i]))
    sr_obpip95_hmhs.append(SR(obpip95_hmhs[i],bsimcash[i]))
    sr_obpic99_hmhs.append(SR(obpic99_hmhs[i],bsimcash[i]))
    sr_obpic98_hmhs.append(SR(obpic98_hmhs[i],bsimcash[i]))
    sr_obpic95_hmhs.append(SR(obpic95_hmhs[i],bsimcash[i]))
    sr_obpip99_hmms.append(SR(obpip99_hmms[i],bsimcash[i]))
    sr_obpip98_hmms.append(SR(obpip98_hmms[i],bsimcash[i]))
    sr_obpip95_hmms.append(SR(obpip95_hmms[i],bsimcash[i]))
    sr_obpic99_hmms.append(SR(obpic99_hmms[i],bsimcash[i]))
    sr_obpic98_hmms.append(SR(obpic98_hmms[i],bsimcash[i]))
    sr_obpic95_hmms.append(SR(obpic95_hmms[i],bsimcash[i]))
    sr_obpip99_hmls.append(SR(obpip99_hmls[i],bsimcash[i]))
    sr_obpip98_hmls.append(SR(obpip98_hmls[i],bsimcash[i]))
    sr_obpip95_hmls.append(SR(obpip95_hmls[i],bsimcash[i]))
    sr_obpic99_hmls.append(SR(obpic99_hmls[i],bsimcash[i]))
    sr_obpic98_hmls.append(SR(obpic98_hmls[i],bsimcash[i]))
    sr_obpic95_hmls.append(SR(obpic95_hmls[i],bsimcash[i]))
    sr_obpip99_mmhs.append(SR(obpip99_mmhs[i],bsimcash[i]))
    sr_obpip98_mmhs.append(SR(obpip98_mmhs[i],bsimcash[i]))
    sr_obpip95_mmhs.append(SR(obpip95_mmhs[i],bsimcash[i]))
    sr_obpic99_mmhs.append(SR(obpic99_mmhs[i],bsimcash[i]))
    sr_obpic98_mmhs.append(SR(obpic98_mmhs[i],bsimcash[i]))
    sr_obpic95_mmhs.append(SR(obpic95_mmhs[i],bsimcash[i]))
    sr_obpip99_mmms.append(SR(obpip99_mmms[i],bsimcash[i]))
    sr_obpip98_mmms.append(SR(obpip98_mmms[i],bsimcash[i]))
    sr_obpip95_mmms.append(SR(obpip95_mmms[i],bsimcash[i]))
    sr_obpic99_mmms.append(SR(obpic99_mmms[i],bsimcash[i]))
    sr_obpic98_mmms.append(SR(obpic98_mmms[i],bsimcash[i]))
    sr_obpic95_mmms.append(SR(obpic95_mmms[i],bsimcash[i]))
    sr_obpip99_mmls.append(SR(obpip99_mmls[i],bsimcash[i]))
    sr_obpip98_mmls.append(SR(obpip98_mmls[i],bsimcash[i]))
    sr_obpip95_mmls.append(SR(obpip95_mmls[i],bsimcash[i]))
    sr_obpic99_mmls.append(SR(obpic99_mmls[i],bsimcash[i]))
    sr_obpic98_mmls.append(SR(obpic98_mmls[i],bsimcash[i]))
    sr_obpic95_mmls.append(SR(obpic95_mmls[i],bsimcash[i]))
    sr_obpip99_lmhs.append(SR(obpip99_lmhs[i],bsimcash[i]))
    sr_obpip98_lmhs.append(SR(obpip98_lmhs[i],bsimcash[i]))
    sr_obpip95_lmhs.append(SR(obpip95_lmhs[i],bsimcash[i]))
    sr_obpic99_lmhs.append(SR(obpic99_lmhs[i],bsimcash[i]))
    sr_obpic98_lmhs.append(SR(obpic98_lmhs[i],bsimcash[i]))
    sr_obpic95_lmhs.append(SR(obpic95_lmhs[i],bsimcash[i]))
    sr_obpip99_lmms.append(SR(obpip99_lmms[i],bsimcash[i]))
    sr_obpip98_lmms.append(SR(obpip98_lmms[i],bsimcash[i]))
    sr_obpip95_lmms.append(SR(obpip95_lmms[i],bsimcash[i]))
    sr_obpic99_lmms.append(SR(obpic99_lmms[i],bsimcash[i]))
    sr_obpic98_lmms.append(SR(obpic98_lmms[i],bsimcash[i]))
    sr_obpic95_lmms.append(SR(obpic95_lmms[i],bsimcash[i]))
    sr_obpip99_lmls.append(SR(obpip99_lmls[i],bsimcash[i]))
    sr_obpip98_lmls.append(SR(obpip98_lmls[i],bsimcash[i]))
    sr_obpip95_lmls.append(SR(obpip95_lmls[i],bsimcash[i]))
    sr_obpic99_lmls.append(SR(obpic99_lmls[i],bsimcash[i]))
    sr_obpic98_lmls.append(SR(obpic98_lmls[i],bsimcash[i]))
    sr_obpic95_lmls.append(SR(obpic95_lmls[i],bsimcash[i]))
    
sr_obpi = pd.DataFrame({("obpip99","hmhs"):sr_obpip99_hmhs,("obpip98","hmhs"):sr_obpip98_hmhs,("obpip95","hmhs"):sr_obpip95_hmhs,("obpic99","hmhs"):sr_obpic99_hmhs,("obpic98","hmhs"):sr_obpic98_hmhs,("obpic95","hmhs"):sr_obpic95_hmhs,
                        ("obpip99","hmms"):sr_obpip99_hmms,("obpip98","hmms"):sr_obpip98_hmms,("obpip95","hmms"):sr_obpip95_hmms,("obpic99","hmms"):sr_obpic99_hmms,("obpic98","hmms"):sr_obpic98_hmms,("obpic95","hmms"):sr_obpic95_hmms,
                        ("obpip99","hmls"):sr_obpip99_hmls,("obpip98","hmls"):sr_obpip98_hmls,("obpip95","hmls"):sr_obpip95_hmls,("obpic99","hmls"):sr_obpic99_hmls,("obpic98","hmls"):sr_obpic98_hmls,("obpic95","hmls"):sr_obpic95_hmls,
                        ("obpip99","mmhs"):sr_obpip99_mmhs,("obpip98","mmhs"):sr_obpip98_mmhs,("obpip95","mmhs"):sr_obpip95_mmhs,("obpic99","mmhs"):sr_obpic99_mmhs,("obpic98","mmhs"):sr_obpic98_mmhs,("obpic95","mmhs"):sr_obpic95_mmhs,
                        ("obpip99","mmms"):sr_obpip99_mmms,("obpip98","mmms"):sr_obpip98_mmms,("obpip95","mmms"):sr_obpip95_mmms,("obpic99","mmms"):sr_obpic99_mmms,("obpic98","mmms"):sr_obpic98_mmms,("obpic95","mmms"):sr_obpic95_mmms,
                        ("obpip99","mmls"):sr_obpip99_mmls,("obpip98","mmls"):sr_obpip98_mmls,("obpip95","mmls"):sr_obpip95_mmls,("obpic99","mmls"):sr_obpic99_mmls,("obpic98","mmls"):sr_obpic98_mmls,("obpic95","mmls"):sr_obpic95_mmls,
                        ("obpip99","lmhs"):sr_obpip99_lmhs,("obpip98","lmhs"):sr_obpip98_lmhs,("obpip95","lmhs"):sr_obpip95_lmhs,("obpic99","lmhs"):sr_obpic99_lmhs,("obpic98","lmhs"):sr_obpic98_lmhs,("obpic95","lmhs"):sr_obpic95_lmhs,
                        ("obpip99","lmms"):sr_obpip99_lmms,("obpip98","lmms"):sr_obpip98_lmms,("obpip95","lmms"):sr_obpip95_lmms,("obpic99","lmms"):sr_obpic99_lmms,("obpic98","lmms"):sr_obpic98_lmms,("obpic95","lmms"):sr_obpic95_lmms,
                        ("obpip99","lmls"):sr_obpip99_lmls,("obpip98","lmls"):sr_obpip98_lmls,("obpip95","lmls"):sr_obpip95_lmls,("obpic99","lmls"):sr_obpic99_lmls,("obpic98","lmls"):sr_obpic98_lmls,("obpic95","lmls"):sr_obpic95_lmls})


# In[47]:


sr_obpi['simid'] = sr_obpi.index
sr_obpi1 = sr_obpi.melt(id_vars=["simid"])
sr_obpi1.columns = ['simid','strategy','MCtype','SR']
fig = plt.figure(figsize = (10,5)) 
ax = sns.boxplot(x="MCtype", y="SR", hue="strategy",
                 data=sr_obpi1, palette="Set3")

