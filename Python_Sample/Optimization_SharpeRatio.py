		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
import pandas as pd  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
import matplotlib.pyplot as plt  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
import numpy as np  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
import datetime as dt  	
import scipy.optimize as spo
import os
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
# Code to optimize a portfolio of stocks based on their Sharpe Ratio Value

def optimize_portfolio(sd=dt.datetime(2008,6,1), ed=dt.datetime(2009,6,1), \
    syms=['GOOG', 'AAPL', 'GLD', 'XOM', 'IBM']  , gen_plot=False):  

    def symbol_to_path(symbol, base_dir=None):  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        """Return CSV file path given ticker symbol."""  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        if base_dir is None:  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
            base_dir = os.environ.get("MARKET_DATA_DIR", 'data/')  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        return os.path.join(base_dir, "{}.csv".format(str(symbol))) 

    def get_data(symbols, dates, addSPY=True, colname = 'Adj Close'):                                                                                                                                   
        """Read stock data (adjusted close) for given symbols from CSV files."""                                                                                                                                    
        df = pd.DataFrame(index=dates)                                                                                                                                      
        if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent                                                                                                                                     
            symbols = ['SPY'] + list(symbols) # handles the case where symbols is np array of 'object'                                                                                                                                      
                                                                                                                                        
        for symbol in symbols:                                                                                                                                      
            df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',                                                                                                                                     
                    parse_dates=True, usecols=['Date', colname], na_values=['nan'])                                                                                                                                     
            df_temp = df_temp.rename(columns={colname: symbol})                                                                                                                                     
            df = df.join(df_temp)                                                                                                                                   
            if symbol == 'SPY':  # drop dates SPY did not trade                                                                                                                                     
                df = df.dropna(subset=["SPY"])                                                                                                                                      
                                                                                                                                        
        return df

    def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):                                                                                                                                     
        import matplotlib.pyplot as plt                                                                                                                                     
        """Plot stock prices with a custom title and meaningful axis labels."""                                                                                                                                     
        ax = df.plot(title=title, fontsize=12)                                                                                                                                      
        ax.set_xlabel(xlabel)                                                                                                                                   
        ax.set_ylabel(ylabel)                                                                                                                                   
        plt.show()               		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # Read in adjusted closing prices for given symbols, date range  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    dates = pd.date_range(sd, ed)  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    prices = prices_all[syms]  # only portfolio symbols  
  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    prices_SPY = prices_all['SPY']
    prices_SPY = prices_SPY/prices_SPY.iloc[0]  # only SPY, for comparison later  
    #print(prices_SPY)		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # find the allocations for the optimal portfolio  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    normed_=prices/prices.iloc[0]	

    def sharpe_ratio(allocs, normed_):
        alloced = normed_ * allocs
        start_value_= 1
        pos_value = alloced * start_value_
        port_val = pos_value.sum(axis=1)
        daily_rets = pd.DataFrame(data=port_val)
        daily_rets = (daily_rets / daily_rets.shift(1))-1
        daily_rets = daily_rets[1:]
        risk_free_ret  = 0.0
        trading_days = 252.0
        sr_ = np.sqrt(252.0) * float(daily_rets.mean())	/ (float(daily_rets.std()))  
        return -1.0* sr_


    bounnds = tuple((0, 1) for x in range(0,len(syms)))
    constraints = ({ 'type': 'eq', 'fun': lambda allocs: 1.0 - np.sum(allocs) })
    x_guess= np.full((1,len(syms)), 1.0/len(syms)) 
    # Optimizing Allocations of Portfolio by Sharpe Ratio
    alloced_opt = spo.minimize(sharpe_ratio, x_guess,args=normed_ ,method='SLSQP', bounds=bounnds,constraints=constraints, tol=1e-10)
    allocs = alloced_opt.x

    # Share Ratio Definition
    def sharpe_ratio_opt(allocs, normed_):
        alloced = normed_ * allocs
        start_value_= 1
        pos_value = alloced * start_value_
        port_val = pos_value.sum(axis=1)
        daily_rets = pd.DataFrame(data=port_val)
        daily_rets = (daily_rets / daily_rets.shift(1))-1
        daily_rets = daily_rets[1:]
        risk_free_ret  = 0.0
        trading_days = 252.0
        sr_ = np.sqrt(252.0) * float(daily_rets.mean())	/ (float(daily_rets.std()))  
        return (port_val[-1]/port_val.iloc[0])-1, float(daily_rets.mean()), float(daily_rets.std()),  sr_, port_val 

    cr, adr, sddr, sr, port_val = sharpe_ratio_opt(allocs, normed_)
    #print(cr) 		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
 		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # Compare daily portfolio value with SPY using a normalized plot  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    if gen_plot:  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        # add code to plot here  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        plt.plot(df_temp)#, title="Stock prices", xlabel="Date", ylabel="Price")
        plt.title("Portfolio Optimization - Sharpe Ratio")
        plt.xlabel("Date")
        plt.ylabel("Returns")
        plt.gca().legend(('Portfolio','SPY'), loc='lower right')
        plt.savefig('plot.png', bbox_inches='tight')      		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        pass  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    return allocs, cr, adr, sddr, sr  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
def test_code():  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
 	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # Defining time and Tickers to optimize 
    start_date = dt.datetime(2010,6,1)  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    end_date = dt.datetime(2011,6,1)  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    symbols = ['GOOG', 'JNJ', 'XOM', 'JPM', 'JCP', 'AA', 'HAL', 'KMB', 'GLD', 'AAPL'] 		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # Assess the portfolio  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # Print statistics  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Start Date: {start_date}")  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    print(f"End Date: {end_date}")  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Symbols: {symbols}")  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Allocations:{allocations}")  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Sharpe Ratio: {sr}")  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Average Daily Return: {adr}")  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Cumulative Return: {cr}")  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
if __name__ == "__main__":  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    test_code()  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
