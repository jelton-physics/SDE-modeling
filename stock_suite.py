"""
Evaluate lots of stock stuff
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.parser import parse
import random


path = "C:\\Users\\jelto\\OneDrive\\Documents\\Research_Projects\\stock_suite\\"

# Import the data from file 'stock_prices_latest.csv'
df = pd.read_csv(path + "stock_prices_latest.csv", parse_dates=['date'])
# df = pd.read_csv(path + "test_data_1.csv", parse_dates=['date'])

df['date'] = pd.to_datetime(df['date'])
N = df.shape[0]


# # Create a small test data set
# df2 = df[df['date'] >= '2018-01-01']
# df2 = df2[(df2['symbol'] == 'MSFT') | (df2['symbol'] == 'AAPL') | (df2['symbol'] == 'AMZN') | (df2['symbol'] == 'BCBP')]
# df2.to_csv(path+'test_data_1.csv', index=False)

# range of available dates
earliest_date = min(df['date'])
latest_date = max(df['date'])

# Number of different stocks
num_stocks = df['symbol'].nunique()
all_symbols = list(df['symbol'].unique())


def time_window(x, num_days):
     
        if num_days is None:
            x_dates = x.reset_index(drop=True)
        else:
            x_dates = x.tail(num_days).reset_index(drop=True)
        
        return_val = (x_dates['close'][len(x_dates)-1] - x_dates['open'][0])/x_dates['open'][0]
        
        return return_val



def aggregate_returns(data, symbols=None, date_start=None, date_end=None, agg_time_by='month', apply_function=None, plot=False):

    # Pick out an individual stock
    df = data[data['symbol'] == 'MSFT']
    n_rows_init = len(df)

    df.sort_values(by='date', ascending=True, inplace=True)

    if date_start is not None:
        df = df[df['date'] >= date_start]

    if date_end is not None:
        df = df[df['date'] <= date_end]

    df.reset_index(drop=True, inplace=True)

    n_rows = len(df)

    if plot:
        # Time series plot
        plt.plot(df['date'], df['close_adjusted'], linestyle = 'dotted')
        plt.title('Time Series Plot of Adjusted Closing Price by Date')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.xticks(rotation=45)
        plt.show()

    df['year'] = [str(df['date'][i])[0:4] for i in range(n_rows)]

    if agg_time_by == 'year':
        df_grouped = df.groupby('year')
    elif agg_time_by == 'month':
        df['month'] = [str(df['date'][i])[5:7] for i in range(n_rows)]
        df_grouped = df.groupby(['year', 'month'])
    elif agg_time_by == 'day':
        df['month'] = [str(df['date'][i])[5:7] for i in range(n_rows)]
        df['day'] = [str(df['date'][i])[8:10] for i in range(n_rows)]
        df_grouped = df.groupby(['year', 'month', 'day'])
    else:
        print(1)

    # Gain between adjacent rows


    return 2



def eoy_returns(data, symbols=None, date_start=None, date_end=None, agg_time_by='month', apply_function=None, num_days=7, date_shift_days=0, plot=False):

    df = data[data['symbol'].isin(symbols)]
    n_rows_init = len(df)

    df.sort_values(by='date', ascending=True, inplace=True)

    if date_start is not None:
        df = df[df['date'] >= date_start]

    if date_end is not None:
        df = df[df['date'] <= date_end]

    df.reset_index(drop=True, inplace=True)

    df['year'] = df['date'].dt.year
    min_year = min(df['year'])
    max_year = max(df['year'])
    num_yrs = max_year - min_year + 1

    # Apply date shift (shift back 3 days, then time period of interest is 12-21 through 12-31)
    if date_shift_days != 0:
        df['date'] = df['date'] - timedelta(days=date_shift_days)
    
    # Drop extra rows in fake year
    df = df[df['year'] >= min_year]

    # Pick out an individual stock
    # Make sure first symbol contains the max date range
    first_sym_loop = (df.groupby('symbol')['date'].max() - df.groupby('symbol')['date'].min()).idxmax()
    symbols.remove(first_sym_loop)
    symbols.insert(0, first_sym_loop)

    first_sym = True
    n_s_all = num_yrs*[1]
    for symbol in symbols:
        df_sym = df[df['symbol'] == symbol]

        if plot:
            # Time series plot
            plt.plot(df_sym['date'], df_sym['close_adjusted'], linestyle = 'dotted')
            plt.title('Time Series Plot of Adjusted Closing Price by Date')
            plt.xlabel('Date')
            plt.ylabel('Adjusted Close Price')
            plt.xticks(rotation=45)
            plt.show()

        if agg_time_by == 'year':
            df_grouped = df_sym.groupby('year')
        elif agg_time_by == 'month':
            df_sym['month'] = df_sym['date'].dt.month
            df_grouped = df_sym.groupby(['year', 'month'])
        elif agg_time_by == 'day':
            df_sym['month'] = df_sym['date'].dt.month
            df_sym['day'] = df_sym['date'].dt.day
            df_grouped = df_sym.groupby(['year', 'month', 'day'])

        # Apply time_window function to compute return over different lengths of time
        df_agg_days = df_grouped.apply(time_window, num_days, include_groups=False)
        df_agg_days = df_agg_days.to_frame(name="return")

        df_agg_days.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_agg_days.dropna(inplace=True)

        df_agg_yr = df_grouped.apply(time_window, None, include_groups=False)
        df_agg_yr = df_agg_yr.to_frame(name="yr_return")

        df_agg_yr['yr_return_norm'] = df_agg_yr['yr_return']*(num_days/252)

        df_agg_yr['days_return'] = df_agg_days['return']

        df_agg_yr.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_agg_yr.dropna(inplace=True)

        if first_sym:
            df_agg_yr_all = df_agg_yr.copy(deep=True)
            first_sym = False
        else:
            for i in range(num_yrs):
                curr_yr = min_year + i
                # Only update the average if the current symbol has data from this year
                if curr_yr in df_agg_yr.index: 
                    n_s_all[i] += 1
                    n_s_curr = n_s_all[i]
                    # df_agg_yr_all['yr_return'][curr_yr] = (1.0/n_s_curr)*df_agg_yr['yr_return'][curr_yr]  + ((n_s_curr - 1.0)/n_s_curr)*df_agg_yr_all['yr_return'][curr_yr] 
                    df_agg_yr_all.loc[curr_yr, 'yr_return'] = (1.0/n_s_curr)*df_agg_yr['yr_return'][curr_yr]  + ((n_s_curr - 1.0)/n_s_curr)*df_agg_yr_all['yr_return'][curr_yr]
                    # df_agg_yr_all['yr_return_norm'][curr_yr]  = df_agg_yr_all['yr_return'][curr_yr] *(num_days/252)
                    df_agg_yr_all.loc[curr_yr, 'yr_return_norm'] = df_agg_yr_all['yr_return'][curr_yr] *(num_days/252)
                    # df_agg_yr_all['days_return'][curr_yr]  = (1.0/n_s_curr)*df_agg_yr['days_return'][curr_yr]  + ((n_s_curr - 1.0)/n_s_curr)*df_agg_yr_all['days_return'][curr_yr]  
                    df_agg_yr_all.loc[curr_yr, 'days_return'] = (1.0/n_s_curr)*df_agg_yr['days_return'][curr_yr]  + ((n_s_curr - 1.0)/n_s_curr)*df_agg_yr_all['days_return'][curr_yr]
                  
    df_agg_yr_all['eoy_excess_gain'] = df_agg_yr_all['days_return'] - df_agg_yr_all['yr_return_norm']
    df_agg_yr_all['N_stocks'] = n_s_all
    df_agg_yr_all['larger_returns_bool'] = [1 if df_agg_yr_all['eoy_excess_gain'][i] > 0 else 0 for i in df_agg_yr_all.index]

    n_yrs = len(df_agg_yr_all)
    n_greater = np.sum(df_agg_yr_all['larger_returns_bool'])
    frac_greater = float(n_greater)/n_yrs


    return df_agg_yr_all, frac_greater





# aggregate_returns(data=df, symbols=['MSFT'], date_start='2010-01-01', agg_time_by='year', apply_function=time_window, plot=False)

# sym = ['MSFT', 'AAPL', 'AMZN']
sym = random.sample(all_symbols, k=5000)

df_agg_yr_all, frac_greater = eoy_returns(data=df, symbols=sym, date_start='1998-01-01', date_end='2021-01-03', agg_time_by='year', apply_function=time_window, num_days=7, date_shift_days=3, plot=False)



# Create a DB-type row for each stock

# Restrict date range
first_date = '2000-01-01'
last_date = '2010-12-29'

df_r = df[(df['Date'] >= first_date) & (df['Date'] <= last_date)]

# Time series plot
plt.plot(df_r['Date'], df_r['High'], linestyle = 'dotted')
plt.title('Time Series Plot')
plt.xlabel('Time')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.show()

zzz = 1