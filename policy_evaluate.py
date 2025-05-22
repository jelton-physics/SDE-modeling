"""
Evaluate different buy/sell policies on a collectionf of different tickers.
"""

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from mplcursors import cursor
import yfinance as yf
# pd.options.display.max_rows=10  # To decrease printouts


all_tickers = {"MLGO": ['2025-02-21', '2025-02-22'], "KDLY": ['2025-02-10', '2025-02-15'], "DOMH": ['2025-02-10', '2025-02-15'], "AIFF": ['2025-02-10', '2025-02-15'], "SOPA": ['2025-02-10', '2025-02-15'], "NNE": ['2025-01-17', '2025-02-07'], "DWTX": ['2025-01-22', '2025-01-29'], "BBAI": ['2025-02-03', '2025-02-15'],
                "NVDA": ['2025-02-10', '2025-02-15'], "RCAT": ['2025-02-10', '2025-02-15'], "RGTI": ['2025-02-10', '2025-02-15'], "SOUN": ['2025-02-10', '2025-02-15']}
period = '60d'
interval = '5m'
initial_investment = 1.0
desired_times = '09:30|16:00'
prepost = True  # whether to get extended hours data
run_plots = True
run_policies = True

# all_tickers = {"KDLY": ['2025-02-10', '2025-02-15'], "DOMH": ['2025-02-10', '2025-02-15'], "AIFF": ['2025-02-10', '2025-02-15'], "SOPA": ['2025-02-10', '2025-02-15'], "NNE": ['2025-01-17', '2025-02-07'], "DWTX": ['2025-01-22', '2025-01-29'], "BBAI": ['2025-02-03', '2025-02-15'],
#                 "NVDA": ['2025-02-10', '2025-02-15'], "RCAT": ['2025-02-10', '2025-02-15'], "RGTI": ['2025-02-10', '2025-02-15'], "SOUN": ['2025-02-10', '2025-02-15']}
# period = '60d'
# interval = '5m'
# initial_investment = 1.0
# desired_times = '09:30|16:00'
# prepost = True  # whether to get extended hours data
# run_plots = True
# run_policies = False


df_results = pd.DataFrame(columns = ['ticker', 'policy_1_return'])
df_results['ticker'] = list(all_tickers.keys())


########################################################################################

### Policy 1: Buy at beginning of period, sell at end
def policy_1(df, initial_investment):

    price_buy = df.iloc[0]['Open']
    price_sell = df.iloc[len(df)-1]['Close']
    gain = price_sell - price_buy
    gain_factor = price_sell/price_buy
    percent_gain = gain/price_buy

    final_investment = initial_investment*gain_factor
    profit = final_investment - initial_investment
    return_investment = profit/initial_investment
    return_percent = return_investment*100

    return final_investment, return_investment, return_percent


### Policy 2: Buy at 12:00 each day, sell next morning at 10:00
def policy_2(df, initial_investment):

    df_two_times = df[(df['Time'] == '10:00:00') | (df['Time'] == '12:00:00')].reset_index(drop=True)

    final_investment = initial_investment
    for i in range(1, len(df_two_times)):
        if i % 2 == 1:
            price_buy = df_two_times.iloc[i]['Open']
        else:
            price_sell = df_two_times.iloc[i]['Close']
            gain_factor = price_sell/price_buy
            final_investment = final_investment*gain_factor

    profit = final_investment - initial_investment
    return_investment = profit/initial_investment
    return_percent = return_investment*100

    return final_investment, return_investment, return_percent


### Policy "same day surge": Buy a stock that has already gone up a lot early, hope it will surge even higher then sell.
# Buy at 11:00, sell at 14:00 is too naive but can start with this.
def policy_same_day_surge(df, initial_investment):

    morning_time = '09:35:00'
    morning_row = df[df['Time'] == morning_time]
    morning_val = morning_row['Open'].values[0]

    buy_factor = 1.5
    buy_time = '10:45:00'

    for j in range(len(df)):
        if (df['Open'][j] >= morning_val*buy_factor) & (df['Time'][j] >= buy_time):
            price_buy = df['Open'][j]
            break

    sell_factor = 1.9
    sell_time = '13:30:00'

    for k in range(j, len(df)):
        if (df['Open'][k] >= price_buy*sell_factor) & (df['Time'][k] >= sell_time):
            price_sell = df['Open'][k]
            break

    gain_factor = price_sell/price_buy
    final_investment = initial_investment*gain_factor

    profit = final_investment - initial_investment
    return_investment = profit/initial_investment
    return_percent = return_investment*100

    return final_investment, return_investment, return_percent


########################################################################################


### Loop through all tickers and run policies

for ticker_symbol, start_end_dates in all_tickers.items():

    start_date = start_end_dates[0]
    end_date_next_day = start_end_dates[1]  # irrational convention for filtering dates, end date is one less than this

    dat = yf.Ticker(ticker_symbol)
    # dat.info
    # dat.calendar
    # dat.analyst_price_targets
    # dat.quarterly_income_stmt

    # period must be of the format 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max, etc.
    # granularity of 1m can only go back 8 trading days
    # granularity of 5m can go back 60 trading days
    # dat.option_chain(dat.options[0]).calls

    # dat.history(period='8d', interval="1m")
    df = dat.history(period=period, interval=interval, auto_adjust=True, prepost=prepost).reset_index()

    # Alt format
    # df2 = yf.download(ticker_symbol, period="60d", interval="5m", actions=True, auto_adjust=True).reset_index()

    # Filter out to only a speific date range
    df = df[df['Datetime'] >= start_date]
    df = df[df['Datetime'] < end_date_next_day]
    df.reset_index(drop=True, inplace=True)

    # Separate date and time and add a string Datetime for plotting
    df['Date'] = [str(d.date()) for d in df['Datetime']]
    df['Time'] = [str(d.time()) for d in df['Datetime']]
    df['Datetime_string'] = df['Datetime'].astype(str)
    df['Datetime_string'] = df['Datetime_string'].str[:-9]
    df['Datetime_string'] = df['Datetime_string'].str[2:]
    xticks = df['Datetime_string'][df['Datetime_string'].str.contains(desired_times)]

    
    ### Plots
    if run_plots:
        # Time series plot
        plt.plot(df['Datetime_string'], df['Open'], linestyle = 'solid')
        plt.title(f'Time Series Plot of Price for {ticker_symbol}')
        plt.xlabel('Date and Time')
        plt.ylabel('Price')
        plt.xticks(xticks) 
        plt.xticks(rotation='vertical')
        plt.grid()
        plt.tight_layout()
        cursor(hover=True)
        plt.show()


    ### Run all policies

    if run_policies:

        final_investment_1, return_investment_1, return_percent_1 = policy_same_day_surge(df, initial_investment)

        # final_investment_1, return_investment_1, return_percent_1 = policy_1(df, initial_investment)

        # final_investment_2, return_investment_2, return_percent_2 = policy_2(df, initial_investment)

        idx = df_results[df_results['ticker'] == ticker_symbol].index.values[0]
        df_results.loc[idx, 'policy_same_day_surge_return'] = return_investment_1
        # df_results.loc[idx, 'policy_1_return'] = return_investment_1
        # df_results.loc[idx, 'policy_2_return'] = return_investment_2


zzz = 1