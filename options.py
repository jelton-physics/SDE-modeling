"""
Evaluate options prices.
"""


import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt




def LeisenReimerBinomial(OutputFlag, AmeEurFlag, CallPutFlag, S, X, T, r, c, v, n):
    # This functions calculates the implied volatility of American and European options
    # This code is based on "The complete guide to Option Pricing Formulas" by Espen Gaarder Haug (2007)
    # Translated from a VBA code
    # OutputFlag:
    # "P" Returns the options price
    # "d" Returns the options delta
    # "a" Returns an array containing the option value, delta and gamma
    # AmeEurFlag:
    # "a" Returns the American option value
    # "e" Returns the European option value
    # CallPutFlag:
    # "C" Returns the call value
    # "P" Returns the put value
    # S is the share price at time t
    # X is the strike price
    # T is the time to maturity in years (days/365)
    # r is the risk-free interest rate
    # c is the cost of carry rate
    # v is the volatility
    # n determines the stepsize

    # Start of the code
    # rounds n up tot the nearest odd integer (the function is displayed below the LeisenReimerBinomial function in line x)
    n = round_up_to_odd(n)

    # Creates a list with values from 0 up to n (which will be used to determine to exercise or not)
    n_list = np.arange(0, (n + 1), 1)

    # Checks if the input option is a put or a call, if not it returns an error
    if CallPutFlag == 'C':
        z = 1
    elif CallPutFlag == 'P':
        z = -1
    else:
        return 'Call or put not defined'

    # d1 and d2 formulas of the Black-Scholes formula for European options
    d1 = (np.log(S / X) + (c + v ** 2 / 2) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)

    # The Preizer-Pratt inversion method 1
    hd1 = 0.5 + np.sign(d1) * (0.25 - 0.25 * np.exp(-(d1 / (n + 1 / 3 + 0.1 / (n + 1))) ** 2 * (n + 1 / 6))) ** 0.5

    # The Preizer-Prat inversion method 2
    hd2 = 0.5 + np.sign(d2) * (0.25 - 0.25 * np.exp(-(d2 / (n + 1 / 3 + 0.1 / (n + 1))) ** 2 * (n + 1 / 6))) ** 0.5

    # Calculates the stepsize in years
    dt = T / n

    # The probability of going up
    p = hd2

    # The up and down factors
    u = np.exp(c * dt) * hd1 / hd2
    d = (np.exp(c * dt) - p * u) / (1 - p)
    df = np.exp(-r * dt)

    # Creates the most right column of the tree
    max_pay_off_list = []
    for i in n_list:
        i = i.astype('int')
        max_pay_off = np.maximum(0, z * (S * u ** i * d ** (n - i) - X))
        max_pay_off_list.append(max_pay_off)

    # The binominal tree
    for j in np.arange(n - 1, 0 - 1, -1):
        for i in np.arange(0, j + 1, 1):
            i = i.astype(int)  # Need to be converted to a integer
            if AmeEurFlag == 'e':
                max_pay_off_list[i] = (p * max_pay_off_list[i + 1] + (1 - p) * max_pay_off_list[i]) * df
            elif AmeEurFlag == 'a':
                max_pay_off_list[i] = np.maximum((z * (S * u ** i * d ** (j - i) - X)),
                                                 (p * max_pay_off_list[i + 1] + (1 - p) * max_pay_off_list[i]) * df)
        if j == 2:
            gamma = ((max_pay_off_list[2] - max_pay_off_list[1]) / (S * u ** 2 - S * u * d) - (
                    max_pay_off_list[1] - max_pay_off_list[0]) / (S * u * d - S * d ** 2)) / (
                            0.5 * (S * u ** 2 - S * d ** 2))
        if j == 1:
            delta = ((max_pay_off_list[1] - max_pay_off_list[0])) / (S * u - S * d)
    price = max_pay_off_list[0]

    # Put all the variables in the list
    variable_list = [delta, gamma, price]

    # Return values
    if OutputFlag == 'P':
        return price
    elif OutputFlag == 'd':
        return delta
    elif OutputFlag == 'g':
        return gamma
    elif OutputFlag == 'a':
        return variable_list
    else:
        return 'Indicate if you want to return P, d, g or a'


def round_up_to_odd(n):
    # This function returns a number rounded up to the nearest odd integer
    # For example when n = 100, the function returns 101
    return np.ceil(n) // 2 * 2 + 1




# Inputs

ticker = 'META'
type = 'call'
strike = 630
buy_date = date(2025,3,10)
exp_date = date(2025,4,4)
premium_buy = 15.99
stock_val_buy = 600
num_contracts = 2
purchase_cost = 3199.3


DTE_init = (exp_date-buy_date).days


### Function params

# OutputFlag:
# "P" Returns the options price
# "d" Returns the options delta
# "a" Returns an array containing the option value, delta and gamma
# AmeEurFlag:
# "a" Returns the American option value
# "e" Returns the European option value
# CallPutFlag:
# "C" Returns the call value
# "P" Returns the put value
# S is the share price at time t
# X is the strike price
# T is the time to maturity in years (days/365)
# r is the risk-free interest rate
# c is the cost of carry rate
# v is the volatility
# n determines the stepsize


stock_vals = np.arange(0, 0.3, 0.005)

stock_vals = stock_val_buy*(1 + stock_vals)

stock_returns = np.round(100*(stock_vals - stock_val_buy)/stock_val_buy, 1)
option_returns = np.array(len(stock_returns)*[0.0])

# Perhaps look at option vals as function of DTE and stock val separate and together, 3Dplot?
DTE_fixed = 4

for i, curr_val in enumerate(stock_vals):

    option_res = LeisenReimerBinomial(OutputFlag='P', AmeEurFlag='a', CallPutFlag='C', S=curr_val, X=strike, T=(DTE_fixed/365.0), r=0.0407, c=0.01, v=0.4020, n=300)

    option_returns[i] = np.round(100*(option_res - premium_buy)/premium_buy, 1)

stock_returns = list(stock_returns)
option_returns = list(option_returns)


plt.scatter(stock_returns, option_returns)
plt.show()


zzz = 1

