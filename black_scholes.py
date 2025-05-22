"""
Evaluate Black-Scholes model option price for different input parameters.
"""


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


S = 50.0
K = 53.0
r = 0.05
T = 1.0
sig = 0.1


def black_scholes(S, K, r, T, sig):

    d1 = (np.log(S/K) + T*(r + 0.5*sig*sig))/sig*np.sqrt(T)
    d2 = d1 - sig*np.sqrt(T)

    phi1 = norm.cdf(d1)
    phi2 = norm.cdf(d2)

    C = phi1*S - phi2*K*np.exp(-r*T)

    return C


ans1 = black_scholes(S=S, K=K, r=r, T=T, sig=sig)


zzz = 1



