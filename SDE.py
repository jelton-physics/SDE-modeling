"""
Create simulated time series paths using analytic solution for geometric Brownian motion.
"""


import numpy as np
import matplotlib.pyplot as plt

# mu = 1
# n = 1000
# dt = 0.1
# x0 = 100
# np.random.seed(1)

# sigma = np.arange(0.8, 2, 0.2)

# x = np.exp(
#     (mu - sigma ** 2 / 2) * dt
#     + sigma * np.sqrt(dt) * np.random.normal(0, 1, size=(len(sigma), n)).T
# )
# x = np.vstack([np.ones(len(sigma)), x])
# x = x0 * x.cumprod(axis=0)

# plt.plot(x)
# plt.legend(np.round(sigma, 2))
# plt.xlabel("$t$")
# plt.ylabel("$x$")
# plt.title(
#     "Realizations of Geometric Brownian Motion with different variances\n $\mu=1$"
# )
# plt.show()



mu = 1.0
# sigma = np.arange(0.8, 2, 0.2)
sigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
dt = 0.01
T_i = 0.0
T_f = 1.0
S0 = 100
np.random.seed(10)

t_r = np.arange(T_i, T_f, dt)

for sig in sigma:
    S = np.zeros(len(t_r))
    B = np.zeros(len(t_r))
    S[0] = S0
    B[0] = np.random.normal(loc=0.0, scale=np.sqrt(dt))
    for i in range(1, len(t_r)):
        t = t_r[i]
        dB = np.random.normal(loc=0.0, scale=np.sqrt(dt))
        B[i] = B[i-1] + dB
        S[i] = S0*np.exp((mu - 0.5*(sig**2))*t + sig*B[i])

    plt.plot(t_r, S, label=sig)
    plt.xlabel("$t$")
    plt.ylabel("$x$")
    plt.title(f"Realizations of Geometric Brownian Motion with different variances\n $\mu={mu}$")
plt.legend()
plt.show()

zzz = 1