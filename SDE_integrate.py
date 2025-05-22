"""
Create simulated time series paths by integrating stochastic differential equations for different models.
Use geometric Brownian motion, GBM with addition of Poisson jumps, GBM with addition of oscillatory term.
"""


import numpy as np
import matplotlib.pyplot as plt


class SDE_TimeSeries():
    """
    Blah.
    """

    def __init__(self, X):
        self.X = X


    def mu_ou(self, y, t):
        """Implement the Ornstein–Uhlenbeck mu."""
        return THETA * (MU - y)


    def mu_t(self, y, t):
        """Implement mu."""
        return MU


    def sigma(self, y, t):
        """Implement sigma."""
        return SIGMA


    def dW(self, delta_t):
        """Sample a random number at each call."""
        return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))


    def run_simulation_ornstein_uhlenbeck(self):
        ys = np.zeros(TS.size)
        ys[0] = Y_INIT
        for i in range(1, TS.size):
            # t = T_INIT + (i - 1) * DT
            t = TS[i-1]
            y = ys[i - 1]
            ys[i] = y + self.mu_ou(y, t)*DT + self.sigma(y, t)*self.dW(DT)
        return ys


    def run_simulation_gbm(self):
        ys = np.zeros(TS.size)
        ys[0] = Y_INIT
        for i in range(1, TS.size):
            # t = T_INIT + (i - 1) * DT
            t = TS[i-1]
            y = ys[i - 1]

            a = self.mu_t(y, t)*y
            b1 = self.sigma(y, t)*y
            gam = y + a*DT + b1*np.sqrt(DT)
            b2 = self.sigma(gam, t)*gam
            rk_term = 0.5*(b2 - b1)*(self.dW(DT)**2 - DT)*(1.0/np.sqrt(DT))

            # Euler
            # ys[i] = y + a*DT + b1*self.dW(DT)

            # RK
            ys[i] = y + a*DT + b1*self.dW(DT) + rk_term

        return ys


    def run_simulation_poisson_jump(self):
        ys = np.zeros(TS.size)
        ys[0] = Y_INIT
        for i in range(1, TS.size):
            t = TS[i-1]
            y = ys[i - 1]

            a = self.mu_t(y, t)*y
            b1 = self.sigma(y, t)*y
            gam = y + a*DT + b1*np.sqrt(DT)
            b2 = self.sigma(gam, t)*gam
            rk_term = 0.5*(b2 - b1)*(self.dW(DT)**2 - DT)*(1.0/np.sqrt(DT))


            poisson_var = np.random.poisson(lam = LAMBDA, size=1)[0]
            if poisson_var == 0:
                jump_size = 0
            else:
                jump_size = 1
                for j in range(poisson_var):
                    J_g = np.random.normal(loc=BETA, scale=ETA)
                    jump_size = jump_size*(np.exp(J_g) - 1)

            # Euler
            # ys[i] = y + a*DT + b1*self.dW(DT) + y*jump_size

            # RK
            ys[i] = y + a*DT + b1*self.dW(DT) + rk_term + y*jump_size

        return ys


    def run_simulation_oscillation(self):
        ys = np.zeros(TS.size)
        xs = np.zeros(TS.size)
        ys[0] = Y_INIT
        xs[0] = X_INIT
        for i in range(1, TS.size):
            t = TS[i-1]
            y = ys[i - 1]
            x = xs[i - 1]

            a = self.mu_t(y, t)*y
            b1 = self.sigma(y, t)*y
            gam = y + a*DT + b1*np.sqrt(DT)
            b2 = self.sigma(gam, t)*gam
            rk_term = 0.5*(b2 - b1)*(self.dW(DT)**2 - DT)*(1.0/np.sqrt(DT))


            # poisson_var = np.random.poisson(lam = LAMBDA, size=1)[0]
            # if poisson_var == 0:
            #     jump_size = 0
            # else:
            #     jump_size = 1
            #     for j in range(poisson_var):
            #         J_g = np.random.normal(loc=BETA, scale=ETA)
            #         jump_size = jump_size*(np.exp(J_g) - 1)

            # Euler
            xs[i] = x + y*DT
            ys[i] = y + a*DT + b1*self.dW(DT) - K*x*DT

            # RK
            # ys[i] = y + a*DT + b1*self.dW(DT) + rk_term + y*jump_size

        return ys


###################################################################################################


T_INIT = 0
T_END = 1
DT = 0.01
N = float(T_END - T_INIT) / DT
TS = np.arange(T_INIT, T_END + DT, DT)
Y_INIT = 100
X_INIT = 0

"""Stochastic model constants."""
THETA = 0.7
MU = 0.2
SIGMA = 1.0
LAMBDA = 0.2
BETA = 0.0
ETA = 0.1
K = 5.0
X1 = 1


if __name__ == "__main__":

    NUM_SIMS = 5

    # np.random.seed(20)

    for _ in range(NUM_SIMS):

        S1 = SDE_TimeSeries(X1)

        # ys = S1.run_simulation_gbm()
        ys = S1.run_simulation_poisson_jump()
        # ys = S1.run_simulation_oscillation()
        plt.plot(TS, ys)
        plt.xlabel("time")
        plt.ylabel("y")
        plt.title(f"Realizations of Geometric Brownian Motion with $\mu={MU}$, $\sigma={SIGMA}$")
    plt.show()

    zzz = 1