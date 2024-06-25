import numpy as np

from product.params import Params


class MonteCarloPricer():
    def __init__(self, S0, T, params: Params, dt, N = 100000, seed = None):
        self._S0 = S0
        self._T = T
        self._dt = dt
        self._sigma, self._r, self._b, _ = params.unpack()
        self._N = N # number of paths
        self._seed = seed

        self._Nt = int(self._T / self._dt) # number of observations
        self._arr_S = np.ones((self._Nt + 1, self._N)) # (T+1) x paths
        self._arr_payoff = np.empty(self._N)

    @property
    def seed(self): return self._seed
    
    @seed.setter
    def seed(self, seed): self._seed = seed

    """
    dS = mu * S * dt + sigma * S * dW
    d lnS = (mu - sigma^2/2) * dt + sigma * dW
    ln S(t+Dt) - ln St = (mu - sigma^2/2) * Dt + sigma * e * sqrt(Dt)
    ln ST - ln S0 = (mu - sigma^2/2) * T + sigma * Sum(e) * sqrt(Dt)
    ST = S0 * exp(...)
    """

    def gen_S_paths(self):
        self._arr_S = np.ones((self._Nt + 1, self._N)) # reset

        np.random.seed(self.seed)
        e = np.random.normal(size=(self._Nt, self._N)) # T x paths
        
        self._arr_S[0, :] = self._S0
        mu = self._b - self._sigma**2/2 # different from the mu' in pricing formulas: mu' = mu/sigma**2

        self._arr_S[1:, :] = self._S0 * np.exp(
            np.cumsum(
                mu * self._dt + self._sigma * e * np.sqrt(self._dt),
                axis=0
            )
        )

        """ another way
        W = np.cumsum(e, axis=0) # cumumlated e

        T_vals = np.linspace(self._dt, self._T, self._Nt)
        T_arr = np.ones((self._Nt, self._N)) * T_vals.reshape(-1,1) # element-wise multiplication
        # OR
        T_arr = np.cumsum(self._dt * np.ones((self._Nt, self._N)), axis=0)

        self._arr_S[1:, :] = self._S0 * np.exp(
            mu * T_arr + self._sigma * W * np.sqrt(self._dt)
        )
        """
    
    def gen_payoffs(self, product):
        self._arr_payoff = np.empty(self._N) # reset

        for j in range(0, self._N):
            barrier = option.barrier
            ko = barrier.test_ko(self._arr_S[:, j])
            if ko:
                self._arr_payoff[j] = option.payoff_ko()
            else:
                self._arr_payoff[j] = option.payoff_not_ko(self._arr_S[-1, j])

    def price(self, product):
        self.gen_S_paths()
        self.gen_payoffs(product)

        return self._arr_payoff.mean() * np.exp(- self._r * self._T)
    