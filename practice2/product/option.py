import numpy as np
from scipy.stats import norm
from scipy.special import zeta

from product.params import Params
from product.barrier import Barrier

class Option():
    def __init__(self, spot, T, params: Params):
        # S
        self._S = spot
        # time to maturity
        self._T = T
        # (volatility, rate, dividend) = (sigma, r, r - coc)
        self._params = params
    
    @property
    def S(self):
        return self._S
    
    @S.setter
    def S(self, spot):
        self._S = spot

    @property
    def T(self):
        return self._T
    
    @T.setter
    def T(self, T):
        self._T = T

    @property
    def params(self):
        return self._params
    
    @property
    def sigma(self):
        return self.params.sigma

    @property
    def r(self):
        return self.params.r

    @property
    def coc(self):
        return self.params.coc
    
    @property
    def q(self):
        return self.params.q
    
    @sigma.setter
    def sigma(self, sigma):
        self._params = Params(sigma, self.r, self.coc)


# up-in or down-in, cash at expiration
class BinaryBarrier(Option):
    def __init__(
        self, spot, T, params: Params,
        H, K, up_in
    ):
        super().__init__(spot, T, params)
        self._up_in = up_in
        self._H = H # barrier
        self._K = K # knock-out return
    
    @property
    def H(self):
        return self._H

    @property
    def K(self):
        return self._K

    @property
    def up_in(self):
        return self._up_in
    
    def price(self):        
        S, T, H, K = self.S, self.T, self.H, self.K
        sigma, r, coc = self.params.get_params()

        # knocked-in
        if (self.up_in and S >= H) or ((not self.up_in) and S <= H):
            return K * np.exp(-r*T)     

        vol = sigma * np.sqrt(T)

        # mu = (coc - sigma^2/2) / sigma^2
        mu = coc / sigma**2 - 0.5

        eta = -1
        phi = 1
        if not self.up_in:
            eta = -eta
            phi = -phi

        x2 = np.log(S/H) / vol + (1+mu) * vol
        y2 = np.log(H/S) / vol + (1+mu) * vol
        
        B2 = K*np.exp(-r*T)*norm.cdf(phi*(x2-vol))
        B4 = K*np.exp(-r*T)*(H/S)**(2*mu)*norm.cdf(eta*(y2-vol))

        return B2 + B4


# no rebate, up-out call or down-out put
class StandardBarrier(Option):
    def __init__(
        self, spot, T, params: Params,
        X, H, up_out
    ):
        super().__init__(spot, T, params)
        self._up_out = up_out
        self._X = X # strike
        self._H = H # barrier
    
    @property
    def X(self):
        return self._X

    @property
    def H(self):
        return self._H

    @property
    def up_out(self):
        return self._up_out

    def price(self):
        S, T, X, H = self.S, self.T, self.X, self.H

        # knocked-out
        if (self.up_out and S >= H) or ((not self.up_out) and S <= H):
            return 0

        sigma, r, coc = self.params.get_params()
        q = self.params.q

        vol = sigma * np.sqrt(T)

        # mu = (coc - sigma^2/2) / sigma^
        mu = coc / sigma**2 - 0.5

        x1 = np.log(S/X) / vol + (1+mu) * vol
        x2 = np.log(S/H) / vol + (1+mu) * vol

        y1 = np.log(H**2 / (S*X)) / vol + (1+mu) * vol
        y2 = np.log(H/S) / vol + (1+mu) * vol

        eta = -1
        phi = 1
        if not self.up_out:
            eta = -eta
            phi = -phi
        
        A = S*np.exp(-q*T)*norm.cdf(phi*x1) - X*np.exp(-r*T)*norm.cdf(phi*(x1-vol))
        B = S*np.exp(-q*T)*norm.cdf(phi*x2) - X*np.exp(-r*T)*norm.cdf(phi*(x2-vol))
        C = S*np.exp(-q*T)*(H/S)**(2*mu+2)*norm.cdf(eta*y1) - X*np.exp(-r*T)*(H/S)**(2*mu)*norm.cdf(eta*(y1-vol))
        D = S*np.exp(-q*T)*(H/S)**(2*mu+2)*norm.cdf(eta*y2) - X*np.exp(-r*T)*(H/S)**(2*mu)*norm.cdf(eta*(y2-vol))

        A *= phi
        B *= phi
        C *= phi
        D *= phi

        return A - B + C - D

class SharkFin(Option):
    def __init__(
        self, spot, T, params: Params,
        X, H, dt, r_min, r_ko, participation, is_call
    ):
        super().__init__(spot, T, params)
        self._X = X # strike
        self._H = H # barrier
        self._dt = dt
        self._r_min = r_min
        self._r_ko = r_ko # knock-out return
        self._pt = participation
        self._is_call = is_call
        self._ko = False
        self._name = 'Shark Fin'

    @property
    def X(self):
        return self._X

    @property
    def H(self):
        return self._H

    @property
    def dt(self):
        return self._dt

    @property
    def r_min(self):
        return self._r_min

    @property
    def r_ko(self):
        return self._r_ko
    
    @property
    def pt(self):
        return self._pt
    
    @property
    def is_call(self):
        return self._is_call
    
    @property
    def barrier(self):
        if self.is_call:
            return Barrier(H_in=None, H_out=self.H, ki_type='na', ko_type='up')
        else:
            return Barrier(H_in=None, H_out=self.H, ki_type='na', ko_type='down')
    
    @property
    def ko(self):
        return self.barrier.test_ko(self.S)
    
    @property
    def name(self):
        if self.is_call:
            return f'{self._name} Call'
        else:
            return f'{self._name} Put'

    def payoff_ko(self):
        return self.r_ko

    def payoff_not_ko(self, S_t):
        if self.is_call:
            return self.pt * max(S_t - self.X, 0) + self.r_min
        else:
            return self.pt * max(self.X - S_t, 0) + self.r_min

    def price(self):
        S, T, X = self.S, self.T, self.X
        params = self.params

        K = self.r_ko - self.r_min

        # adjust barrier to discrete monitoring
        sgn = 1 if self.is_call else -1
        beta = - zeta(0.5) / np.sqrt(2 * np.pi)
        H = self.H * np.exp(sgn * beta * self.sigma * np.sqrt(self.dt))

        bin_bar = BinaryBarrier(S, T, params, H, K, up_in = self.is_call)
        std_bar = StandardBarrier(S, T, params, X, H, up_out = self.is_call)

        return self.pt * std_bar.price() + bin_bar.price() + self.r_min * np.exp(-params.r*T)
    