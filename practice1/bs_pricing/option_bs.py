import numpy as np


class OptionBS():
    def __init__(self, spot, T, params):
        # S
        self._spot = spot
        # time to maturity
        self._T = T
        # (volatility, rate, dividend) = (sigma, r, q)
        self._params = params
    
    # Getter methods
    @property
    def spot(self):
        return self._spot

    @property
    def T(self):
        return self._T

    @property
    def params(self):
        return self._params

    # Setter methods
    @spot.setter
    def spot(self, spot):
        self._spot = spot

    @T.setter
    def T(self, T):
        self._T = T

    @params.setter
    def params(self, params):
        self._params = params

    # discount factor
    def _Z(self, discount_rate):
        return np.exp(- discount_rate * self.T)

    # d1 and d2
    def _d(self, number, strike):
        if number == 1:
            sgn = 1
        elif number == 2:
            sgn = -1
        else:
            raise Exception('Error: Must be d1 or d2.')
        
        S = self.spot
        K = strike
        T = self.T
        sigma, r, q = self.params._get_params()

        return (np.log(S/K) + (r-q + sgn*sigma**2 / 2)*T) / (sigma * np.sqrt(T))
    
    def set_S_T(self, spot_price, time_to_maturity):
        self.spot = spot_price
        self.T = time_to_maturity
