import numpy as np
from scipy.stats import norm
from .option_bs import OptionBS


class EuVanilla(OptionBS):
    def __init__(self, spot, T, strike, params, is_call):
        super().__init__(spot, T, params)
        self._strike = strike
        self._is_call = is_call
        self._sgn = 1 if is_call else -1
        self._name = 'European Vanilla'

    # Getter method
    @property
    def strike(self):
        return self._strike

    @property
    def is_call(self):
        return self._is_call
    
    @property
    def name(self):
        if self.is_call:
            return f'{self._name} Call'
        else:
            return f'{self._name} Put'

    # Setter method
    @strike.setter
    def strike(self, strike):
        self._strike = strike

    @is_call.setter
    def is_call(self, is_call):
        self._is_call = is_call
        self._sgn = 1 if is_call else -1

    def _d(self, number):
        return super()._d(number=number, strike=self.strike)

    # return = PV of expected gross return on S
    def _Nd_S(self):
        return norm.cdf(self._sgn * self._d(1))

    # cost = proba of ITM (S_T > K), i.e. paying the strike K
    def _Nd_K(self):
        return norm.cdf(self._sgn * self._d(2))


    # delta = PV of contract return from S, adjusted for dividends
    def delta(self):
        q = self.params.dividend

        return self._sgn * self._Z(q) * self._Nd_S()

    # Call = SN(d1)Z(q) - KN(d2)Z(r) = delta * S - KN(d2)Z(r)
    # Put  = KN(-d2)Z(r) - SN(-d1)Z(q) = delta * S + KN(d2)Z(r)
    def price(self):
        S = self.spot
        K = self.strike
        r = self.params.rate
        
        return_from_spot = S * self.delta()
        cost_of_strike = K * self._Z(r) * self._Nd_K()

        if self.is_call:
            return return_from_spot - cost_of_strike
        else:
            return return_from_spot + cost_of_strike
    

    def gamma(self):
        S = self.spot
        T = self.T
        sigma = self.params.volatility
        q = self.params.dividend

        return self._Z(q) * norm.pdf(self._d(1)) / (S * sigma * np.sqrt(T))
    

    def vega(self):
        S = self.spot
        T = self.T
        q = self.params.dividend

        return self._Z(q) * S * np.sqrt(T) * norm.pdf(self._d(1))
    

class Digital(OptionBS):
    def __init__(self, spot, T, strike, params, is_call):
        super().__init__(spot, T, params)
        self._strike = strike
        self._is_call = is_call
        self._sgn = 1 if is_call else -1
        self._name = 'Digital'
    
    # Getter method
    @property
    def strike(self):
        return self._strike

    @property
    def is_call(self):
        return self._is_call

    @property
    def name(self):
        if self.is_call:
            return f'{self._name} Call'
        else:
            return f'{self._name} Put'

    # Setter method
    @strike.setter
    def strike(self, strike):
        self._strike = strike
    
    @is_call.setter
    def is_call(self, is_call):
        self._is_call = is_call
        self._sgn = 1 if is_call else -1

    def _d(self, number):
        return super()._d(number=number, strike=self.strike)

    # cost = proba of ITM (S_T > K)
    def _Nd_K(self):
        return norm.cdf(self._sgn * self._d(2))


    # PV of expected value of exercise
    # shape is similar to vanilla delta
    def price(self):
        r = self.params.rate

        return self._Z(r) * self._Nd_K()

    # shape is similar to vanilla gamma
    def delta(self):
        S = self.spot
        T = self.T
        sigma = self.params.volatility
        r = self.params.rate

        return self._sgn * self._Z(r) * norm.pdf(self._d(2)) / (S * sigma * np.sqrt(T))
    

    def gamma(self):
        S = self.spot
        T = self.T
        sigma = self.params.volatility

        return self._sgn * - self.delta() * self._d(1) / (S * sigma * np.sqrt(T))
    

    def vega(self):
        sigma = self.params.volatility
        r = self.params.rate

        return self._sgn * - self._Z(r) * norm.pdf(self._d(2)) * self._d(1) / sigma
    

class Spread(OptionBS):
    def __init__(self, spot, T, K_lo, K_hi, params, is_call):
        super().__init__(spot, T, params)
        self._K_lo = K_lo
        self._K_hi = K_hi
        self._is_call = is_call
        self._sgn = 1 if is_call else -1
        self._name = 'Spread'

        self._option_lo = EuVanilla(spot, T, K_lo, params, is_call)
        self._option_hi = EuVanilla(spot, T, K_hi, params, is_call)


    @property
    def K_lo(self):
        return self._K_lo

    @property
    def K_hi(self):
        return self._K_hi
    
    @property
    def is_call(self):
        return self._is_call

    @property
    def name(self):
        if self.is_call:
            return f'Bull Call {self._name}'
        else:
            return f'Bear Put {self._name}'

    @K_lo.setter
    def K_lo(self, K_lo):
        self._K_lo = K_lo
        self._option_lo.strike = K_lo

    @K_hi.setter
    def K_hi(self, K_hi):
        self._K_hi = K_hi
        self._option_hi.strike = K_hi
    
    @is_call.setter
    def is_call(self, is_call):
        self._is_call = is_call
        self._option_lo.is_call = is_call
        self._option_hi.is_call = is_call
        self._sgn = 1 if is_call else -1

    def set_S_T(self, spot_price, time_to_maturity):
        self.spot = spot_price
        self.T = time_to_maturity
        self._option_lo.set_S_T(self.spot, self.T)
        self._option_hi.set_S_T(self.spot, self.T)
    
    def price(self):
        return self._sgn * (self._option_lo.price() - self._option_hi.price())
    
    def delta(self):
        return self._sgn * (self._option_lo.delta() - self._option_hi.delta())
    
    def gamma(self):
        return self._sgn * (self._option_lo.gamma() - self._option_hi.gamma())
    
    def vega(self):
        return self._sgn * (self._option_lo.vega() - self._option_hi.vega())
    