from copy import deepcopy
from pricer.mc_pricer import MonteCarloPricer as MCPricer

# method = {'closed-form', 'monte-carlo'}
# 'monte-carlo' args: [dt]
class Pricer():
    def __init__(self, option, method='closed-form', args=[]) -> None:
        self._option = option
        self._method = method
        self._args = args

    def price(self, dS=0, dsigma=0):
        if (not dS) or (not dsigma):
            option = deepcopy(self._option)
            option.S += dS
            option.sigma += dsigma
        else:
            option = self._option

        if self._method == 'closed-form':
            return option.price()
        elif self._method == 'monte-carlo':
            dt = self._args[0]
            mc_pricer = MCPricer(option.S, option.T, dt, option.params)
            return mc_pricer.price(option)
        else:
            raise Exception('Error: Pricing method undefined.')


class GreekCalculator():
    def __init__(self, pricer: Pricer) -> None:
        self._pricer = pricer

    # D = (P1 - P0) / dS
    def delta(self, dS):
        p_1 = self._pricer.price(dS = dS)
        p_2 = self._pricer.price(dS = -dS)

        return (p_1 - p_2) / (dS * 2)
    
    # D1 = (P1 - P0) / dS, D2 = (P0 - P2) / dS, where P2 < P0 < P1
    # G = (D1 - D2) / dS = (P1 - 2*P0 + P2) / (dS)^2
    def gamma(self, dS):
        p_0 = self._pricer.price()
        p_1 = self._pricer.price(dS = dS)
        p_2 = self._pricer.price(dS = -dS)

        return (p_1 - 2*p_0 + p_2) / dS**2

    # V = (P1 - P0) / dsigma
    def vega(self, dsigma):
        p_1 = self._pricer.price(dsigma = dsigma)
        p_2 = self._pricer.price(dsigma = -dsigma)

        return (p_1 - p_2) / (dsigma * 2)


def get_pricing_results(option, method='closed-form', args=[], price_only=False):
    pricer = Pricer(option, method=method, args=args)
    
    if price_only:
        return [pricer.price()]
    else:
        greek_calc = GreekCalculator(pricer)
        return [
            pricer.price(),
            greek_calc.delta(dS = 0.01),
            greek_calc.gamma(dS = 0.01),
            greek_calc.vega(dsigma = 0.01)
        ]
