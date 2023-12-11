class Params():
    def __init__(self, volatility, rate, dividend) -> None:
        # sigma
        self._volatility = volatility
        # r
        self._rate = rate
        # q
        self._dividend = dividend

    # Getter methods
    @property
    def volatility(self):
        return self._volatility

    @property
    def rate(self):
        return self._rate

    @property
    def dividend(self):
        return self._dividend

    # Setter methods
    @volatility.setter
    def volatility(self, volatility):
        self._volatility = volatility

    @rate.setter
    def rate(self, rate):
        self._rate = rate

    @dividend.setter
    def dividend(self, dividend):
        self._dividend = dividend


    def _get_params(self):
        return (self.volatility, self.rate, self.dividend)
