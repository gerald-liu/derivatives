class Params():
    def __init__(self, volatility, rate, coc) -> None:
        # sigma
        self._sigma = volatility
        # r
        self._r = rate
        # coc
        self._coc = coc

    @property
    def sigma(self):
        return self._sigma

    @property
    def r(self):
        return self._r

    @property
    def coc(self):
        return self._coc

    @property
    def q(self):
        return self._r - self._coc

    def get_params(self):
        return (self.sigma, self.r, self.coc)
