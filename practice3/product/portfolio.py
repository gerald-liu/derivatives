import numpy as np

class Portfolio():
    def __init__(self, products=[], weights=[]):
        self._products = products
        self._weights = weights
        self._product_prices = []
        self._price = 0
        self._T = 0
    
    def add_product(self, product, weight):
        self._products.append(product)
        self._weights.append(weight)

    def set_weights(self):
        length = len(self._products)
        weights = np.array(self._weights)
        if length != 0 and weights.sum() == 0: # weights unspecified, use equal weights
            self._weights = [1/length for _ in range(length)]
        else:
            self._weights = (weights / weights.sum()).tolist()

    def reset(self):
        self._products = []
        self._weights = []

    def size(self):
        return len(self._products)
    
    def is_empty(self):
        return self.size() == 0
    
    def get_products(self):
        return self._products
    
    def price(self):
        self.set_weights()

        for p in self._products:
            self._product_prices.append(p.price())

        weighted_prices = np.array(self._weights) * np.array(self._product_prices)
        return weighted_prices.sum()
    
    def copy(self, S=None, T=None, dS=0, dT=0, dsigma=0):
        new_products = []
        for p in self._products:
            new_products.append(p.copy(S, T, dS, dT, dsigma))
        return Portfolio(products=new_products)
    
    def check_maturity(self):
        pass
