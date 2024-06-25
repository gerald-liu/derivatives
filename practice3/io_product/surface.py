import numpy as np
import pandas as pd

from product.option import Option
from pricer.pricer import get_pricing_results

# asset = portfolio or individual product
class Surface():
    def __init__(self, S_range: np.ndarray, T_range: np.ndarray, asset: Option, greeks=False):
        self._S_range = S_range
        self._T_range = T_range
        self._asset = asset
        self._greeks = greeks

        self._X = None
        self._Y = None
        self._price = np.empty((S_range.shape[0], T_range.shape[0]))
        self._delta = np.empty((S_range.shape[0], T_range.shape[0]))
        self._gamma = np.empty((S_range.shape[0], T_range.shape[0]))
        self._vega = np.empty((S_range.shape[0], T_range.shape[0]))

        self._generated = False
        self._type_map = {
            'price': self._price,
            'delta': self._delta,
            'gamma': self._gamma,
            'vega': self._vega
        }
    
    # generate points for plotting
    def generate(self):
        # X.shape = (len_S_range, len_T_range)
        self._X, self._Y = np.meshgrid(self._S_range, self._T_range, indexing='ij')
        
        for i in range(self._S_range.shape[0]):
            for j in range(self._T_range.shape[0]):
                self._asset.S = self._S_range[i]
                self._asset.T = self._T_range[j]
                asset = self._asset.copy(S=self._S_range[i], T=self._T_range[j])

                results = get_pricing_results(asset, greeks=self._greeks)

                self._price[i][j] = results[0]

                if self._greeks:
                    self._delta[i][j] = results[1]
                    self._gamma[i][j] = results[2]
                    self._vega[i][j] = results[3]

        self._generated = True

    # return the dataframe
    def table(self, type):
        if not self._generated:
            self.generate()
        
        df = pd.DataFrame(
            self._type_map[type].T,
            index = self._T_range,
            columns = self._S_range
        )

        return df

    # plot the graph, return nothing
    def plot(self, ax, type):
        if not self._generated:
            self.generate()

        ax.plot_surface(
            self._X, self._Y, self._type_map[type],
            cmap='viridis'
        )
        
        ax.set_title(f'{type}')
        ax.set_xlabel("Spot Price")
        ax.set_ylabel("Time to Maturity")
