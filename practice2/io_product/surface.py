import numpy as np
import pandas as pd

from pricer.pricer import get_pricing_results

class Surface():
    def __init__(self, S_range: np.ndarray, T_range: np.ndarray, option, price_only = False):
        self._S_range = S_range
        self._T_range = T_range
        self._option = option
        self._price_only = price_only

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
                self._option.S = self._S_range[i]
                self._option.T = self._T_range[j]

                results = get_pricing_results(self._option, price_only = self._price_only)

                self._price[i][j] = results[0]

                if not self._price_only:
                    self._delta[i][j] = results[1]
                    self._gamma[i][j] = results[2]
                    self._vega[i][j] = results[3]

        self._generated = True

    # return the dataframe
    def table(self, type):
        df = pd.DataFrame(
            self._type_map[type].T,
            index = self._T_range,
            columns = self._S_range
        )

        # df.to_csv(f'./output/{self._option.name} {type}.csv')
        # print(df)
        return df

    # plot the graph, return nothing
    def plot(self, ax, type):
        if not self._generated:
            self.generate()

        ax.plot_surface(
            self._X, self._Y, self._type_map[type],
            cmap='viridis'
        )
        
        ax.set_title(f'{self._option.name} {type}')
        ax.set_xlabel("Spot Price")
        ax.set_ylabel("Time to Maturity")
