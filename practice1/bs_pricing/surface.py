import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Surface():
    def __init__(self, spot_range, T_range, option):
        self._spot_range = spot_range
        self._T_range = T_range
        self._option = option
        self._X = None
        self._Y = None
        self._price = np.empty((spot_range.shape[0], T_range.shape[0]))
        self._delta = np.empty((spot_range.shape[0], T_range.shape[0]))
        self._gamma = np.empty((spot_range.shape[0], T_range.shape[0]))
        self._vega = np.empty((spot_range.shape[0], T_range.shape[0]))
        self._generated = False
        self._type_map = {
            'price': self._price,
            'delta': self._delta,
            'gamma': self._gamma,
            'vega': self._vega
        }

    @property
    def spot_range(self):
        return self._spot_range

    @property
    def T_range(self):
        return self._T_range
    
    @property
    def option(self):
        return self._option
    
    # generate points for plotting
    def generate(self):
        # X.shape = (len_spot_range, len_T_range)
        self._X, self._Y = np.meshgrid(self.spot_range, self.T_range, indexing='ij')
        
        for i in range(self.spot_range.shape[0]):
            for j in range(self.T_range.shape[0]):
                S = self.spot_range[i]
                T = self.T_range[j]
                self.option.set_S_T(S, T)
                self._price[i][j] = self.option.price()
                self._delta[i][j] = self.option.delta()
                self._gamma[i][j] = self.option.gamma()
                self._vega[i][j] = self.option.vega()

        self._generated = True

    # return the dataframe
    def table(self, type):
        df = pd.DataFrame(
            self._type_map[type].T,
            index = self.T_range,
            columns = self.spot_range
        )

        # df.to_csv(f'./output/{self.option.name} {type}.csv')
        # print(df)
        return df

    # plot the graph, return nothing
    def plot(self, ax, type):
        ax.plot_surface(
            self._X, self._Y, self._type_map[type],
            cmap='viridis'
        )
        
        plot_title = f'{self.option.name} {type}'
        ax.set_title(plot_title)
        ax.set_xlabel("Spot Price")
        ax.set_ylabel("Time to Maturity")
