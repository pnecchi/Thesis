
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class CointegratedSeriesGenerator(object):
    """ """

    # Data directory
    DATA_DIR = '~/Documents/University/6_Anno_Poli/7_Thesis/Data/Input/'

    def __init__(self):
        """Constructor. """

        # Continuous model parameters
        self.dt          = 1.0 / 250  # daily sampling frequence
        self.S_0_1       = 100.0      # Initial price of first stock
        self.S_0_2       = 100.0      # Initial price of second stock
        self.gamma_0     = 0.0        # Initial spread
        self.sigma_1     = 0.3        # First stock volatility
        self.sigma_2     = 0.05        # Noise volatility
        self.sigma_gamma = 0.15        # Spread volatility
        self.theta       = 0.05       # Mean-reversion rate

        # Discrete model parameters
        self.drift_1 = - 0.5 * self.sigma_1**2 * self.dt
        self.vol_1   = self.sigma_1 * np.sqrt(self.dt)
        self.drift_gamma = np.exp(-self.theta * self.dt)
        self.vol_gamma = self.sigma_gamma * np.sqrt((1 - np.exp(- 2.0 * self.theta * self.dt)) / (2.0 * self.theta))
        self.vol_2   = np.sqrt(self.dt) * self.sigma_2

        # Vector of simulated values
        self.log_S_t_1 = None
        self.log_S_t_2 = None
        self.S_t_1   = None
        self.S_t_2   = None
        self.gamma_t = None

        # Dataframe
        self.df = None

    def run(self, n_steps):
        """Simulate the model.

        Parameters
        ----------
            n_steps: int
                Number of steps to simulate
        """

        # Initialize vectors
        self.log_S_t_1 = np.zeros(n_steps + 1)
        self.log_S_t_2 = np.zeros(n_steps + 1)
        self.gamma_t   = np.zeros(n_steps + 1)
        self.log_S_t_1[0] = np.log(self.S_0_1)
        self.log_S_t_2[0] = np.log(self.S_0_2)
        self.gamma_t[0]   = self.gamma_0

        # Simulate random variables
        #dW = np.random.randn(n_steps)
        #delta = self.drift_1 + self.vol_1 * dW
        #delta = np.cumsum(delta)
        #self.S_t_1 = self.S_0_1 * np.ones(n_steps + 1)
        #self.S_t_1[1:] *= np.exp(delta)


        self.log_S_t_1[1:] = self.drift_1 + self.vol_1 * np.random.randn(n_steps)
        self.gamma_t[1:]   = self.vol_gamma * np.random.randn(n_steps)

        # Cumulate
        self.log_S_t_1 = np.cumsum(self.log_S_t_1)
        for i in xrange(len(self.gamma_t) - 1):
            self.gamma_t[i+1] += self.drift_gamma * self.gamma_t[i]

        dW_2 = np.cumsum(np.random.randn(n_steps))
        self.log_S_t_2[1:] = self.log_S_t_1[1:] + self.gamma_t[1:] + self.vol_2 * dW_2

        # Compute prices
        self.S_t_1 = np.exp(self.log_S_t_1)
        self.S_t_2 = np.exp(self.log_S_t_2)

        # Combine vectors in Dataframe
        self.df = pd.DataFrame.from_dict({'ASSET_1': self.S_t_1, 'ASSET_2': self.S_t_2})

        # Compute daily returns
        self.df = ((self.df / self.df.shift(1)) - 1).ix[1:, :]

    def to_csv(self, filename='cointegrated.csv'):
        """Dumps data in .csv file in the data directory.

        Parameters
        ----------
            filename: str
                Output file name [default: historical.csv]
        """
        with open(os.path.expanduser(self.DATA_DIR + filename), 'w') as f:
            a = csv.writer(f, delimiter=',', lineterminator='\n')
            a.writerows([[self.df.shape[0], self.df.shape[1]]])
            self.df.to_csv(f, index=False)


if __name__ == "__main__":
    n_steps = 10000
    generator = CointegratedSeriesGenerator()
    generator.run(n_steps)
    generator.to_csv()
