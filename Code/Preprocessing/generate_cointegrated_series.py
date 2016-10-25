
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')

class CointegratedSeriesGenerator(object):
    """
    Class that simulates the paths for two risky assets with a mean-reverting
    spread process.
    """

    # Data directory
    DATA_DIR = '~/Documents/University/6_Anno_Poli/7_Thesis/Data/Input/'

    def __init__(self):
        """Constructor. """

        # Continuous model parameters
        self.dt          = 1.0 / 365  # daily sampling frequence
        self.S_0_1       = 100.0      # Initial price of first stock
        self.S_0_2       = 110.0      # Initial price of second stock
        self.gamma_0     = 0.0        # Initial spread
        self.sigma_1     = 0.20       # Volatility for the first stock
        self.sigma_2     = 0.15       # Volatility for the second stock
        self.sigma_gamma = 0.20       # Volatility for the spread
        self.theta       = 0.15       # Mean-reversion rate
        self.rho         = 0.8        # Correlation between W^1 and W^2

        # Discrete model parameters
        self.drift_1     = - 0.5 * self.sigma_1 * self.sigma_1 * self.dt
        self.drift_2     = - 0.5 * self.sigma_2 * self.sigma_2 * self.dt
        self.drift_gamma = np.exp(-self.theta * self.dt)
        self.vol_1       = self.sigma_1 * np.sqrt(self.dt)
        self.vol_2       = self.sigma_2 * np.sqrt(self.dt)
        self.vol_gamma   = self.sigma_gamma * np.sqrt((1 - np.exp(- 2.0 * self.theta * self.dt)) / (2.0 * self.theta))

        # Dataframe to store data (will be initialized later)
        self.df = None

    def run(self, n_steps):
        """Simulate the model.

        Parameters
        ----------
            n_steps: int
                Number of steps to simulate
        """

        # Initialize vectors
        log_S_t_1 = np.zeros(n_steps + 1)
        log_S_t_2 = np.zeros(n_steps + 1)
        gamma_t   = np.zeros(n_steps + 1)

        # Initial conditions
        log_S_t_1[0] = np.log(self.S_0_1)
        log_S_t_2[0] = np.log(self.S_0_2)
        gamma_t[0]   = self.gamma_0

        # Simulate white noises
        eps = np.random.randn(n_steps, 3)
        eps[:, 1] = self.rho * eps[:, 0] + np.sqrt(1.0 - self.rho * self.rho) * eps[:, 1]

        # Iterate
        for i in xrange(1, n_steps):
            gamma_t[i]   = self.drift_gamma * gamma_t[i-1] + self.vol_gamma * eps[i-1, 2]
            log_S_t_1[i] = log_S_t_1[i-1] + self.drift_1 + self.vol_1 * eps[i-1, 0]
            log_S_t_2[i] = log_S_t_2[i-1] + self.drift_2 + self.vol_2 * eps[i-1, 1] + gamma_t[i] - gamma_t[i-1]

        # Compute prices
        S_t_1 = np.exp(log_S_t_1)
        S_t_2 = np.exp(log_S_t_2)

        #dW = np.random.randn(n_steps)
        #delta = self.drift_1 + self.vol_1 * dW
        #delta = np.cumsum(delta)
        #self.S_t_1 = self.S_0_1 * np.ones(n_steps + 1)
        #self.S_t_1[1:] *= np.exp(delta)


        # self.log_S_t_1[1:] = self.drift_1 + self.vol_1 * np.random.randn(n_steps)
        # self.gamma_t[1:]   = self.vol_gamma * np.random.randn(n_steps)

        # # Cumulate
        # self.log_S_t_1 = np.cumsum(self.log_S_t_1)
        # for i in xrange(len(self.gamma_t) - 1):
            # self.gamma_t[i+1] += self.drift_gamma * self.gamma_t[i]

        # dW_2 = np.cumsum(np.random.randn(n_steps))
        # self.log_S_t_2[1:] = self.log_S_t_1[1:] + self.gamma_t[1:] + self.vol_2 * dW_2

        # # Compute prices
        # self.S_t_1 = np.exp(self.log_S_t_1)
        # self.S_t_2 = np.exp(self.log_S_t_2)

        # Combine vectors in Dataframe
        self.df = pd.DataFrame.from_dict({'ASSET_1': S_t_1, 'ASSET_2': S_t_2})

        plt.figure(figsize=(10, 5), facecolor='white')
        self.df.plot()
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Effect of Mean-Reverting Spread')
        plt.legend(['$S_t^1$', '$S_t^2$'])
        plt.grid(True)
        plt.show()
        plt.savefig(os.path.expanduser('~/Documents/University/6_Anno_Poli/7_Thesis/Images/Cointegrated_Series.eps'),
                    dpi=1000, format='eps', facecolor='white', figsize=(10, 5), bbox_inches='tight')

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
    n_steps = 20000
    generator = CointegratedSeriesGenerator()
    generator.run(n_steps)
    generator.to_csv()
