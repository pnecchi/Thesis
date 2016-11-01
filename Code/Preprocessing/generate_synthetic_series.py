################################################################################
# Description: Generator of synthetic price series according to Moody & Saffell
#              "Learning to trade via direct reinforcement" (2001)
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        ven 27 mag 2016 14:27:25 CEST
################################################################################

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')


class PriceGenerator(object):

    """ Synthetic price series generator used in  Moody & Saffell - Learning to
        trade via direct reinforcement (2001). We generate log price series as
        random walks with autoregressive trend processes that are designed to
        have tradeable structure. These syntethic series will be used to test
        various learning algorithms in a controlled experiment.
    """

    # Mean reversion speed
    alpha = 0.9

    # Price standard deviation
    sigma = 10.0

    def __init__(self):
        """ Initialize price generator. """
        pass

    def generateSeries(self, T):
        """ generate synthetic price series of length T.

        Args:
            T (int): price series length.

        Returns:
            price (np.array): price series
        """
        # Simulate gaussian noises
        epsilon = np.random.randn(T)
        nu = np.random.randn(T)

        # Initialize drift process
        beta = np.zeros(T+1)
        beta[0] = 0.0

        # Initialize logprice process
        p = np.zeros(T+1)
        p[0] = 1.0

        # Compute processes
        for t in range(1, T+1):
            p[t] = p[t-1] + beta[t-1] + self.sigma * epsilon[t-1]
            beta[t] = self.alpha * beta[t-1] + nu[t-1]

        # Rescale logprices
        p /= p.max() - p.min()

        # Compute prices
        price = np.exp(p)
        return price


def main():
    # Price series length
    T = 10000

    # Initialize generator
    generator = PriceGenerator()

    # Generate prices
    price = generator.generateSeries(T)

    # Store price series in pandas dataframe
    df = pd.DataFrame(price,
                      columns=['SYNT'])

    # Plot price seris
    df.plot(title='Synthetic Price Series', lw=2)
    plt.show()

    # Compute daily returns
    daily_returns = (df / df.shift(1)) - 1
    daily_returns = daily_returns.ix[1:]

    # Write df to csv file
    with open('synthetic_high_vol.csv', 'w') as f:
        a = csv.writer(f, delimiter=',', lineterminator='\n')
        a.writerows([[daily_returns.shape[0], daily_returns.shape[1]]])
        daily_returns.to_csv(f, index=False)


if __name__ == "__main__":
    main()
