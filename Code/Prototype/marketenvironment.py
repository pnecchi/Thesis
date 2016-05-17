################################################################################
# Description: Financial market environment for the automatic asset allocation
#              task. It is based on the PyBrain architecture.
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        sab 14 mag 2016 19:21:10 CEST
################################################################################

import numpy as np
import pandas as pd
from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment


class MarketEnvironment(Environment, Named):
    """ Financial market environent for the automatic asset allocation problem.
        The market consists of I+1 assets that can be traded only at discrete
        times. The state of the system consists of the assets past returns and.
        The actions that the learning agent can take are the porfolio
        allocations in the different assets.
    """

    # Current time step
    currentTimeStep = 0

    # Total number of time steps
    nSamples = 0

    # Risk-free rate
    X = 0.0

    # Number of past returns considered by the trading system
    P = 0

    def __init__(self, inputFile, X=0., P=0):
        """ Initialize market environment from inputFile containing the time
            series of the asses returns. The initial capital is assumed to be
            entirely invested in the risk-free asset.

        Args:
            inputFile (str): the path to the .csv file containing market data
            X (double): daily risk-free interest rate
            P (int): the number of past time steps in the system state
        """
        # Read past asset returns from inputFile into a pandas DataFrame
        marketData = pd.read_csv(inputFile, index_col=0)
        N, I = marketData.shape

        # Add column for risk-free asset
        rfData = pd.DataFrame(data=X, index=marketData.index, columns=['RF'])
        self.data = rfData.join(marketData, how='left')

        # Size of action space: number of tradable assets
        self.indim = I + 1

        # Size of state space: number of tradable assets + current allocations
        self.outdim = (P + 1) * (I + 1)

        # Initialize other variables
        self.currentTimeStep = P
        self.nSamples = N
        self.X = X
        self.P = P

    def getSensors(self):
        """ Retrieve the current state of the market, i.e. the last P+1 returns
            for each asset.

        Returns:
            state (np.array): the system observable state
        """
        t = self.currentTimeStep

        # Extract past returns from dataset and flatten into numpy array
        # TODO: It is useless to use the risk-free rate multiple times
        pastReturns = self.data.iloc[t-self.P:t, :].values.flatten()
        return pastReturns

    def getAssetReturns(self):
        """ Retrieve current time step asset returns.

        Returns:
            returns (np.array): the current time step returns
        """
        return self.data.iloc[self.currentTimeStep, :].values

    def performAction(self, action):
        """ Perform an action on the market, i.e. specify a new allocation for
            the current time interval. We assume that the allocation does not
            influence the dynamics of asset returns.

        Args:
            action (np.array): the new portfolio allocation
        """
        # Increment time step
        self.currentTimeStep += 1

    def reset(self):
        """ Reset market environment to initial time step and reset allocation
        """
        self.currentTimeStep = self.P

    def getDate(self):
        """ Return current market date.

        Returns:
            currentDate (date): current market date
        """
        currentDate = self.data.iloc[self.currentTimeStep].name
        return currentDate

