################################################################################
# Description: Asset allocation task, built on the PyBrain architecture.
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        dom 15 mag 2016 14:44:13 CEST
################################################################################

import numpy as np
import pandas as pd

from pybrain.rl.environments.episodic import EpisodicTask
from tradingperformance import portfolioSimpleReturn


class AssetAllocationTask(EpisodicTask):
    """ The asset allocation task on a financial market. The agent can trade I+1
        different assets only at discrete times by specifying a portfolio
        allocation. The state that the agent observes is given by the past
        returns of the tradable assets augmented with the current allocation.
        The reward function is the portfolio log-return."""

    def __init__(self,
                 environment,
                 deltaP,
                 deltaF,
                 deltaS,
                 discount,
                 backtest=False):
        """ Standard constructor for the asset allocation task.

        Args:
            environment (Environment): market environment object
            deltaP (double): proportional transaction costs rate
            deltaF (double): fixed transaction cost rate
            deltaS (double): short selling borrowing cost rate
            discount (double): discount factor
            backtest (bool): flag for training mode or test mode
        """
        # Initialize episodic task
        EpisodicTask.__init__(self, environment)

        # Transaction costs
        self.deltaP = deltaP
        self.deltaF = deltaF
        self.deltaS = deltaS

        # Discount factor
        self.discount = discount

        # Backtesting
        self.backtest = backtest

        # Report stores allocations and portfolio log-returns for backtesting
        self.report = pd.DataFrame(columns=list(self.env.data.columns) +
                                           ['ptfLogReturn'])

        # Initialize allocation
        self.initializeAllocation()

    def getObservation(self):
        """ An augmented observation of the underlying environment state that
            also includes the current portfolio weights, right before
            realloacation.

        Returns:
            state (np.array): the augmented state (size (P+1) * (I+1))
        """
        # Observe past asset returns from the environment
        pastReturns = EpisodicTask.getObservation(self)

        # Return augmented state
        return np.concatenate((pastReturns, self.currentAllocation))

    def performAction(self, action):
        """ Perform action on the underlying environment, i.e specify new asset
        allocation.

        Args:
            action (np.array): new allocation
        """
        # Cache new asset allocation for computing rewards
        self.newAllocation = action
        # Perform action
        EpisodicTask.performAction(self, action)

    def getReward(self):
        """ Function that returns the portfolio simple returns associated with
            the specified allocation.

        Returns:
            ptfSimpleReturn (double): portfolio simple return
        """
        # Retrieve current time step asset returns
        assetReturns = self.env.getAssetReturns()

        # Compute portfolio return associated to new allocation
        ptfSimpleReturn = portfolioSimpleReturn(assetReturns,
                                                self.currentAllocation,
                                                self.newAllocation,
                                                self.deltaP,
                                                self.deltaF,
                                                self.deltaS)
        # Update allocation weights
        self.currentAllocation = self.newAllocation * \
            (1.0 + assetReturns) / (1.0 + ptfSimpleReturn)

        # Compute portfolio log-return
        ptfLogReturn = np.log(1.0 + ptfSimpleReturn)

        # Store allocation and return when backtesting
        if self.backtest:
            reportEntry = np.append(self.newAllocation, ptfLogReturn)
            currentDate = self.env.getDate()
            self.report.ix[currentDate, :] = reportEntry

        return ptfLogReturn


    def isFinished(self):
        """ Function that checks if the current episode is over. To define an
            episode, we consider a receding horizon of length T.

        Returns:
            over (bool): flag that indicates if the current episode is over
        """
        if self.env.currentTimeStep >= self.env.finalTimeStep:
            self.initializeAllocation()
            return True
        else:
            return False

    def initializeAllocation(self):
        """ Initialize portfolio allocation at the beginning of an episode
        """
        # Equally weighted portfolio
        # self.currentAllocation = np.ones(self.env.indim) / self.env.indim
        # Everything invested in the risk-free asset
        self.currentAllocation = np.zeros(self.env.indim)
        self.currentAllocation[0] = 1.0
        # Random initialization
        # temp = np.random.rand(self.env.indim)
        # self.currentAllocation = temp / np.sum(temp)

    def setEvaluationInterval(self, start, end):
        """ Set the time interval to be considered in the evaluation. This
        function is used to change the evaluation interval during the backtest
        procedure.

        Args:
            start (int): start time index
            end (int): end time index
        """
        self.env.setEvaluationInterval(start, end)

    def trainingMode(self):
        """ Set the task in training mode
        """
        self.backtest = False

    def backtestMode(self):
        """ Set the task in backtest mode
        """
        self.backtest = True
