################################################################################
# Description: Asset allocation task, built on the PyBrain architecture.
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        dom 15 mag 2016 14:44:13 CEST
################################################################################

import numpy as np

from pybrain.rl.environments.episodic import EpisodicTask
from tradingperformance import portfolioSimpleReturn


class AssetAllocationTask(EpisodicTask):
    """ The asset allocation task on a financial market. The agent can trade I+1
        different assets only at discrete times by specifying a portfolio
        allocation. The state that the agent observes is given by the past
        returns of the tradable assets augmented with the current allocation.
        The reward function is the portfolio log-return."""

    # Episode time step
    t = 0

    # Episode receding horizon
    T = 0

    # Transaction costs
    deltaP = 0.0
    deltaF = 0.0
    deltaS = 0.0

    # Last time step asset returns
    assetReturns = None

    # Current portfolio allocation
    currentAllocation = None

    # New portfolio allocation
    newAllocation = None

    def __init__(self, environment, deltaP, deltaF, deltaS, discount, T):
        """ Standard constructor for the asset allocation task.

        Args:
            environment (Environment): market environment object
            deltaP (double): proportional transaction costs rate
            deltaF (double): fixed transaction cost rate
            deltaS (double): short selling borrowing cost rate
            discount (double): discount factor
            T (int): receding horizon for episodic task
        """
        # Initialize episodic task
        EpisodicTask.__init__(self, environment)
        self.T = T

        # Transaction costs
        self.deltaP = deltaP
        self.deltaF = deltaF
        self.deltaS = deltaS

        # Discount factor
        self.discount = discount

        # Initialize allocation
        self.currentAllocation = np.zeros(self.env.indim)
        self.currentAllocation[0] = 1.0

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
        # Increment episode time step
        self.t += 1

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
        self.assetReturns = self.env.getAssetReturns()

        # Compute portfolio return associated to new allocation
        ptfSimpleReturn = portfolioSimpleReturn(self.assetReturns,
                                                self.currentAllocation,
                                                self.newAllocation,
                                                self.deltaP,
                                                self.deltaF,
                                                self.deltaS)
        # Update allocation weights
        self.currentAllocation = self.newAllocation * \
            (1.0 + self.assetReturns) / (1.0 + ptfSimpleReturn)
        return np.log(1.0 + ptfSimpleReturn)

    def isFinished(self):
        """ Function that checks if the current episode is over. To define an
            episode, we consider a receding horizon of length T.

        Returns:
            over (bool): flag that indicates if the current episode is over
        """
        if self.t >= self.T or self.env.currentTimeStep >= self.env.nSamples:
            self.t = 0
            return True
        else:
            self.t += 1
            return False
