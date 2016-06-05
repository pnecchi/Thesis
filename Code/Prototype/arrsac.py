################################################################################
# Description: Average Reward Risk-Sensitive Actor-Critic Algorithm
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        dom 05 giu 2016 09:48:36 CEST
################################################################################

import numpy as np


class ARRSAC(object):
    """ ARRSAC is a risk-sensitive actor-critic algorithm proposed by Prashanth
        L A and M. Ghavamzadeh in "Variance-Constrained Actor-Critic Algorithms
        for Discounted and Average Reward MDPs". It is a two time-scale
        stochastic optimization problem that maximizes the Sharpe ratio (SR) in
        the average setting.
    """

    # Learning rate
    alphaMoment = 0.01  # Slowest time-scale, for 1st and 2nd moment estimate
    alphaCritic = 0.05  # Middle time-scale, for critics updates
    alphaActor = 0.1    # Fast time-scale, for actor updates

    def __init__(self, actor, criticV, criticU):
        """ Initialize ARRSAC agent.

        Args:
            actor (object): actor, i.e. parametrized policy
            criticV (object): critic for the average adjusted value function
            criticU (object): critic for the average adjusted square value function
        """
        # Initialize actor and critics
        self.actor = actor
        self.criticV = criticV
        self.criticU = criticU

        # Initialize reward cache variable
        self.reward = None

        # Initialize first and second moment of the reward
        self.averageReward = None
        self.averageSquareReward = None

    def getAction(self, obs):
        """ Select action given a state observation.

        Args:
            obs (np.array): observation of the system

        Returns:
            action (np.array): action
        """
        return self.actor.activate(obs)

    def giveReward(self, r):
        """ Receive a reward

        Args:
            r (double): reward
        """
        self.reward = r

    def learn(self):
        """ One learning step of the algorithm """



