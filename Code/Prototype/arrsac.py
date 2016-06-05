################################################################################
# Description: Average Reward Risk-Sensitive Actor-Critic Algorithm
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        dom 05 giu 2016 09:48:36 CEST
################################################################################

import numpy as np
from statistics import ExponentialMovingAverage


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
        self.averageReward = ExponentialMovingAverage(self.alphaMoment)
        self.averageSquareReward = ExponentialMovingAverage(self.alphaMoment)

    def getAction(self, state):
        """ Select action given a state.

        Args:
            state (np.array): state of the system

        Returns:
            action (np.array): action
        """
        self.action = self.actor.getAction(obs)
        return self.action

    def giveReward(self, r):
        """ Receive a reward

        Args:
            r (double): reward
        """
        self.reward = r

    def learn(self):
        """ One learning step of the algorithm """

        # 1) Update first and second moment estimates
        self.averageReward.update(self.reward)
        self.averageSquareReward.update(self.reward**2)

        # 2) Compute TD errors
        # TODO: How to pass the next state to the agent?
        delta = self.reward - self.averageReward.get() + \
            self.criticV(self.nextState) - self.criticV(self.state)
        epsilon = self.reward**2 - self.averageSquareReward.get() + \
            self.criticU(self.nextState) - self.criticU(self.state)

        # 3) Critic update
        newParamV = self.criticV.getParameters() + \
            self.alphaCritic * delta * self.criticV.gradient(self.state)
        self.criticV.setParameters(newParamV)
        newParamU = self.criticU.getParameters() + \
            self.alphaCritic * epsilon * self.criticU.gradient(self.state)
        self.criticU.setParameters(newParamU)

        # 4) Actor update
        likelihoodScore = self.actor.scoreFunction(self.state, self.action)
        variance = self.averageSquareReward.get() - self.averageReward.get()**2
        newParamActor = self.actor.getParameters() + \
            self.alphaActor / np.sqrt(variance) * (delta * likelihoodScore -
            self.averageReward.get() * (epsilon * likelihoodScore -
            2.0 * self.averageReward.get() * delta * likelihoodScore) / (2.0 * variance))
        # TODO: Manage projection
        self.actor.setParameters(newParamActor)
