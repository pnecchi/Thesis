################################################################################
# Description: TradingSystem class based on PyBrain LoggingAgent and
#              LearningAgent that implements a trading system that takes care of
#              feature extraction, selecting actions, learning, and reporting.
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        lun 16 mag 2016 23:18:37 CEST
################################################################################

# import ffn            # for trading performance computation
import pandas as pd
import numpy as np
from pybrain.rl.agents.agent import Agent

# TODO feature extraction, e.g. deep auto-encoder?


class TradingSystem(Agent):
    """ Trading system // that interacts with the financial market. It takes care
        of feature extraction, selecting allocations, learning from the
        interaction with the environment and reporting the strategy performance.
        Moreover, the trading system can be set in training mode, during which
        it uses rollouts from the system to optimize its controller parameters,
        and trading mode, during which it trades based on the learnt parameters.
    """

    # Cache tuple <S, A, R>
    lastObservation = None
    lastAction = None
    lastReward = None

    # Objective function estimate
    nRewards = 0
    averageReward = None

    # Flag for backtesting: False for training mode, True for backtest mode
    backtest = None

    def __init__(self, agent, backtest=False, history=None):
        """ Initialize the trading system.

        Args:
            agent (Agent): trading agent
            backtest (bool): flag for training mode or test mode
        """
        self.agent = agent
        self.backtest = backtest

        # History
        self.history = history

    def integrateObservation(self, obs):
        """ Cache the last observation in a temporary variable until action is
        called and reward is given.

        Args:
            obs (np.array): system observation
        """
        self.lastObservation = obs
        self.lastAction = None
        self.lastReward = None

    def getAction(self):
        """ Select the action, i.e. the new allocation, given the last
            observation. It makes sure that the observation has been received
            before selecting an action.

        Returns:
            action (np.array): new asset allocation
        """
        assert self.lastObservation is not None
        assert self.lastAction is None
        assert self.lastReward is None

        # Select new allocation given last observation
        self.lastAction = self.agent.getAction(self.lastObservation)
        return self.lastAction

    def giveReward(self, r):
        """ Receive and cache reward received from the market. If the trading
            system is in backtest mode, cache the allocation and the reward in
            the report for later performance evaluations.

        Args:
            r (double): return received by the trading system.
        """
        assert self.lastObservation is not None
        assert self.lastAction is not None
        assert self.lastReward is None

        # Cache reward
        self.lastReward = r
        self.nRewards += 1

        # Update objective function estimate
        if self.averageReward is None:
            self.averageReward = r
        else:
            nu = 1.0 / (self.nRewards + 1)
            self.averageReward = (1 - nu) * self.averageReward + nu * r

        # Broadcast reward to agent
        self.agent.giveReward(self.lastReward)

        # If backtesting, log allocation and reward for performance evaluation
        if self.backtest:
            entry = np.append(self.lastAction, self.lastReward)
            # currentDate = self.env.getDate()
            # TODO: How to pass currentDate to TradingSystem?
            self.history.loc[len(self.history.index)] = entry

    def learn(self):
        """ Let the agent perform a learning step.
        """
        self.agent.learn()

    def reset(self):
        self.lastObservation = None
        self.lastAction = None
        self.lastReward = None

    def tradingPerformance(self):
        """ Compute the performance of the trading system in test mode.

        Returns:
            performanceStats (ffn.core.GroupStats): trading performance
        """
