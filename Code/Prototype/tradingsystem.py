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
from pybrain.rl.agents.optimization import OptimizationAgent

# TODO feature extraction, e.g. deep auto-encoder?


class TradingSystem(OptimizationAgent):
    """ Trading system that interacts with the financial market. It takes care
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

    # Flag for backtesting: False for training mode, True for test mode
    backtest = None

    # If backtest is True, the trading system stores all allocations and returns
    report = None

    def __init__(self, module, learner, backtest=False, report=None):
        """ Initialize the trading system.

        Args:
            module (Module): controller that maps observations to allocations
            learner (Learner): learner that updates the controller parameters
            backtest (bool): flag for training mode or test mode
            report (pd.Dataframe): empty pd dataframe with correct column names
        """
        self.module = module
        self.learner = learner
        self.backtest = backtest
        self.report = report

    def integrateObservation(self, obs):
        """ Cache the last observation in a temporary variable until action is
        called and reward is given.

        Args:
            obs (np.array): system observation
        """
        self.lastObservation = obs

    def getAction(self):
        """ Select the action, i.e. the new allocation, given the last
            observation. It makes sure that the observation has been received
            before selecting an action.

        Returns:
            action (np.array): new asset allocation
        """
        # Select new allocation given last observation
        # TODO: We can use some features extracted from the lastObservation
        #       for instance using a Deep Auto-Encoder
        self.lastAction = self.module.activate(self.lastObservation)

        return self.lastAction

    def getReward(self, r):
        """ Receive and cache reward received from the market. If the trading
            system is in backtest mode, cache the allocation and the reward in
            the report for later performance evaluations.

        Args:
            r (double): return received by the trading system.
        """
        self.lastReward = r
        print self.backtest
        if self.backtest:
            print self.lastAction, self.lastReward
            self.report.loc[-1] = np.concatenate(self.lastAction,
                                                 self.lastReward)

#    def reset(self):
#        self.lastObservation = None
#        self.lastAction = None
#        self.lastReward = None

    def tradingPerformance(self):
        """ Compute the performance of the trading system in test mode.

        Returns:
            performanceStats (ffn.core.GroupStats): trading performance
        """
        ptfCumLogReturns = self.report['ptfLogReturn'].cumsum(axis=0)
        ptfValues = 100.0 * np.exp(ptfCumLogReturns)
        # performanceStats = ffn.core.GroupStats(ptfValues)
        # return performanceStats
