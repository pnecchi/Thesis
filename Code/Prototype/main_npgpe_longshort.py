################################################################################
# Description: Main file for automatic asset allocation problem
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        dom 15 mag 2016 11:43:34 CEST
################################################################################


from marketenvironment import MarketEnvironment
from assetallocationtask import AssetAllocationTask
from discretecontroller import DiscreteController
from npgpe import NPGPE
from tradingsystem import TradingSystem
from pybrain.rl.experiments.continuous import ContinuousExperiment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('seaborn-pastel')

def main():
    """ Main program for automatic asset allocation problem.
    """
    # Directories
    input_data_dir = '../../Data/Input/'
    output_data_dir = '../../Data/Output/'

    # Experiment parameters
    batch = 1              # Number of samples per learning step
    nEpochs = 200          # Number of learning epochs
    nLearningSteps = 1000  # Number of learning steps per epoch
    nTestSteps = 200       # Number of test steps

    # Paramenters
    X = 0.05 / 252    # Daily risk-free rate
    deltaP = 0.001   # Proportional transaction costs
    deltaF = 0.0      # Fixed transaction costs
    deltaS = 0.000    # Short-selling borrowing costs
    P = 20            # Number of past days the agent considers
    discount = 0.95   # Discount factor

    # Initialize the market environment
    market = MarketEnvironment(input_data_dir + 'daily_returns.csv', X, P)
    # nSamples = len(market.data)
    # nPeriods = (nSamples - start + 1) / (trainingIntervalLength + testIntervalLength)

    # Initialize the asset allocation tasks
    task = AssetAllocationTask(market, deltaP, deltaF, deltaS, discount)

    # Initialize controller module
    controller = DiscreteController(task.outdim)

    # Initialize agent
    agent = NPGPE(controller)
    agent.alphaMu = 0.05 # 0.05
    agent.alphaC  = 0.025 # 0.025

    # History
    history = pd.DataFrame(columns=list(market.data.columns) + ['ptfLogReturn'])

    # Initialize trading system
    tradingSystem = TradingSystem(agent=agent, history=history)

    # Continuous experiment
    experiment = ContinuousExperiment(task, tradingSystem)

    # Set initial and final time steps for training
    initialTimeStep = P
    finalTimeStep = initialTimeStep + nLearningSteps
    task.setEvaluationInterval(initialTimeStep, finalTimeStep)

    # Learning
    averageReward = np.array([])
    for epoch in xrange(nEpochs):
        for step in xrange(nLearningSteps):
            experiment.doInteractionsAndLearn()
        averageReward = np.append(averageReward, tradingSystem.averageReward)
        print('Epoch ' + str(epoch) + ' - Average reward = ' + str(tradingSystem.averageReward))
        market.reset()

    # Objective function
    plt.plot(np.arange(nEpochs), averageReward, lw=2)
    plt.title('Learning Process')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.show()

    # Set initial and final time for backtesting
    initialTimeStep = finalTimeStep
    finalTimeStep = initialTimeStep + nTestSteps
    task.setEvaluationInterval(initialTimeStep, finalTimeStep)

    # Backtesting
    tradingSystem.backtest = True

    for step in xrange(nTestSteps):
        experiment.doInteractionsAndLearn()

    # Print allocations
    tradingSystem.history['JPM'].plot(title='Portfolio Allocation - PGPE', lw=2)
    plt.ylim(-1.2, 1.2)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Allocation')
    plt.show()

    # Print cumulative log-returns
    buyHold = market.data.ix[market.initialTimeStep+1:market.finalTimeStep+1, 'JPM']
    buyHoldCumLogReturns = np.log(buyHold + 1.0).cumsum(axis=0)

    ptfCumLogReturns = tradingSystem.history['ptfLogReturn'].cumsum(axis=0)
    ptfCumLogReturns.index = buyHoldCumLogReturns.index

    cumProfit = pd.DataFrame(index=buyHoldCumLogReturns.index)


    cumProfit['Buy & Hold'] = np.exp(buyHoldCumLogReturns)
    cumProfit['NPGPE'] = np.exp(ptfCumLogReturns.values.astype(float))
    cumProfit.plot(title='Cumulative Profits - NPGPE',
                   lw=2, grid=True)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Profits')
    plt.show()


if __name__ == "__main__":
    main()
