################################################################################
# Description: Main file for automatic asset allocation problem
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        dom 15 mag 2016 11:43:34 CEST
################################################################################


from pybrain.tools.example_tools import ExTools
from marketenvironment import MarketEnvironment
from assetallocationtask import AssetAllocationTask
from pybrain.optimization.finitedifference.pgpe import PGPE
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules.softmax import SoftmaxLayer
from pybrain.rl.agents.optimization import OptimizationAgent
from pybrain.rl.experiments.episodic import EpisodicExperiment
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
    batch = 1                      # Number of samples per learning step
    prnts = 100                    # Learning steps before printing results
    nEpisodes = 3000/batch/prnts   # Number of rollouts
    nExperiments = 1               # Number of experiments
    et = ExTools(batch, prnts)     # Tool for printing and plotting

    # Paramenters
    X = 0.05 / 252    # Daily risk-free rate
    deltaP = 0.001   # Proportional transaction costs
    deltaF = 0.0      # Fixed transaction costs
    deltaS = 0.0      # Short-selling borrowing costs
    P = 5            # Number of past days the agent considers
    discount = 0.95   # Discount factor

    # Evaluation interval sizes
    start = P + 1
    trainingIntervalLength = 70
    testIntervalLength = 30

    # Initialize the market environment
    market = MarketEnvironment(input_data_dir + 'daily_returns.csv', X, P)
    nSamples = len(market.data)
    nPeriods = (nSamples - start + 1) / (trainingIntervalLength + testIntervalLength)

    # Initialize the asset allocation tasks
    task = AssetAllocationTask(market, deltaP, deltaF, deltaS, discount)

    # Initialize controller module
    module = buildNetwork(market.outdim,  # Input layer
                          market.indim,   # Output layer
                          outclass=SoftmaxLayer)  # Output activation function

    # Initialize learner module
    learner = PGPE(storeAllEvaluations=True,
                   learningRate=0.05,
                   sigmaLearningRate=0.025,
                   batchSize=batch,
                   # momentum=0.05,
                   # epsilon=6.0,
                   rprop=False)

    # Initialize learning agent
    agent = OptimizationAgent(module, learner)
    et.agent = agent

    for period in xrange(5):  #  nPeriods):

        # Set initial and final time steps for training
        initialTimeStep = start
        finalTimeStep = start + trainingIntervalLength
        task.setEvaluationInterval(initialTimeStep, finalTimeStep)
        task.trainingMode()

        # Initialize experiment
        experiment = EpisodicExperiment(task, agent)

        # Train the agent
        for episode in xrange(nEpisodes):
            for i in xrange(prnts):
                experiment.doEpisodes(batch)
            et.printResults((agent.learner._allEvaluations)[-50:-1],
                            1, episode)

        # Set initial and final time steps for training
        initialTimeStep = start + trainingIntervalLength
        finalTimeStep = initialTimeStep + testIntervalLength
        task.setEvaluationInterval(initialTimeStep, finalTimeStep)
        task.backtestMode()

        # Initialize experiment
        experiment = EpisodicExperiment(task, agent)

        # Test the agent
        experiment.doEpisodes(batch)

        # Slide evaluation window
        start += testIntervalLength

    # Print allocations
    task.report.iloc[:, :-1].plot.area(title='Portfolio Allocation - PGPE')
    plt.ylim(0.0, 1.0)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Allocation')
    plt.show()

    # Print cumulative log-returns
    buyHold = market.data.ix[task.report.index, 'SPY']
    buyHoldCumLogReturns = np.log(buyHold + 1.0).cumsum(axis=0)
    ptfCumLogReturns = task.report['ptfLogReturn'].cumsum(axis=0)
    cumLogReturns = pd.DataFrame(index=task.report.index)
    cumLogReturns['Buy & Hold'] = buyHoldCumLogReturns
    cumLogReturns['PGPE'] = ptfCumLogReturns
    cumLogReturns.plot(title='Cumulative Log-Returns - PGPE',
                       lw=2, grid=True)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Log-Returns')
    plt.show()

if __name__ == "__main__":
    main()
