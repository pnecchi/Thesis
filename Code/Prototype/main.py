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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

def main():
    """ Main program for automatic asset allocation problem.
    """
    # Directories
    input_data_dir = '../../Data/Input/'
    output_data_dir = '../../Data/Output/'

    # Experiment parameters
    batch = 1                     # Number of samples per learning step
    prnts = 100                    # Learning steps before printing results
    nEpisodes = 1000/batch/prnts     # Number of rollouts
    nExperiments = 1              # Number of experiments
    et = ExTools(batch, prnts)    # Tool for printing and plotting

    # Paramenters
    X = 0.02 / 252    # Daily risk-free rate
    deltaP = 0.0005   # Proportional transaction costs
    deltaF = 0.0      # Fixed transaction costs
    deltaS = 0.0      # Short-selling borrowing costs
    P = 5             # Number of past days the agent considers
    discount = None   # Discount factor
    T = 100           # Receding horizon for episodic task

    for exp in xrange(nExperiments):
        # Initialize the market environment
        market = MarketEnvironment(input_data_dir + 'daily_returns.csv', X, P)

        # Initialize the asset allocation task
        task = AssetAllocationTask(market, deltaP, deltaF, deltaS, discount, T)

        # Initialize controller module
        module = buildNetwork(market.outdim, market.indim, outclass=SoftmaxLayer)

        # Initialize learner module
        learner = PGPE(storeAllEvaluations=True,
                       learningRate=0.01,
                       sigmaLearningRate=0.01,
                       # momentum=0.0,
                       # epsilon=6.0,
                       rprop=False)

        # Initialize learning agent
        agent = OptimizationAgent(module, learner)
        et.agent = agent

        # Initialize experiment
        experiment = EpisodicExperiment(task, agent)

        # Training
        for episode in xrange(nEpisodes):
            for i in xrange(prnts):
                experiment.doEpisodes(batch)
            et.printResults((agent.learner._allEvaluations)[-50:-1], exp, episode)
        et.addExps()

        # Test
        taskTest = AssetAllocationTask(market, deltaP, deltaF, deltaS, discount, T, True)
        experimentTest = EpisodicExperiment(taskTest, agent)
        for episodes in xrange(10):
            experimentTest.doEpisodes(batch)

        # Performance
        taskTest.report.iloc[:, :-1].plot.area(title='Portfolio Allocation - PGPE')
        plt.ylim(0.0, 1.0)
        plt.xlabel('Date')
        plt.ylabel('Portfolio Allocation')
        plt.show()

        ptfCumLogReturns = taskTest.report['ptfLogReturn'].cumsum(axis=0)
        ptfCumLogReturns.plot(title='Portfolio Cumulative Log-Returns - PGPE', lw=1.5)
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.show()

    # Show results
    et.showExps()


if __name__ == "__main__":
    main()
