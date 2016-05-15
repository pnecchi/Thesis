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


def main():
    """ Main program for automatic asset allocation problem.
    """
    # Directories
    input_data_dir = '../../Data/Input/'
    output_data_dir = '../../Data/Output/'

    # Experiment parameters
    batch = 1                   # Number of samples per learning step
    prnts = 10                  # Learning steps before printing results
    epis = 1000/batch/prnts     # Number of rollouts
    numbExp = 10                # Number of experiments
    et = ExTools(batch, prnts)  # Tool for printing and plotting

    # Paramenters
    X = 0.0 / 252    # Daily risk-free rate
    deltaP = 0.00001 # Proportional transaction costs
    deltaF = 0.0     # Fixed transaction costs
    deltaS = 0.00001 # Short-selling borrowing costs
    P = 5            # Number of past days the agent considers
    discount = 0.99  # Discount factor
    T = 10           # Receding horizon for episodic task

    for runs in xrange(numbExp):
        # Initialize the market environment
        market = MarketEnvironment(input_data_dir + 'daily_returns.csv', X, P)

        # Initialize the asset allocation task
        task = AssetAllocationTask(market, deltaP, deltaF, deltaS, discount, T)

        # Initialize controller module
        module = buildNetwork(market.outdim, 1, task.indim, outclass=SoftmaxLayer)

        # Initialize learner module
        learner = PGPE(storeAllEvaluations = True)

        # Initialize learning agent
        agent = OptimizationAgent(module, learner)
        et.agent = agent

        # Initialize experiment
        experiment = EpisodicExperiment(task, agent)

        # Do the experiment
        for updates in xrange(epis):
            for i in xrange(prnts):
                experiment.doEpisodes(batch)
            et.printResults((agent.learner._allEvaluations)[-50:-1], runs, updates)
        et.addExps()
    et.showExps()


if __name__ == "__main__":
    main()
