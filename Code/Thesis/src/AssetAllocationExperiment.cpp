#include "thesis/AssetAllocationExperiment.h"
#include <fstream>

AssetAllocationExperiment::AssetAllocationExperiment(AssetAllocationTask const &task_,
                                                     Agent const &agent_,
                                                     size_t const &numExperiments_,
                                                     size_t const &numEpochs_,
                                                     size_t const &numTrainingSteps_,
                                                     size_t const &numTestSteps_,
                                                     std::string const &outputDir_,
                                                     std::string const &debugDir_)
    : Experiment(task_, agent_),
      numExperiments(numExperiments_),
      numEpochs(numEpochs_),
      numTrainingSteps(numTrainingSteps_),
      numTestSteps(numTestSteps_),
      blog(taskPtr->getDimAction(), taskPtr->getDimAction(), numTestSteps),
      observationCache(taskPtr->getObservation()),
      actionCache(taskPtr->getDimAction()),
      rewardCache(0.0),
      outputDir(outputDir_),
      debugDir(debugDir_)
{
    /* Nothing to do */
}

AssetAllocationExperiment::AssetAllocationExperiment(AssetAllocationExperiment const &other_)
    : Experiment(*other_.taskPtr, *other_.agentPtr),
      numExperiments(other_.numExperiments),
      numEpochs(other_.numEpochs),
      numTrainingSteps(other_.numTrainingSteps),
      numTestSteps(other_.numTestSteps),
      blog(taskPtr->getDimAction(), taskPtr->getDimAction(), numTestSteps),
      observationCache(taskPtr->getObservation()),
      actionCache(taskPtr->getDimAction()),
      rewardCache(0.0),
      outputDir(other_.outputDir),
      debugDir(other_.debugDir)
{
    /* Nothing to do */
}

std::unique_ptr<Experiment> AssetAllocationExperiment::clone() const
{
    return std::unique_ptr<Experiment>(new AssetAllocationExperiment(*this));
}

void AssetAllocationExperiment::oneInteraction()
{
    // 1) Get observation
    agentPtr->receiveObservation(observationCache);

    // 2) Perform action
    actionCache = agentPtr->getAction();
    taskPtr->performAction(actionCache);

    // 3) Receive reward
    rewardCache = taskPtr->getReward();
    agentPtr->receiveReward(rewardCache);

    // 4) Receive next observation
    observationCache = taskPtr->getObservation();
    agentPtr->receiveNextObservation(observationCache);

    // 5) Dump results in statistics gatherer
    experimentStats.dumpOneResult(rewardCache);
}

void AssetAllocationExperiment::run()
{
    // Perform numExperiments independent experiments
    for (size_t exp = 0; exp < numExperiments; ++exp)
    {
        // Reset backtest log and agent
        agentPtr->reset();
        blog.reset();

        // Open debugging file
        std::ostringstream stringStream;
        stringStream << debugDir << "experiment" << exp << ".csv";
        std::ofstream debugFile;
        debugFile.open(stringStream.str());
        debugFile << "epoch,average,stdev,sharpe,\n";

        // Training
        for (size_t epoch = 0; epoch < numEpochs; ++epoch)
        {
            // Reset task
            taskPtr->reset();
            experimentStats.reset();

            // Signal to agent that a new epoch has started
            agentPtr->newEpoch();

            for (size_t step = 0; step < numTrainingSteps; ++step)
            {
                // Interaction between the task and the agent
                oneInteraction();

                // Learning step
                agentPtr->learn();
            }

            // Print convergence summary
            if (epoch % static_cast<int>(numEpochs / 50) == 0)
            {
                std::vector<std::vector<double>> stats = experimentStats.getStatistics();
                std::cout << "Experiment #" << exp
                          << " - Epoch #" << epoch
                          << " - Average: " << stats[0][0]
                          << " - Standard Deviation: " << stats[0][1]
                          << " - Sharpe Ratio: " << stats[0][2] << std::endl;

                debugFile << epoch << "," << stats[0][0] << "," << stats[0][1]
                          << "," << stats[0][2] << ",\n";
            }
        }
        debugFile.close();

        // Backtest
        for (size_t step = 0; step < numTestSteps; ++step)
        {
            // Interaction between the task and the agent
            oneInteraction();

            // Learning step
            agentPtr->learn();

            // Log (action, reward) tuple
            arma::vec stateCache =
                observationCache.rows(observationCache.size() - 2 * taskPtr->getDimAction(),
                                      observationCache.size() - taskPtr->getDimAction() - 1);
            blog.insertRecord(stateCache, actionCache, rewardCache);
        }

        std::ostringstream stringStreamBacktest;
        stringStreamBacktest << outputDir << "experiment" << exp << ".csv";
        blog.save(stringStreamBacktest.str());
    }
}
