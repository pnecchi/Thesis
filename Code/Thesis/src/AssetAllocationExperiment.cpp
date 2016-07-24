#include "thesis/AssetAllocationExperiment.h"
#include <fstream>

AssetAllocationExperiment::AssetAllocationExperiment(AssetAllocationTask const &task_,
                                                     Agent const &agent_,
                                                     size_t numExperiments_,
                                                     size_t numEpochs_,
                                                     size_t numTrainingSteps_,
                                                     size_t numTestSteps_,
                                                     std::string outputDir_,
                                                     std::string debugDir_)
    : task(task_),
      agentPtr(agent_.clone()),
      numExperiments(numExperiments_),
      numEpochs(numEpochs_),
      numTrainingSteps(numTrainingSteps_),
      numTestSteps(numTestSteps_),
      blog(task.getDimAction(), task.getDimAction(), numTestSteps),
      observationCache(task.getDimObservation()),
      actionCache(task.getDimAction()),
      rewardCache(0.0),
      outputDir(outputDir_),
      debugDir(debugDir_)
{
    /* Nothing to do */
}

void AssetAllocationExperiment::oneInteraction()
{
    // 1) Get observation
    agentPtr->receiveObservation(task.getObservation());

    // 2) Perform action
    actionCache = agentPtr->getAction();
    task.performAction(actionCache);

    // 3) Receive reward
    rewardCache = task.getReward();
    agentPtr->receiveReward(rewardCache);

    // 4) Receive next observation
    observationCache = task.getObservation();
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
            task.reset();
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
            if ((epoch + 1) % static_cast<int>(numEpochs / 20) == 0)
            {
                std::vector<std::vector<double>> stats = experimentStats.getStatistics();
                std::cout << "Experiment #" << exp
                          << " - Epoch #" << epoch + 1
                          << " - Average: " << stats[0][0]
                          << " - Standard Deviation: " << stats[0][1]
                          << " - Sharpe Ratio: " << stats[0][2] << std::endl;

                debugFile << epoch + 1 << "," << stats[0][0] << "," << stats[0][1]
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
                observationCache.rows(observationCache.size() - 2 * task.getDimAction(),
                                      observationCache.size() - task.getDimAction() - 1);
            blog.insertRecord(stateCache, actionCache, rewardCache);
        }

        std::ostringstream stringStreamBacktest;
        stringStreamBacktest << outputDir << "experiment" << exp << ".csv";
        blog.save(stringStreamBacktest.str());
    }
}
