#include "thesis/AssetAllocationExperiment.h"
#include <fstream>

AssetAllocationExperiment::AssetAllocationExperiment(AssetAllocationTask const &task_,
                                                     Agent const &agent_,
                                                     size_t numExperiments_,
                                                     size_t numEpochs_,
                                                     size_t numTrainingSteps_,
                                                     size_t numTestSteps_)
    : task(task_),
      agentPtr(agent_.clone()),
      numExperiments(numExperiments_),
      numEpochs(numEpochs_),
      numTrainingSteps(numTrainingSteps_),
      numTestSteps(numTestSteps_),
      blog(task.getDimAction(), numTestSteps),
      actionCache(task.getDimAction()),
      rewardCache(0.0)
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
    agentPtr->receiveNextObservation(task.getObservation());

    // 5) Backtest
    if (backtestMode)
        blog.insertRecord(actionCache, rewardCache);

    // 6) Dump results in statistics gatherer
    experimentStats.dumpOneResult(rewardCache);
}

void AssetAllocationExperiment::run()
{
    std::ofstream debugFile;
    debugFile.open("../../../Data/Debug/debugExperiment1.csv" );
    debugFile << "epoch,average,stdev,sharpe,\n";

    for (size_t exp = 0; exp < numExperiments; ++exp)
    {
        // Training
        for (size_t epoch = 0; epoch < numEpochs; ++epoch)
        {
            for (size_t step = 0; step < numTrainingSteps; ++step)
            {
                // Interaction between the task and the agent
                oneInteraction();

                // Learning step
                agentPtr->learn();
            }

            //
            std::vector<std::vector<double>> stats = experimentStats.getStatistics();
            std::cout << "Epoch #" << epoch
                      << " - Average: " << stats[0][0]
                      << " - Standard Deviation: " << stats[0][1]
                      << " - Sharpe ratio: " << stats[0][2] << std::endl;

            debugFile << epoch << "," << stats[0][0] << "," << stats[0][1]
                      << "," << stats[0][2] << ",\n";

            // Reset task and statistics
            task.reset();
            experimentStats.reset();
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
            blog.insertRecord(actionCache, rewardCache);
        }

        std::ofstream backtestFile;
        backtestFile.open("../../../Data/Debug/backtestExperiment1.csv" );
        blog.print(backtestFile);
        backtestFile.close();
    }
}
